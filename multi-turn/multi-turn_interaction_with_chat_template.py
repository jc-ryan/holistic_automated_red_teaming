import os
import re
import json
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from safe_rlhf.models import load_pretrained_models, AutoModelForScore
from safe_rlhf.utils import is_same_tokenizer
from safe_rlhf.configs import PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER


def read_json_lines(data_path):
    with open(data_path, 'r', encoding="utf-8") as fr:
        return [json.loads(item) for item in fr.readlines()]


def write_json_lines(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        for item in data:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_with_red_model_chat_template(dialogue):
    """
    dialogue: list of strings
    """
    if len(dialogue) % 2 != 0:
        raise ValueError(
            'The length of input to red model must be even, which must end at the assistant response.'
        )
    else:
        buffer = [PROMPT_BEGIN]
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
            else:
                # Assistant response
                buffer.append(line)
        
        buffer.append("USER:")
    
    return "".join(buffer)


def format_with_blue_model_chat_template(tokenizer, dialogue):
    """
    dialogue: list of strings
    """
    if len(dialogue) % 2 != 1:
        raise ValueError(
            'The length of input to blue model must be odd,which must end at the user response.'
        )
    else:
        messages = []
        for i, line in enumerate(dialogue):
            if i % 2 == 0:
                # User input
                messages.append({"role": "user", "content": line})
            else:
                # Assistant response
                messages.append({"role": "assistant", "content": line})
    
    return tokenizer.apply_chat_template(messages, tokenize=False)


def extract_last_response(text: str) -> str:
    """Extract the last response from the dialogue text based on known patterns."""
    if not text.endswith("</s>"):
        text += "</s>"

    last_response = ""

    # extract the last response (content between the last [/INST] and </s>)) 
    if "[/INST]" in text:
        last_response = text.rpartition("[/INST]")[-1].split("</s>")[0].strip()
    
    # extract the last response (content between the last <|assistant|> and </s>)) 
    if "<|assistant|>" in text:
        last_response = text.rpartition("<|assistant|>")[-1].split("</s>")[0].strip()
    
    # If neither pattern is found, return empty string
    return last_response


class RedTeamTest(object):
    def __init__(self, red_corner_model_name, blue_corner_model_name, 
            red_team_attempts_path, max_length=4096) -> None:
        """Agent (red/blue corner model) initialization"""
        print("Loading Models..")
        self.red_corner_model, self.red_corner_tokenizer = load_pretrained_models(
            red_corner_model_name, model_max_length=max_length, padding_side="left",
            auto_device_mapping=True, trust_remote_code=True
        )
        self.blue_corner_model, self.blue_corner_tokenizer = load_pretrained_models(
            blue_corner_model_name, model_max_length=max_length, padding_side="left",
            auto_device_mapping=True, trust_remote_code=True
        )
        print("Model Loaded.")
        self.dialogue = []
        self.red_team_attempts = read_json_lines(red_team_attempts_path)

    def generate(self, model, tokenizer, prompt):
        input = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=1.1
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]            
    
    def single_case_test(self, num_turns=3):
        print("*"*20 + " Starting red teaming test " + "*"*20)
        red_team_question = random.choice(self.red_team_attempts)["question"]
        self.dialogue += PROMPT_USER.format(input=red_team_question)
        print(self.dialogue)
        for turn_idx in range(num_turns):
            self.dialogue += PROMPT_ASSISTANT
            blue_corner_model_response = self.generate(
                self.blue_corner_model, self.blue_corner_tokenizer, self.dialogue
            )
            self.dialogue = blue_corner_model_response
            print("ASSISTANT:")
            print(self.dialogue.rpartition(PROMPT_ASSISTANT)[-1])

            if turn_idx < num_turns - 1:
                self.dialogue += "USER:"
                red_corner_model_response = self.generate(
                    self.red_corner_model, self.red_corner_tokenizer, self.dialogue
                )
                self.dialogue = red_corner_model_response
                print("USER:")
                print(self.dialogue.rpartition("USER:")[-1])
    
    def test_suite(self, num_turns=5):
        print("*"*20 + " Starting red teaming test " + "*"*20)
        test_results = []
        for question in tqdm(self.red_team_attempts):
            result = {
                "question_id": question["id"],
                "category": question["category"],
                "attack_vector": question["attack_vector"],
                "descriptors": question["descriptors"]
            }
            self.dialogue = [question["question"]]
            # self.dialogue += PROMPT_USER.format(input=question["question"])
            for turn_idx in range(num_turns):
                blue_corner_model_response = self.generate(
                    self.blue_corner_model, self.blue_corner_tokenizer, format_with_blue_model_chat_template(self.blue_corner_tokenizer, self.dialogue) 
                )
                self.dialogue.append(extract_last_response(blue_corner_model_response))

                if turn_idx < num_turns - 1:
                    red_corner_model_response = self.generate(
                        self.red_corner_model, self.red_corner_tokenizer, format_with_red_model_chat_template(self.dialogue)
                    )
                    self.dialogue.append(red_corner_model_response.rpartition("USER:")[-1])
            
            assert len(self.dialogue) % 2 == 0
            result["dialogue"] = self.dialogue
            test_results.append(result)
            self.clear()
        
        return test_results

    def clear(self):
        self.dialogue = []


if __name__ == "__main__":

    red_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/red-teaming-sft-chat-v3"

    # blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/Llama-2-7b-chat-hf"
    blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/zephyr-7b-beta"
    # blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/Mistral-7B-Instruct-v0.1"

    
    red_team_attempts_path = "/YOUR_OWN_PATH/data/multi-turn_test_questions.json"

    red_team_test = RedTeamTest(red_corner_model_path, blue_corner_model_path, red_team_attempts_path)
    
    test_results = red_team_test.test_suite(num_turns=5)

    output_path = os.path.join("/YOUR_OWN_PATH/output", "red-teaming_with_{}.json".format(os.path.basename(blue_corner_model_path)))

    write_json_lines(test_results, output_path)

