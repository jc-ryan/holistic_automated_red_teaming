import os
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
        self.dialogue = PROMPT_BEGIN
        self.batch_dialogues = None
        self.red_team_attempts = read_json_lines(red_team_attempts_path)

    def generate(self, model, tokenizer, prompt):
        input = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=1.1
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]            
    
    def batch_generate(self, model, tokenizer, batch_prompts):
        input = tokenizer(
            batch_prompts, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False
        )
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=1.1
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
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
        for question in tqdm(self.red_team_attempts[:10]):
            result = {
                "question_id": question["id"],
                "category": question["category"],
                "attack_vector": question["attack_vector"],
                "descriptors": question["descriptors"]
            }
            dialogue = [question["question"]]
            self.dialogue += PROMPT_USER.format(input=question["question"])
            for turn_idx in range(num_turns):
                self.dialogue += PROMPT_ASSISTANT
                blue_corner_model_response = self.generate(
                    self.blue_corner_model, self.blue_corner_tokenizer, self.dialogue
                )
                self.dialogue = blue_corner_model_response
                dialogue.append(self.dialogue.rpartition(PROMPT_ASSISTANT)[-1])

                if turn_idx < num_turns - 1:
                    self.dialogue += "USER:"
                    red_corner_model_response = self.generate(
                        self.red_corner_model, self.red_corner_tokenizer, self.dialogue
                    )
                    self.dialogue = red_corner_model_response
                    dialogue.append(self.dialogue.rpartition("USER:")[-1])
            
            assert len(dialogue) % 2 == 0
            result["dialogue"] = dialogue
            test_results.append(result)
            self.clear()
        
        return test_results
    
    def test_suite_batch(self, num_turns=5, batch_size=8):
        print("*"*20 + " Starting red teaming test " + "*"*20)
        test_results = []
        for idx in tqdm(range(0, len(self.red_team_attempts), batch_size)):
            self.clear_batch_conversation(batch_size=batch_size)
            batch_questions = self.red_team_attempts[idx: idx+batch_size]
            batch_results = []
            for question in batch_questions:
                batch_results.append(
                    {
                        "question_id": question["id"],
                        "category": question["category"],
                        "attack_vector": question["attack_vector"],
                        "descriptors": question["descriptors"]
                    }
                )
            
            batch_dialogues = [[question["question"]] for question in batch_questions]
            self.batch_dialogues = [self.batch_dialogues[idx] + PROMPT_USER.format(input=question["question"]) for idx, question in enumerate(batch_questions)]
            
            for turn_idx in range(num_turns):
                self.batch_dialogues = [dialogue + PROMPT_ASSISTANT for dialogue in self.batch_dialogues]
                batch_blue_corner_model_responses = self.batch_generate(
                    self.blue_corner_model, self.blue_corner_tokenizer, self.batch_dialogues
                )
                self.batch_dialogues = batch_blue_corner_model_responses
                for idx, dialogue_list in enumerate(batch_dialogues):
                    dialogue_list.append(self.batch_dialogues[idx].rpartition(PROMPT_ASSISTANT)[-1])

                if turn_idx < num_turns - 1:
                    self.batch_dialogues = [dialogue + "USER:" for dialogue in self.batch_dialogues]
                    batch_red_corner_model_responses = self.batch_generate(
                        self.red_corner_model, self.red_corner_tokenizer, self.batch_dialogues
                    )
                    self.batch_dialogues = batch_red_corner_model_responses
                    for idx, dialogue_list in enumerate(batch_dialogues):
                        dialogue_list.append(self.batch_dialogues[idx].rpartition("USER:")[-1])
            
            for idx, dialogue in enumerate(batch_dialogues):
                assert len(dialogue) % 2 == 0
                batch_results[idx]["dialogue"] = dialogue
            
            test_results.extend(batch_results)
        
        return test_results

    def clear(self):
        self.dialogue = PROMPT_BEGIN
    
    def clear_batch_conversation(self, batch_size=8):
        self.batch_dialogues = [PROMPT_BEGIN for _ in range(batch_size)]


if __name__ == "__main__":
    red_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/red-teaming-sft-chat-v3"

    # blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/beaver-7b-v1.0"
    blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/sft-alpaca"
    
    red_team_attempts_path = "/YOUR_OWN_PATH/data/multi-turn_test_questions.json"

    red_team_test = RedTeamTest(red_corner_model_path, blue_corner_model_path, red_team_attempts_path)
    
    test_results = red_team_test.test_suite(num_turns=5)

    output_path = os.path.join("/YOUR_OWN_PATH/output", "red-teaming_with_{}.json".format(os.path.basename(blue_corner_model_path)))
 
    write_json_lines(test_results, output_path)
    