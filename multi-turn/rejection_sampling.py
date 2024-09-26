import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from safe_rlhf.models import load_pretrained_models, AutoModelForScore
from safe_rlhf.utils import is_same_tokenizer
from safe_rlhf.configs import PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER

BEST_OF_N = 8


def read_json_lines(data_path):
    with open(data_path, 'r', encoding="utf-8") as fr:
        return [json.loads(item) for item in fr.readlines()]


def write_json_lines(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        for item in data:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_model_werights(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype='auto')
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, model_max_length=2048, padding_side="left"
    )
    return model, tokenizer


class RedTeamingRejectionSampling(object):
    def __init__(self, red_corner_model_name, blue_corner_model_name, reward_model_name,
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
        self.reward_model = AutoModelForScore.from_pretrained(reward_model_name, device_map='auto', torch_dtype='auto')
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, use_fast=False)
        
        print("Model Loaded.")
        
        self.dialogue = PROMPT_BEGIN
        self.red_team_attempts = read_json_lines(red_team_attempts_path)

    def generate(self, model, tokenizer, prompt, temperature=1.1):
        input = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=temperature
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def red_corner_generate(self, model, tokenizer, prompt, temperature=1.1):
        """
        Gererate N questions from the red corner model
        """
        input = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=temperature, num_return_sequences=BEST_OF_N
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)            
    
    def blue_corner_generate(self, model, tokenizer, batch_prompts):
        input = tokenizer(
            batch_prompts, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False
        )
        input_ids= input.input_ids.to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids, max_new_tokens=300, do_sample=True, temperature=1.1
            )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_dialogue(self, text: str) -> list:
        """Extract dialogue from the text based on known patterns."""
        dialogue = text.split("USER:")[1:]
        multi_turn = []
        for item in dialogue:
            multi_turn.extend(item.split("ASSISTANT:"))
    
        dialogue = [item.strip() for item in multi_turn if item.strip()]
        return dialogue
    
    def convert_dialogue_to_reward_model_format(self, dialogue):
        dialogue = self.extract_dialogue(dialogue)
        if len(dialogue) % 2 != 0:
            return None
            # raise ValueError(
            #     'The length of `input` must be even, while `input` must end at the assistant response.'
            # )
        else:
            buffer = [PROMPT_BEGIN]
            for i, line in enumerate(dialogue):
                if i % 2 == 0:
                    # User input
                    buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
                else:
                    # Assistant response
                    buffer.extend((line, self.reward_model_tokenizer.eos_token))

        return "".join(buffer)
    
    def batch_reward_computation(self, batch_dialogues):
        """
        batch_dialogues: list of formatted dialogues ['BEGIN of CONVERSATIONS : USER: ASSISTANT: <eos>', ...]
        return: list of safety_score
        """
        dropped_answers = 0
        filtered_dialogues = []
        converted_batch_dialogues = []
        for dialogue in batch_dialogues:
            converted_dialogue = self.convert_dialogue_to_reward_model_format(dialogue)
            if converted_dialogue is None:
                dropped_answers += 1
            else:
                filtered_dialogues.append(dialogue)
                converted_batch_dialogues.append(converted_dialogue)
        
        if dropped_answers > 0:
            print("Dropped {} answers".format(dropped_answers))

        assert len(converted_batch_dialogues) == BEST_OF_N - dropped_answers
        
        # batch_dialogues = [self.convert_dialogue_to_reward_model_format(dialogue) for dialogue in batch_dialogues]
        input_ids = self.reward_model_tokenizer(converted_batch_dialogues, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_ids.to("cuda")
        with torch.no_grad():
            batch_outputs = self.reward_model(**input_ids)
            reward_scores = batch_outputs.end_scores.squeeze(-1).tolist()
    
        return filtered_dialogues, reward_scores

    def rejection_sampling(self):
        print("*"*20 + " Starting rejection sampling " + "*"*20)
        sampling_results = []
        for question in tqdm(self.red_team_attempts):
            result = {
                "question_id": question["id"],
                "category": question["category"],
                "attack_vector": question["attack_vector"],
                "descriptors": question["descriptors"]
            }
            dialogue = [question["question"]]
            self.dialogue += PROMPT_USER.format(input=question["question"])
            red_corner_model_questions = None
            all_dialogues = []
            reward_scores = []
            min_reward_scores = []

            num_turns = random.choice([3, 4, 5])            
            for turn_idx in range(num_turns):                
                if turn_idx == 0:
                    self.dialogue += PROMPT_ASSISTANT
                    blue_corner_model_response = self.generate(
                        self.blue_corner_model, self.blue_corner_tokenizer, self.dialogue
                    )
                    self.dialogue = blue_corner_model_response
                    dialogue.append(self.dialogue.rpartition(PROMPT_ASSISTANT)[-1])
                else:
                    # best_question = None
                    batch_prompts = [q + PROMPT_ASSISTANT for q in red_corner_model_questions]
                    batch_responses = self.blue_corner_generate(
                        self.blue_corner_model, self.blue_corner_tokenizer, batch_prompts
                    )
                    # Some responses that were not generated in the correct format were excluded from the reward calculation
                    filtered_responses, batch_reward_scores = self.batch_reward_computation(batch_responses)
                    best_question_idx = np.argmin(batch_reward_scores)
                    self.dialogue = filtered_responses[best_question_idx]
                    dialogue.extend(self.extract_dialogue(self.dialogue)[-2:])
                    all_dialogues.append([self.extract_dialogue(r)[-2:] for r in filtered_responses])
                    reward_scores.append(batch_reward_scores)
                    min_reward_scores.append(batch_reward_scores[best_question_idx])

                if turn_idx < num_turns - 1:
                    self.dialogue += "USER:"
                    red_corner_model_questions = self.red_corner_generate(
                        self.red_corner_model, self.red_corner_tokenizer, self.dialogue, temperature=random.choice([1.2, 1.3])
                    )
            
            assert len(dialogue) % 2 == 0
            result["dialogue"] = dialogue
            result["all_trajectories"] = all_dialogues
            result["reward_scores"] = reward_scores
            result["min_reward_scores"] = min_reward_scores
            sampling_results.append(result)
            self.clear()
        
        return sampling_results

    def clear(self):
        self.dialogue = PROMPT_BEGIN


if __name__ == "__main__":
    red_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/red-teaming-sft-chat-v3"

    blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/beaver-7b-v1.0"
    # blue_corner_model_path = "/YOUR_OWN_PATH/local_model_weights/vicuna-7b-v1.5"
    

    safety_reward_model_path = "/YOUR_OWN_PATH/local_model_weights/safety-rm-3.0"

    rejection_sampling_data_path = "/YOUR_OWN_PATH/data/rejection_sampling/rejection_sampling_for_{}.json".format(os.path.basename(blue_corner_model_path))

    red_team_test = RedTeamingRejectionSampling(
        red_corner_model_path, blue_corner_model_path, safety_reward_model_path, rejection_sampling_data_path
    )
    
    # test_results = red_team_test.test_suite(num_turns=5)
    test_results = red_team_test.rejection_sampling()

    output_path = os.path.join("/YOUR_OWN_PATH/output", "rejection_sampling_with_{}.json".format(os.path.basename(blue_corner_model_path)))
    
    write_json_lines(test_results, output_path)
    
