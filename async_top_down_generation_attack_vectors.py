import os
import re
import openai
import random
import backoff
import aiolimiter
import asyncio
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing import Any
from aiohttp import ClientSession
from openai import error
from utils import read_json, read_json_lines, append_json_lines, write_json_lines, read_txt

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


openai.api_key = os.environ.get("OPENAI_API_KEY")

ERROR_ERRORS_TO_MESSAGES = {
    error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    error.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    error.ServiceUnavailableError: "OpenAI service unavailable error: {e}",
    error.APIError: "OpenAI API error: {e}",
}

CATEGORY_FOLDER = "./risk_categories"


def load_files(category_folder=CATEGORY_FOLDER, category_index=None, attack_vector=None):
    """
    category_folder: str, path to the folder containing all risk categories
    category_index: int, index of the risk category to load (0-7)
    """
    if attack_vector is not None:
        prompt_file = os.path.join("attack_vectors", "{}.txt".format(attack_vector))
        output_file = "{}_questions.json".format(attack_vector)
    else:
        prompt_file = "prompt.txt"
        output_file = "generated_questions.json"
    
    root, risk_categories, _ = next(os.walk(category_folder))
    taxonomy_paths = [os.path.join(root, category, "taxonomy.json")
                      for category in risk_categories]
    prompt_template_paths = [os.path.join(
        root, category, prompt_file) for category in risk_categories]
    result_paths = [os.path.join(
        root, category, output_file) for category in risk_categories]
    seed_question_paths = []

    if attack_vector is None:
        for category in risk_categories:
            for file in os.listdir(os.path.join(root, category)):
                if file.endswith(".jsonl"):
                    seed_question_paths.append(os.path.join(root, category, file))
    else:
        for category in risk_categories:
            seed_question_paths.append(os.path.join(root, category, "attack_vectors", "{}.jsonl".format(attack_vector)))

    path_list = list(
        zip(seed_question_paths, taxonomy_paths, prompt_template_paths, result_paths))
    # print(path_list)

    seed_quesion_path, taxonomy_path, prompt_template_path, result_path = path_list[
        category_index]

    print(seed_quesion_path)
    red_team_attempts = read_json_lines(seed_quesion_path)
    taxonomy = read_json(taxonomy_path)
    prompt_template = read_txt(prompt_template_path)
    print(prompt_template_path)

    return red_team_attempts, taxonomy, prompt_template, result_path


async def _throttled_openai_chat_completion_acreate(
    model_name: str = "gpt-3.5-turbo",
    messages: list[dict[str, str]] = None,
    temperature: float = 1.2,
    limiter: aiolimiter.AsyncLimiter = None,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(
                        ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    prompts: list[str] = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 1.2,
    requests_per_minute: int = 1008611,
) -> list[list[str]]:

    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model_name=model_name,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    all_responses = []
    for x in responses:
        all_responses.append(x.choices[0].message["content"])
    return all_responses


def prompt_construction(examples, triples_to_infer, prompt_template):
    questions = [item["question"] for item in examples]
    descriptors = [item["descriptors"][0] for item in examples]

    triples_to_infer = [
        "<" + ", ".join(triple) + ">" for triple in triples_to_infer]
    descriptors = ["<" + ", ".join(descriptor) +
                   ">" for descriptor in descriptors]

    hints = descriptors + triples_to_infer
    hints_text = ""
    for idx, hint in enumerate(hints):
        hints_text += f"{idx + 1}. {hint}\n"

    questions_text = ""
    for idx, question in enumerate(questions):
        questions_text += f"{idx + 1}. {question}\n"

    questions_text += str(len(questions) + 1) + ". "
    # print(questions_text)

    prompt = prompt_template.format(hints_text, questions_text)
    # print(prompt)

    return prompt


def triple_distribution(red_team_attempts, triples):
    triple_instances_count = {t: 0 for t in triples}

    for item in red_team_attempts:
        for descriptor in item["descriptors"]:
            t = tuple(descriptor)
            if t in triple_instances_count:
                triple_instances_count[t] += 1

    return triple_instances_count


def adjust_sampling_probs(triple_instances_count):
    min_count = min(triple_instances_count.values())
    adjusted_probs = {}

    for triple, count in triple_instances_count.items():
        adjusted_probs[triple] = (min_count + 1) / (count + 1)

    total_adjusted_probs = sum(adjusted_probs.values())
    for triple, prob in adjusted_probs.items():
        adjusted_probs[triple] = prob / total_adjusted_probs

    return adjusted_probs


def sample_triples(triple_instances_count, sample_num=5):
    sampling_probs = adjust_sampling_probs(triple_instances_count)

    triples = list(sampling_probs.keys())
    triple_probs = list(sampling_probs.values())

    indices_sampled = np.random.choice(
        list(range(len(triples))), size=sample_num, p=triple_probs)
    triples_sampled = [triples[idx] for idx in indices_sampled]
    return triples_sampled


def convert_to_examples(response=None, demonstration_num=5, triples_sampled=None):
    """
    example: {"question": question, "descriptors": descriptors}
    """
    text = str(demonstration_num + 1) + ". " + response
    # print(text)
    # questions = re.findall(r'(?<=\d\.\s).*\?', text)
    questions = re.findall(r'(?<=\d\.\s).*[.?]', text)
    
    if len(questions) != demonstration_num:
        print(text)
        return []

    examples = []
    for q, t in zip(questions, triples_sampled):
        examples.append({"question": q, "descriptors": [t]})
    
    return examples


def top_down_generate(model_name="gpt-3.5-turbo", category_index=None, attack_vector=None, demonstration_num=5, iterations=2000, batch_size=10):
    red_team_attempts, taxonomy, prompt_template, result_path = load_files(
        category_index=category_index, attack_vector=attack_vector)

    if os.path.exists(result_path):
        generated_questions = read_json_lines(result_path)
    else:
        generated_questions = []

    triples = []
    for axe, buckets in taxonomy.items():
        for bucket, terms in buckets.items():
            for term in terms:
                triples.append((axe, bucket, term))

    triple_instances_count = triple_distribution(red_team_attempts, triples)
    if len(generated_questions) > 0:
        for item in generated_questions:
            for descriptor in item["descriptors"]:
                t = tuple(descriptor)
                if t in triple_instances_count:
                    triple_instances_count[t] += 1

    # print(triple_instances)
    print("orinal triple instances count: {}".format(
        sum(triple_instances_count.values())))
    # print(triple_instances_count)

    new_generated_questions = []

    for _ in tqdm(range(iterations)):
        examples = random.sample(
            red_team_attempts + generated_questions + new_generated_questions, demonstration_num * batch_size)
        triples_to_infer = sample_triples(
            triple_instances_count=triple_instances_count, sample_num=demonstration_num*batch_size)

        prompt_list = []
        for i in range(0, demonstration_num * batch_size, demonstration_num):
            prompt_list.append(prompt_construction(
                examples=examples[i:i+demonstration_num], triples_to_infer=triples_to_infer[i:i+demonstration_num], prompt_template=prompt_template))
        # print(prompt_list[0])
        assert len(prompt_list) == batch_size, "number of prompts is not correct"
        
        batch_responses = asyncio.run(
            generate_from_openai_chat_completion(model_name=model_name, prompts=prompt_list))

        assert len(batch_responses) == batch_size, "number of batch responses is not correct"
        
        generated_examples = []
        for i, response in enumerate(batch_responses):
            generated_examples.extend(convert_to_examples(
                response=response, demonstration_num=demonstration_num, triples_sampled=triples_to_infer[i*demonstration_num:(i+1)*demonstration_num]))

        new_generated_questions.extend(generated_examples)

        for item in generated_examples:
            for descriptor in item["descriptors"]:
                t = tuple(descriptor)
                if t in triple_instances_count:
                    triple_instances_count[t] += 1
        # for triple in triples_to_infer:
        #     triple_instances_count[triple] += 1

        if _ % 2 == 0:
            append_json_lines(new_generated_questions, result_path)
            generated_questions.extend(new_generated_questions)
            new_generated_questions = []

    print("final triple instances count: {}".format(
        sum(triple_instances_count.values())))

    append_json_lines(new_generated_questions, result_path)

    return new_generated_questions


if __name__ == "__main__":
    model_name = "gpt-3.5-turbo"
    # top_down_generate(model_name=model_name, category_index=1, iterations=2, batch_size=10)
    # top_down_generate(model_name=model_name, category_index=4, attack_vector="debating", iterations=5, batch_size=50)
    # top_down_generate(model_name=model_name, category_index=6, attack_vector="false_premise", iterations=5, batch_size=50)
    # top_down_generate(model_name=model_name, category_index=0, attack_vector="role_play_lm", iterations=5, batch_size=5)
    # top_down_generate(model_name=model_name, category_index=1, attack_vector="role_play", iterations=5, batch_size=30)
    # top_down_generate(model_name=model_name, category_index=0, attack_vector="implicit", iterations=5, batch_size=50)
    top_down_generate(model_name=model_name, category_index=4, attack_vector="implicit", iterations=5, batch_size=30)
    # top_down_generate(model_name=model_name, category_index=6, attack_vector="implicit", iterations=5, batch_size=50)
    # top_down_generate(model_name=model_name, category_index=0, attack_vector="realistic", iterations=20, batch_size=10)
