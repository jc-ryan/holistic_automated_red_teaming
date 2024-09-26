import os
import json
from collections import defaultdict


ATTACK_VECTORS = ["realistic", "implicit", "role_play", "role_play_lm", "false_premise", "debating"]


def read_json_lines(data_path):
    with open(data_path, 'r', encoding="utf-8") as fr:
        return [json.loads(item) for item in fr.readlines()]


def write_json_lines(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        for item in data:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def filter_unexisted_path(attack_vector_question_paths, attack_vector_seed_question_paths, risk_category_attack_vectors):
    new_risk_category_attack_vectors = []
    new_attack_vector_question_paths = []
    new_attack_vector_seed_question_paths = []

    for category_vector, question_path, seed_question_path in zip(risk_category_attack_vectors, attack_vector_question_paths, attack_vector_seed_question_paths):
        if os.path.exists(question_path) and os.path.exists(seed_question_path):
            new_risk_category_attack_vectors.append(category_vector)
            new_attack_vector_question_paths.append(question_path)
            new_attack_vector_seed_question_paths.append(seed_question_path)

    return new_risk_category_attack_vectors, new_attack_vector_seed_question_paths, new_attack_vector_question_paths


def display_question_num(question_dict):
    for category, questions in question_dict.items():
        print(category, len(questions))


def deduplicate_questions(questions):
    deduplicated_questions = []
    for question in questions:
        if question not in deduplicated_questions and len(question.keys()) == 4:
            deduplicated_questions.append(question)

    return deduplicated_questions


def question_distribution(category_folder="./risk_categories"):
    """question number distribution"""
    root, risk_categories, _ = next(os.walk(category_folder))
    # print(risk_categories)
    generated_question_paths = []
    attack_vector_seed_question_paths = []
    attack_vector_question_paths = []

    risk_category_attack_vectors = []

    for category in risk_categories:
        generated_question_paths.append(os.path.join(root, category, "generated_questions.json"))
        for attack_vector in ATTACK_VECTORS:
            risk_category_attack_vectors.append((category, attack_vector))
            attack_vector_seed_question_paths.append(os.path.join(root, category, "attack_vectors", "{}.jsonl".format(attack_vector)))
            attack_vector_question_paths.append(os.path.join(root, category, "{}_questions.json".format(attack_vector)))

    # print(len(risk_category_attack_vectors))
    # print(len(attack_vector_question_paths))
    # print(len(attack_vector_seed_question_paths))    

    risk_category_attack_vectors, attack_vector_seed_question_paths, attack_vector_question_paths = filter_unexisted_path(attack_vector_question_paths, attack_vector_seed_question_paths, risk_category_attack_vectors)

    generated_questions = {category: read_json_lines(path) for category, path in zip(risk_categories, generated_question_paths)}
    attack_vector_questions = {category_vector: read_json_lines(path) for category_vector, path in zip(risk_category_attack_vectors, attack_vector_question_paths)}
    
    total_generated_question_num(generated_questions, attack_vector_questions)


def total_generated_question_num(generated_questions, attack_vector_questions):
    category_question_num = defaultdict(int)
    for category, questions in generated_questions.items():
        category_question_num[category] += len(questions)
    
    for category_vector, questions in attack_vector_questions.items():
        category, _ = category_vector
        category_question_num[category] += len(questions)
    
    print("total generated questions: ", sum(category_question_num.values()))
    for category, num in category_question_num.items():
        print(category, num)


if __name__ == "__main__":
    question_distribution()