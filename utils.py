import json


def read_json(data_path):
    with open(data_path, encoding="utf-8") as fr:
        return json.load(fr)


def write_json(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)


def read_json_lines(data_path):
    with open(data_path, 'r', encoding="utf-8") as fr:
        return [json.loads(item) for item in fr.readlines()]


def write_json_lines(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        for item in data:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_json_lines(data, target_path):
    with open(target_path, "a", encoding="utf-8") as fw:
        for item in data:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_txt(data_path):
    with open(data_path, "r", encoding="utf-8") as fr:
        return "".join(fr.readlines()).strip()