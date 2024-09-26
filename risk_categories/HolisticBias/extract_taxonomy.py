import json
from collections import defaultdict


def read_json(data_path):
    with open(data_path, encoding="utf-8") as fr:
        return json.load(fr)


def write_json(data, target_path):
    with open(target_path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)


def contains_dict(lst):
    """judge if there is at least one dict element in a list"""
    return any(isinstance(item, dict) for item in lst)


def extract_from_orgin(origin_descriptors):
    """discard extra fields like preference, remain pure descriptor terms"""
    taxonomy = dict()
    for axe, buckets in origin_descriptors.items():
        
        new_buckets = defaultdict(list)
        for bucket, descriptors in buckets.items():
            pure_descriptors = [item["descriptor"] if isinstance(item, dict) else item for item in descriptors]
            new_buckets[bucket] = pure_descriptors
        
        taxonomy[axe] = new_buckets
    
    return taxonomy


if __name__ == "__main__":
    origin_data = read_json("./HolisticBias/descriptors.json")
    new_taxonomy = extract_from_orgin(origin_data)
    write_json(new_taxonomy, "./HolisticBias/taxonomy.json")