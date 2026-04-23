import json
import os
import random
from collections import defaultdict, Counter
from tqdm import tqdm



def show_top_question_types(samples, topk=20):
    counter = Counter(s["question_type"] for s in samples)
    total = len(samples)

    print(f"Total samples: {total}")
    print(f"Unique question types: {len(counter)}")
    print(f"Top {topk} question types:")
    for qtype, cnt in counter.most_common(topk):
        print(f"{qtype}: {cnt} ({cnt / total:.2%})")


def load_vqa_split(question_file, annotation_file, image_prefix):
    
    with open(question_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    with open(annotation_file, "r", encoding="utf-8") as f:
        a_data = json.load(f)

    questions = q_data["questions"]
    annotations = a_data["annotations"]

    qid_to_question = {
        q["question_id"]: {
            "image_id": q["image_id"],
            "question": q["question"]
        }
        for q in tqdm(questions)
    }

    samples = []
    for ann in tqdm(annotations):
        qid = ann["question_id"]
        if qid not in qid_to_question:
            continue
        answers = [a["answer"] for a in ann["answers"]]
        q_item = qid_to_question[qid]
        image_id = q_item["image_id"]
        image_name = f"{image_prefix}_{image_id:012d}.jpg"

        sample = {
            "question_id": qid,
            "image_id": image_id,
            "image": image_name,
            "question": q_item["question"],
            "answer": ann["multiple_choice_answer"],
            "all_answers": answers,
            "answer_type": ann.get("answer_type", "other"),
            "question_type": ann.get("question_type", "unknown")
        }
        samples.append(sample)

    return samples


def stratified_sample(samples, n, seed=42):
    random.seed(seed)
    groups = defaultdict(list)

    for s in tqdm(samples):
        groups[s["answer_type"]].append(s)

    total = len(samples)
    sampled = []

    for answer_type, group in tqdm(groups.items()):
        k = max(1, round(n * len(group) / total))
        k = min(k, len(group))
        sampled.extend(random.sample(group, k))

    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        used_image_ids = {x["image_id"] for x in sampled}
        remaining = [s for s in samples if s["image_id"] not in used_image_ids]
        extra = random.sample(remaining, min(n - len(sampled), len(remaining)))
        sampled.extend(extra)

    return sampled


def remove_overlap(source_samples, used_samples):
    used_image_ids = {x["image_id"] for x in used_samples}
    return [s for s in source_samples if s["image_id"] not in used_image_ids]


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    random_seed = 42

    train_question_file = "./coco/questions/v2_OpenEnded_mscoco_train2014_questions.json"
    train_annotation_file = "./coco/Annotations/v2_mscoco_train2014_annotations.json"

    val_question_file = "./coco/questions/v2_OpenEnded_mscoco_val2014_questions.json"
    val_annotation_file = "./coco/Annotations/v2_mscoco_val2014_annotations.json"

    output_dir = "./coco/vqav2_subsets"
    os.makedirs(output_dir, exist_ok=True)

    train_samples = load_vqa_split(
        train_question_file,
        train_annotation_file,
        image_prefix="COCO_train2014"
    )
    print(f"Full train samples: {len(train_samples)}")
    train_subset = stratified_sample(train_samples, n=20000, seed=random_seed)
    save_json(train_subset, os.path.join(output_dir, "vqav2_train_20k.json"))
    print("Saved:")
    print(os.path.join(output_dir, "vqav2_train_20k.json"))

    val_samples = load_vqa_split(
        val_question_file,
        val_annotation_file,
        image_prefix="COCO_val2014"
    )
    print(f"Full val samples: {len(val_samples)}")
    val_subset = stratified_sample(val_samples, n=4000, seed=random_seed)
    save_json(val_subset, os.path.join(output_dir, "vqav2_val_4k.json"))
    print(os.path.join(output_dir, "vqav2_val_4k.json"))
    # test
    val_remaining = remove_overlap(val_samples, val_subset)
    test_subset = stratified_sample(val_remaining, n=6000, seed=random_seed + 1)
    save_json(test_subset, os.path.join(output_dir, "vqav2_test_6k.json"))
    print(os.path.join(output_dir, "vqav2_test_6k.json"))


    show_top_question_types(train_subset, topk=20)
    show_top_question_types(val_subset, topk=20)
    show_top_question_types(test_subset, topk=20)
