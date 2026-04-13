import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel, ViTMAEConfig
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Dict, List

processor = AutoImageProcessor.from_pretrained("./vit-mae-model")
config = ViTMAEConfig.from_pretrained("./vit-mae-model")
config.mask_ratio = 0.0
mae_model = ViTMAEModel.from_pretrained("./vit-mae-model", config=config).eval().to("cuda")

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"


def process_vqa_sample(sample: Dict, tokenizer: AutoTokenizer, mae_model, processor, image_dir: str, max_length: int = 64) -> Dict:
    try:
        image_name = sample["image"]
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            return None
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = mae_model(**inputs, output_hidden_states=True)
            image_features = outputs.hidden_states[-1][0][1:]
        
        question = sample["question"]
        answer = sample["answer"]

        question_prefix = f"Question: {question} Answer: "

        prefix_ids = tokenizer(question_prefix, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        qa_ids = prefix_ids + answer_ids
        answer_start_idx = len(prefix_ids)

        qa_ids = qa_ids[:max_length]
        qa_attention_mask = [1] * len(qa_ids)

        pad_len = max_length - len(qa_ids)
        if pad_len > 0:
            qa_ids = qa_ids + [tokenizer.pad_token_id] * pad_len
            qa_attention_mask = qa_attention_mask + [0] * pad_len
        
        # 确定答案类型
        answer_lower = answer.lower().strip()
        if answer_lower in ['yes', 'no']:
            answer_type = 0  # yes/no
        elif answer_lower.isdigit() or (answer_lower.replace('.', '').replace('-', '').isdigit()):
            answer_type = 1  # number
        else:
            answer_type = 2  # other
        
        return {
            "input_ids": image_features,  # 图像特征 (tensor)
            "attention_mask": torch.ones(len(image_features), dtype=torch.long),  # 图像attention mask
            "qa_ids": torch.tensor(qa_ids),  # 问题+答案的token ids
            "qa_attention_mask": torch.tensor(qa_attention_mask),  # 问题+答案的attention mask
            "answer_start_idx": answer_start_idx,
            "answer_type": answer_type,
            "image_name": image_name,
            "question": question,
            "answer": answer,
            "image_id": sample["image_id"],
            "all_answers": sample.get("all_answers", [])
        }
    except Exception as e:
        print(f"处理样本时出错: {e}")
        return None


def process_vqa_data(data: List[Dict], tokenizer: AutoTokenizer, mae_model, processor, image_dir: str, max_length: int = 64) -> List[Dict]:
    """处理VQA数据"""
    processed_data = []
    for sample in tqdm(data, desc="Processing VQA Data"):
        processed = process_vqa_sample(sample, tokenizer, mae_model, processor, image_dir, max_length)
        if processed is not None:
            processed_data.append(processed)
    return processed_data


def create_vqa_dataset(processed_data: List[Dict]) -> Dataset:
    """创建VQA数据集"""
    def dataset_generator():
        for item in processed_data:
            yield {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "qa_ids": item["qa_ids"],
                "qa_attention_mask": item["qa_attention_mask"],
                "answer_start_idx": item["answer_start_idx"],
                "answer_type": item["answer_type"],
                "image_name": item["image_name"],
                "question": item["question"],
                "answer": item["answer"],
                "image_id": item["image_id"],
                "all_answers": item["all_answers"],
            }
    
    return Dataset.from_generator(dataset_generator)


def main():
    # 加载数据
    COCO_PATH = "./coco"
    
    print("加载数据集...")
    with open("./coco/vqav2_subsets/vqav2_train_20k.json", 'r', encoding='utf-8') as f:
        train_subset = json.load(f)
    
    with open("./coco/vqav2_subsets/vqav2_val_4k.json", 'r', encoding='utf-8') as f:
        val_subset = json.load(f)
    
    with open("./coco/vqav2_subsets/vqav2_test_6k.json", 'r', encoding='utf-8') as f:
        test_subset = json.load(f)

    
    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    print(f"测试集大小: {len(test_subset)}")
    
    # 处理数据
    print("\n处理训练集...")
    image_dir = os.path.join(COCO_PATH, "train2014")
    train_processed = process_vqa_data(train_subset, tokenizer, mae_model, processor, image_dir)
    print(f"训练集处理完成，有效样本数: {len(train_processed)}")
    
    print("处理验证集...")
    image_dir_val = os.path.join(COCO_PATH, "val2014")
    val_processed = process_vqa_data(val_subset, tokenizer, mae_model, processor, image_dir_val)
    print(f"验证集处理完成，有效样本数: {len(val_processed)}")
    
    print("处理测试集...")
    test_processed = process_vqa_data(test_subset, tokenizer, mae_model, processor, image_dir_val)
    print(f"测试集处理完成，有效样本数: {len(test_processed)}")
    
    print("\n创建数据集...")
    train_dataset = create_vqa_dataset(train_processed)
    val_dataset = create_vqa_dataset(val_processed)
    test_dataset = create_vqa_dataset(test_processed)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    print("\n保存数据集...")
    train_dataset.save_to_disk('train_dataset_vqa_vit_clean_20k')
    print("训练集已保存到 train_dataset_vqa_vit_clean")
    
    val_dataset.save_to_disk('val_dataset_vqa_vit_clean_4k')
    print("验证集已保存到 val_dataset_vqa_vit_clean")
    
    test_dataset.save_to_disk('test_dataset_vqa_vit_clean_6k')
    print("测试集已保存到 test_dataset_vqa_vit_clean")


if __name__ == "__main__":
    main()