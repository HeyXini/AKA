import os
import torch
import torch.nn as nn
import json
import math
from transformers import (
    AutoTokenizer,
)
import pickle
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from typing import Dict, List
from transformers import AutoImageProcessor, ViTMAEForPreTraining

from transformers import ViTMAEConfig

processor = AutoImageProcessor.from_pretrained("./vit-mae-model")
mae_model = ViTMAEForPreTraining.from_pretrained("./vit-mae-model").eval().to("cuda")


def preprocess_caption(caption: str) -> str:
    import re

    caption = caption.lower()

    caption = re.sub(r'[;:"\'\(\)\[\]\{\}\,\-\_\=\+\*\/\\\|\<\>\?\#\@\$\%\^\&\~\`]', '', caption)

    caption = re.sub(r'\s+', ' ', caption)

    caption = caption.strip()

    patterns_to_remove = [
        r'^a photo of\s+',
        r'^this is a\s+',
        r'^this is an\s+'
    ]
    
    for pattern in patterns_to_remove:
        caption = re.sub(pattern, '', caption)
    
    return caption


def process_text(text: str, tokenizer: AutoTokenizer, max_length: int = 90) -> Dict:
    text_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )

    input_ids = text_inputs["input_ids"][0]

    valid_len = (input_ids != tokenizer.pad_token_id).sum().item()
    if valid_len == max_length:
        print(valid_len,text)

    return  input_ids



def process_flickr(data, data_dir):
    samples = []
    img_ori = ''
    for list_data in tqdm(data, desc="Processing Flickr Data"):
        img, cap = list_data
        if img_ori == img:
            samples.append({"image_vector": img_tensor.clone(), "caption": cap, "image_name":img})
        else:
            img_ori = img
            image = Image.open(os.path.join(data_dir, img)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = mae_model(**inputs, output_hidden_states=True)
            img_tensor_origin = outputs.hidden_states[-1]
            img_tensor = img_tensor_origin[0]

            samples.append({"image_vector": img_tensor.clone(), "caption": cap, "image_name":img})
    return samples


def patchify_image(image_tensor, patch_size=32):
    """
    image_tensor - [B, C, H, W]
    patches_flat - [B, N, C*P*P]
    """
    C, H, W = image_tensor.shape # 3，224，224
    # 切分非重叠补丁：[B, C, H, W] → [B, C, N_h, N_w, P, P]，N_h=H/P，N_w=W/P
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # 调整维度并展平：→ [B, N, C*P*P]，N=N_h*N_w（补丁总数）
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    N = patches.shape[0] * patches.shape[1]
    patches_flat = patches.view(N, C * patch_size * patch_size)
    return patches_flat

def create_dataset(image_data: List[Dict]) -> Dataset:
    """
    Args:
        image_data: [{"image_vector": np.array, "caption": str}, ...]
        text_data: [{"text": str, "response": str}, ...]
    """

    def dataset_generator():
        # 交替生成图像和文本样本
        max_len = len(image_data)
        for i in range(max_len):
            if i < len(image_data):
                yield {
                    "type": "image",
                    "input_ids": image_data[i]["image_vector"], #(196,768)
                    "attention_mask": torch.ones(len(image_data[i]["image_vector"]),dtype=torch.long), #(196,)
                    "label": process_text(image_data[i]["caption"], tokenizer), #(196,)
                    "image_name":image_data[i]["image_name"]
                }

    return Dataset.from_generator(dataset_generator)

model_name = "Qwen/Qwen2.5-7B"
flickr_dir = "Flickr8k_dataset/Flickr8k_images"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)
tokenizer.padding_side = "right"



# 存储数据
with open("dataset/flickr_train.json", "r", encoding="utf-8") as f:
    train_flickr = json.load(f)
with open("dataset/flickr_val.json", "r", encoding="utf-8") as f:
    val_flickr = json.load(f)
with open("dataset/flickr_test.json", "r", encoding="utf-8") as f:
    test_flickr = json.load(f)


train_flickr_format = process_flickr(train_flickr, flickr_dir)
val_flickr_format = process_flickr(val_flickr, flickr_dir)
test_flickr_format = process_flickr(test_flickr, flickr_dir)


print("创建数据集...")
train_dataset = create_dataset(train_flickr_format)
val_dataset = create_dataset(val_flickr_format)
test_dataset = create_dataset(test_flickr_format)



# 保存数据集
train_dataset.save_to_disk('train_dataset_flickr_random_vit_clean')
print("训练集已保存到 train_dataset_flickr_random_vit_clean")
val_dataset.save_to_disk('val_dataset_flickr_random_vit_clean')
print("验证集已保存到 val_dataset_flickr_random_vit_clean")
test_dataset.save_to_disk('test_dataset_flickr_random_vit_clean')
print("测试集已保存到 test_dataset_flickr_random_vit_clean")



