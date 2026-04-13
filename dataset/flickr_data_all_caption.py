

import json
import torch

flickr_dir = "../Flickr8k_dataset/Flickr8k_images"
flickr_token = "../Flickr8k_dataset/Flickr8k.token.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_flickr(token_file):
    captions = []
    with open(token_file, 'r') as f:
        for line in f:
            img_id, cap = line.strip().split('\t')
            img = img_id.split('#')[0]
            captions.append([img, cap])
    return captions



def load_flickr_split(caption_dict, split_file):
    with open(split_file, 'r') as f:
        image_names = {line.strip() for line in f.readlines()}
    split_data = [item for item in caption_dict if item[0] in image_names]
    
    return split_data


flickr_caps = load_flickr(flickr_token)

train_flickr = load_flickr_split(
    flickr_caps,
    "../Flickr8k_dataset/Flickr_8k.trainImages.txt"
)
val_flickr = load_flickr_split(
    flickr_caps,
    "../Flickr8k_dataset/Flickr_8k.devImages.txt"
)
test_flickr = load_flickr_split(
    flickr_caps,
    "../Flickr8k_dataset/Flickr_8k.testImages.txt"
)


with open("flickr_train.json", "w", encoding="utf-8") as f:
    json.dump(train_flickr, f)
with open("flickr_val.json", "w", encoding="utf-8") as f:
    json.dump(val_flickr, f)
with open("flickr_test.json", "w", encoding="utf-8") as f:
    json.dump(test_flickr, f)


