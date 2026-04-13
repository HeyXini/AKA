import os
import json

def process_flickr30k(data_dir):
    captions_file = os.path.join(data_dir, 'captions.txt')
    image_captions = {}
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        next(f)  
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            image_name, caption = parts
            
            if caption.startswith('"') and caption.endswith('"'):
                caption = caption[1:-1]
            
            caption = caption.strip()
            
            if image_name not in image_captions:
                image_captions[image_name] = []
            image_captions[image_name].append(caption)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_file = os.path.join(data_dir, f'{split}.txt')
        output_file =  f'flickr30k_{split}.json'

        with open(split_file, 'r', encoding='utf-8') as f:
            image_ids = [line.strip() for line in f]
        
        output_data = []
        for image_id in image_ids:
            image_name = f'{image_id}.jpg'
            if image_name in image_captions:
                captions = image_captions[image_name]
                for caption in captions:
                    output_data.append([image_name, caption])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False)
        
        print(f'Processed {split} split: {len(output_data)} captions for {len(image_ids)} images')

if __name__ == '__main__':
    data_dir = '../Flickr30k'
    # output_dir = ''
    
    # os.makedirs(output_dir, exist_ok=True)
    
    # process_flickr30k(data_dir, output_dir)
    process_flickr30k(data_dir)
