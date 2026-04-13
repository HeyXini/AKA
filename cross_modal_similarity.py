# -*- coding: utf-8 -*-
# Improved Cross-Modal Feature Similarity Quantification Experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_from_disk
from peft import PeftModel

import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]

if 'Times New Roman' in available_fonts:
    plt.rcParams['font.family'] = ['Times New Roman']
    print("使用字体: Times New Roman")
else:
    plt.rcParams['font.family'] = ['DejaVu Sans']
    print("警告: 未找到Times New Roman字体，将使用默认字体")

    
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import gaussian_kde

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultimodalModel(nn.Module):
    """Multimodal model supporting image and text inputs"""

    def __init__(self, model, tokenizer=None):
        super().__init__()
        self.lm = model
        self.tokenizer = tokenizer
        self.device = DEVICE

        hidden_dim = model.config.hidden_size

        self.image_proj = nn.Linear(768, hidden_dim).to(self.device, dtype=model.dtype)
        self.caption_start = nn.Parameter(torch.randn(1, 1, hidden_dim, device=self.device, dtype=model.dtype))

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                att_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None,
                caption: Optional[torch.Tensor] = None,
                ):
        image_embeds = self.image_proj(input_ids)
        B, H, D = image_embeds.shape

        if hasattr(self.lm, 'model') and hasattr(self.lm.model, 'model'):
            caption_embeds = self.lm.model.model.embed_tokens(caption)
        elif hasattr(self.lm, 'model'):
            caption_embeds = self.lm.model.embed_tokens(caption)
        else:
            caption_embeds = self.lm.embed_tokens(caption)

        inputs_embeds = torch.cat((image_embeds, self.caption_start.expand(B, -1, -1), caption_embeds), dim=1)

        labels = torch.cat([
            torch.full((B, H + 1), -100, device=DEVICE),
            labels,
        ], dim=1)

        attention_mask = torch.cat([att_mask, torch.ones((B, 1), device=DEVICE, dtype=torch.long), mask2], dim=1)

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=self.lm.config.use_cache
        )


    def get_image_ffn_features(self, image_inputs, return_all_layers=False, return_activation=False, pooling_strategy='mean'):
        """Get image FFN features with support for multiple pooling strategies
        
        Args:
            image_inputs: Image inputs
            return_all_layers: Whether to return features from all layers
            return_activation: Whether to return features after activation layer
            pooling_strategy: Pooling strategy ('mean', 'max', 'cls', 'attention')
            
        Returns:
            If return_all_layers is True, returns a list of features from each layer
            Otherwise returns features from the last layer
        """
        with torch.no_grad():
            input_ids = image_inputs.to(self.device)
            B, H, D = input_ids.shape
            
            model_dtype = next(self.lm.parameters()).dtype
            input_ids_dtype = input_ids.to(model_dtype)
            image_embeds = self.image_proj(input_ids_dtype)
            
            caption_start = self.caption_start.expand(B, -1, -1).to(model_dtype)
            inputs_embeds = torch.cat([image_embeds, caption_start], dim=1)
            
            seq_length = inputs_embeds.shape[1]
            attention_mask = torch.ones((B, seq_length), device=self.device, dtype=model_dtype)

            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states
            layer_features = []

            transformer = self.lm.model.model
            layers = transformer.layers

            for i, layer in enumerate(layers):
                if return_activation:
                    normed = layer.input_layernorm(hidden_states[i])

                    gate = layer.mlp.gate_proj(normed)
                    up = layer.mlp.up_proj(normed)
                    activation_output = torch.nn.functional.silu(gate) * up
                    layer_features.append(activation_output)
                else:
                    normed = layer.input_layernorm(hidden_states[i])

                    ffn_output = layer.mlp(normed)
                    layer_features.append(ffn_output)

            pooled_features = []
            for feature in layer_features:
                if pooling_strategy == 'mean':
                    pooled = torch.mean(feature, dim=1)
                elif pooling_strategy == 'max':
                    pooled = torch.max(feature, dim=1)[0]
                elif pooling_strategy == 'cls':
                    pooled = feature[:, 0, :]
                elif pooling_strategy == 'attention':
                    attention_weights = torch.softmax(feature.mean(dim=-1), dim=1)
                    attention_weights = attention_weights.unsqueeze(-1)
                    pooled = torch.sum(feature * attention_weights, dim=1)
                else:
                    pooled = torch.mean(feature, dim=1)
                
                pooled_features.append(pooled)
            
            if return_all_layers:
                return pooled_features
            else:
                return pooled_features[-1]





def compute_cosine_similarity(features1, features2):
    features1_norm = F.normalize(features1, dim=-1)
    features2_norm = F.normalize(features2, dim=-1)
    similarity = torch.sum(features1_norm * features2_norm, dim=-1)
    return similarity


def load_model(model_path, base_model="Qwen/Qwen2.5-0.5B", use_quantized=True):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if use_quantized else None

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        model,
        model_path,
    )

    multimodal_model = MultimodalModel(model, tokenizer)

    custom_layers_path = os.path.join(model_path, "custom_layers.bin")
    custom_weights = torch.load(custom_layers_path, map_location=DEVICE)
    multimodal_model.image_proj.load_state_dict(custom_weights["image_proj"])
    multimodal_model.caption_start.data = custom_weights["caption_start"]

    multimodal_model.eval()

    original_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    original_model.eval()

    return multimodal_model, tokenizer, original_model


def plot_similarity_distribution(similarities, random_similarities, other_image_similarities, layer_name, save_path):
    plt.figure(figsize=(10, 6))
    
    plt.rcParams.update({'font.size': 12})
    
    all_data = similarities + random_similarities
    
    if len(all_data) > 1:
        x_range = np.linspace(min(all_data) - 0.05, max(all_data) + 0.05, 200)
    else:
        x_range = np.linspace(-1, 1, 200)
    
    if len(similarities) > 1:
        kde_similar = gaussian_kde(similarities)
        y_similar = kde_similar(x_range)
    
    if len(random_similarities) > 1:
        kde_random = gaussian_kde(random_similarities)
        y_random = kde_random(x_range)
    
    if len(similarities) > 1:
        plt.plot(x_range, y_similar, color='#3498DB', linewidth=2)
    if len(random_similarities) > 1:
        plt.plot(x_range, y_random, color='#E67E22', linewidth=2)
    
    plt.hist(similarities, bins=30, alpha=0.3, color='#3498DB', density=True, edgecolor='#3498DB', linewidth=1)
    plt.hist(random_similarities, bins=30, alpha=0.3, color='#E67E22', density=True, edgecolor='#E67E22', linewidth=1)
    
    plt.xlabel('Cosine Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Similarity Distribution - {layer_name}', fontsize=16)
    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=12, width=1, length=4)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()





def main():
    model_path = "./checkpoints/model_ckpt/final_model"
    base_model = "Qwen/Qwen2.5-7B"
    model, tokenizer, original_model = load_model(model_path, base_model)
    
    class SimpleTextFeatureExtractor:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = DEVICE
        
        def get_text_ffn_features(self, text, return_all_layers=False, return_activation=False, pooling_strategy='mean'):
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=60)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

                hidden_states = outputs.hidden_states
                layer_features = []

                if hasattr(self.model, 'model'):
                    transformer = self.model.model
                    layers = transformer.layers
                else:
                    layers = self.model.layers

                for i, layer in enumerate(layers):
                    if return_activation:
                        normed = layer.input_layernorm(hidden_states[i])
                        gate = layer.mlp.gate_proj(normed)
                        up = layer.mlp.up_proj(normed)
                        activation_output = torch.nn.functional.silu(gate) * up
                        layer_features.append(activation_output)
                    else:
                        normed = layer.input_layernorm(hidden_states[i])
                        ffn_output = layer.mlp(normed)
                        layer_features.append(ffn_output)
                
                pooled_features = []
                for feature in layer_features:
                    if pooling_strategy == 'mean':
                        valid_mask = attention_mask.unsqueeze(-1).to(dtype=feature.dtype)
                        sum_feature = torch.sum(feature * valid_mask, dim=1)
                        count = torch.sum(valid_mask, dim=1)
                        count = torch.maximum(count, torch.tensor(1.0, device=self.device, dtype=feature.dtype))
                        pooled = sum_feature / count
                    elif pooling_strategy == 'max':
                        valid_mask = attention_mask.unsqueeze(-1).to(dtype=feature.dtype)
                        masked_feature = feature * valid_mask + (1 - valid_mask) * torch.finfo(feature.dtype).min
                        pooled = torch.max(masked_feature, dim=1)[0]
                    elif pooling_strategy == 'cls':
                        pooled = feature[:, 0, :]
                    elif pooling_strategy == 'attention':
                        attention_weights = torch.softmax(feature.mean(dim=-1), dim=1)
                        attention_weights = attention_weights.unsqueeze(-1)
                        pooled = torch.sum(feature * attention_weights, dim=1)
                    else:
                        pooled = torch.mean(feature, dim=1)
                    
                    pooled_features.append(pooled)
                
                if return_all_layers:
                    return pooled_features
                else:
                    return pooled_features[-1]
    
    text_extractor = SimpleTextFeatureExtractor(original_model, tokenizer)

    test_dataset = load_from_disk('test_dataset_flickr_vit_clean')

    num_layers = model.lm.config.num_hidden_layers
    
    pooling_strategies = ['mean']

    for pooling_strategy in pooling_strategies:
        print(f"\n{'='*80}")

        layer_similarities_ffn = {i: [] for i in range(num_layers)}
        layer_random_similarities_ffn = {i: [] for i in range(num_layers)}
        layer_similarities_activation = {i: [] for i in range(num_layers)}
        layer_random_similarities_activation = {i: [] for i in range(num_layers)}
        

        processed_images = set()
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]
            if sample.get('type') != 'image':
                continue
            
            image_name = sample.get('image_name', f'img_{i}')
            if image_name in processed_images:
                continue
            processed_images.add(image_name)
            
            image_inputs = torch.tensor(sample['input_ids']).unsqueeze(0).to(DEVICE)
            
            if torch.all(image_inputs == 0):
                print(f"Warning: image_inputs for sample {i} is all zeros")
                continue
            
            with torch.no_grad():
                fimg_layers_ffn = model.get_image_ffn_features(image_inputs, return_all_layers=True, return_activation=False, pooling_strategy=pooling_strategy)
                fimg_layers_activation = model.get_image_ffn_features(image_inputs, return_all_layers=True, return_activation=True, pooling_strategy=pooling_strategy)
                
                caption_similarities_ffn = {i: [] for i in range(num_layers)}
                caption_similarities_activation = {i: [] for i in range(num_layers)}
                
                for j in range(i, min(i + 5, len(test_dataset))):
                    caption_sample = test_dataset[j]
                    if caption_sample.get('type') != 'image' or caption_sample.get('image_name', f'img_{j}') != image_name:
                        break
                    
                    caption_ids = torch.tensor(caption_sample['label'])
                    caption = tokenizer.decode(caption_ids, skip_special_tokens=True).strip()
                    
                    ftxt_layers_ffn = text_extractor.get_text_ffn_features(caption, return_all_layers=True, return_activation=False, pooling_strategy=pooling_strategy)
                    ftxt_layers_activation = text_extractor.get_text_ffn_features(caption, return_all_layers=True, return_activation=True, pooling_strategy=pooling_strategy)
                    
                    for layer_idx in range(num_layers):
                        similarity = compute_cosine_similarity(fimg_layers_ffn[layer_idx], ftxt_layers_ffn[layer_idx]).item()
                        caption_similarities_ffn[layer_idx].append(similarity)
                    
                    for layer_idx in range(num_layers):
                        similarity = compute_cosine_similarity(fimg_layers_activation[layer_idx], ftxt_layers_activation[layer_idx]).item()
                        caption_similarities_activation[layer_idx].append(similarity)
                
                for layer_idx in range(num_layers):
                    if caption_similarities_ffn[layer_idx]:
                        avg_sim = np.mean(caption_similarities_ffn[layer_idx])
                        layer_similarities_ffn[layer_idx].append(avg_sim)
                    
                    if caption_similarities_activation[layer_idx]:
                        avg_sim_activation = np.mean(caption_similarities_activation[layer_idx])
                        layer_similarities_activation[layer_idx].append(avg_sim_activation)
                
                fixed_caption = "Irrelevant text set by yourself"
                
                random_ftxt_layers_ffn = text_extractor.get_text_ffn_features(fixed_caption, return_all_layers=True, return_activation=False, pooling_strategy=pooling_strategy)
                random_ftxt_layers_activation = text_extractor.get_text_ffn_features(fixed_caption, return_all_layers=True, return_activation=True, pooling_strategy=pooling_strategy)
                
                for layer_idx in range(num_layers):
                    random_similarity = compute_cosine_similarity(fimg_layers_ffn[layer_idx], random_ftxt_layers_ffn[layer_idx]).item()
                    layer_random_similarities_ffn[layer_idx].append(random_similarity)
                
                for layer_idx in range(num_layers):
                    random_similarity = compute_cosine_similarity(fimg_layers_activation[layer_idx], random_ftxt_layers_activation[layer_idx]).item()
                    layer_random_similarities_activation[layer_idx].append(random_similarity)

        save_dir = f"./similarity_results_{pooling_strategy}"
        os.makedirs(save_dir, exist_ok=True)
        
        middle_layer = num_layers // 2
        layers_to_analyze = [middle_layer, num_layers - 1]
        last_six_layers = list(range(max(0, num_layers - 15), num_layers))

        print("-" * 80)
        
        for layer_idx in last_six_layers:
            similarities = layer_similarities_ffn[layer_idx]
            random_similarities = layer_random_similarities_ffn[layer_idx]
            
            avg_similarity = np.mean(similarities)
            avg_random_similarity = np.mean(random_similarities)
            std_similarity = np.std(similarities)
            std_random_similarity = np.std(random_similarities)
            
            print(f"\nLayer {layer_idx + 1}:")
            print(f"{avg_similarity:.4f} ± {std_similarity:.4f}, comparison1: {avg_random_similarity:.4f} ± {std_random_similarity:.4f}")
            print(f"difference: {avg_similarity - avg_random_similarity:.4f}")

            
            layer_name = f"Layer {layer_idx + 1} (FFN Output, {pooling_strategy})"
            save_path = os.path.join(save_dir, f"similarity_distribution_layer_{layer_idx + 1}_ffn.png")
            plot_similarity_distribution(similarities, random_similarities, None, layer_name, save_path)

        for layer_idx in last_six_layers:
            similarities = layer_similarities_activation[layer_idx]
            random_similarities = layer_random_similarities_activation[layer_idx]
            
            avg_similarity = np.mean(similarities)
            avg_random_similarity = np.mean(random_similarities)
            std_similarity = np.std(similarities)
            std_random_similarity = np.std(random_similarities)
            
            print(f"\nLayer {layer_idx + 1}:")
            print(f"{avg_similarity:.4f} ± {std_similarity:.4f}, comparison1: {avg_random_similarity:.4f} ± {std_random_similarity:.4f}")
            print(f"difference: {avg_similarity - avg_random_similarity:.4f}")
            
            layer_name = f"Layer {layer_idx + 1} (Activation, {pooling_strategy})"
            save_path = os.path.join(save_dir, f"similarity_distribution_layer_{layer_idx + 1}_activation.png")
            plot_similarity_distribution(similarities, random_similarities, None, layer_name, save_path)

        for layer_idx in layers_to_analyze:
            similarities = layer_similarities_ffn[layer_idx]
            random_similarities = layer_random_similarities_ffn[layer_idx]
            
            if not similarities or not random_similarities:
                continue
            
            avg_similarity = np.mean(similarities)
            avg_random_similarity = np.mean(random_similarities)
            std_similarity = np.std(similarities)
            std_random_similarity = np.std(random_similarities)
            
            print(f"\nLayer {layer_idx + 1}:")
            print(f"{avg_similarity:.4f} ± {std_similarity:.4f}, comparison1: {avg_random_similarity:.4f} ± {std_random_similarity:.4f}")
            print(f"difference: {avg_similarity - avg_random_similarity:.4f}")

            layer_name = f"Layer {layer_idx + 1} (Final FFN, {pooling_strategy})"
            save_path = os.path.join(save_dir, f"similarity_distribution_layer_{layer_idx + 1}_final_ffn.png")
            plot_similarity_distribution(similarities, random_similarities, None, layer_name, save_path)

        for layer_idx in layers_to_analyze:
            similarities = layer_similarities_activation[layer_idx]
            random_similarities = layer_random_similarities_activation[layer_idx]
            
            if not similarities or not random_similarities:
                continue
            
            avg_similarity = np.mean(similarities)
            avg_random_similarity = np.mean(random_similarities)
            std_similarity = np.std(similarities)
            std_random_similarity = np.std(random_similarities)
            
            print(f"\nLayer {layer_idx + 1}:")
            print(f" {avg_similarity:.4f} ± {std_similarity:.4f}, comparison1: {avg_random_similarity:.4f} ± {std_random_similarity:.4f}")
            print(f"difference: {avg_similarity - avg_random_similarity:.4f}")

            layer_name = f"Layer {layer_idx + 1} (Activation, {pooling_strategy})"
            save_path = os.path.join(save_dir, f"similarity_distribution_layer_{layer_idx + 1}_final_activation.png")
            plot_similarity_distribution(similarities, random_similarities, None, layer_name, save_path)

        plt.figure(figsize=(16, 10))
        
        plt.rcParams.update({'font.size': 12})
        
        layer_indices = list(range(num_layers))
        
        avg_similarities_ffn = [np.mean(layer_similarities_ffn[i]) for i in layer_indices]
        avg_random_similarities_ffn = [np.mean(layer_random_similarities_ffn[i]) for i in layer_indices]
        
        avg_similarities_activation = [np.mean(layer_similarities_activation[i]) for i in layer_indices]
        avg_random_similarities_activation = [np.mean(layer_random_similarities_activation[i]) for i in layer_indices]
        
        plt.plot(layer_indices, avg_similarities_ffn, color='#3498DB', linewidth=2, marker='o', markersize=6)
        plt.plot(layer_indices, avg_random_similarities_ffn, color='#90CAF9', linewidth=2, marker='o', markersize=6, linestyle='--')
        
        plt.plot(layer_indices, avg_similarities_activation, color='#E67E22', linewidth=2, marker='x', markersize=6)
        plt.plot(layer_indices, avg_random_similarities_activation, color='#F39C12', linewidth=2, marker='x', markersize=6, linestyle='--')
        
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Average Cosine Similarity', fontsize=14)
        plt.title(f'Average Similarity Across Layers (Pooling: {pooling_strategy})', fontsize=16)
        plt.xticks(layer_indices, [f'{i+1}' for i in layer_indices], fontsize=10)
        plt.grid(False)
        plt.tick_params(axis='both', which='major', labelsize=12, width=1, length=4)
        
        save_path = os.path.join(save_dir, "average_similarity_across_layers_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("\nExperiment completed！")


if __name__ == "__main__":
    main()
