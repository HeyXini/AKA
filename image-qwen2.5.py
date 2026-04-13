# -*- coding: utf-8 -*-
# Qwen2.5 Multimodal Model Fine-tuning Script
import os
import sys
import builtins
import socket

# Save original print function
original_print = builtins.print

# Override print function to filter out socket connection logs
def filtered_print(*args, **kwargs):
    # Check if any argument contains socket connection information
    for arg in args:
        if isinstance(arg, str) and "socket.socket fd=" in arg:
            return  # Filter out socket connection logs
        if hasattr(arg, '__str__'):
            arg_str = str(arg)
            if "socket.socket fd=" in arg_str:
                return  # Filter out socket connection logs
    # Otherwise call original print function
    original_print(*args, **kwargs)

# Apply filtered print function
builtins.print = filtered_print

# Try to redirect socket module output
socket_original_str = socket.socket.__str__
def socket_filtered_str(self):
    return "<socket.socket>"  # Simplify socket object string representation
socket.socket.__str__ = socket_filtered_str
socket.socket.__repr__ = socket_filtered_str

from mpmath.math2 import sqrt2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_from_disk

import numpy as np
import math
from tqdm import tqdm
from typing import Optional, List, Dict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from timm.layers import build_sincos2d_pos_embed
import regex as re
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ViTMAEForPreTraining
import random

# Fix random seed to ensure experiment reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables to ensure reproducibility
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# New: Set CUDA memory allocation configuration to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Global IDF calculator instance
idf_calculator = None



# Load locally saved processor and model
processor = AutoImageProcessor.from_pretrained("./vit-mae-model")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# IDF loss calculator class
class IDFLossCalculator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idf_dict = self._build_idf_dict()
    
    def _build_idf_dict(self):
        """Build IDF dictionary, using a simple implementation here, should be based on larger corpus in practice"""
        # Simple IDF values based on common word frequency
        common_words = {
            'a': 0.1, 'an': 0.1, 'the': 0.1,
            'in': 0.2, 'on': 0.2, 'at': 0.2,
            'is': 0.15, 'are': 0.15, 'was': 0.15,
            'were': 0.15, 'be': 0.15
        }
        return common_words
    
    def compute_idf_weight(self, tokens):
        """Calculate IDF weight"""
        token_texts = self.tokenizer.convert_ids_to_tokens(tokens.cpu().numpy())
        weights = []
        for token in token_texts:
            clean_token = token.replace('Ġ', '')  # Remove prefix
            # Look up weight in IDF dictionary, use default value 1.0 if not found
            weight = self.idf_dict.get(clean_token.lower(), 1.0)
            weights.append(weight)
        return torch.tensor(weights, device=tokens.device, dtype=torch.float32)




class MultimodalModel(nn.Module):
    """Multimodal model supporting image and text inputs"""

    def __init__(self, model, tokenizer=None,
                 history_dropout_p: float = 0.3,  # Recommended starting value: 0.3
                 history_dropout_protect: int = 2,  # Don't drop first 2 caption tokens for stability
                 history_dropout_noise_std: float = 0,  # Optional: 0=zero out; >0=add Gaussian noise
                 ):
        super().__init__()
        # Base model
        self.lm = model
        self.tokenizer = tokenizer
        self.device = DEVICE

        self.history_dropout_p = history_dropout_p
        self.history_dropout_protect = history_dropout_protect
        self.history_dropout_noise_std = history_dropout_noise_std

        hidden_dim = model.config.hidden_size
        self.mode_embeddings = nn.Embedding(2, hidden_dim).to(self.device)
        self.image_proj = nn.Linear(768, hidden_dim).to(self.device, dtype=model.dtype)
        self.caption_start = nn.Parameter(torch.randn(1, 1, hidden_dim, device=self.device, dtype=model.dtype))

       

    def print_trainable_parameters(self):
        # 1. Count LoRA layer parameters (call PeftModel method)
        # print("\n===== LoRA Layer Trainable Parameters =====")
        # self.lm.print_trainable_parameters()

        # 2. Manually count custom layer parameters (image_proj + mode_embeddings + caption_start)
        print("\n===== Custom Layer Trainable Parameters =====")
        custom_trainable_params = 0
        custom_total_params = 0
        # Iterate through custom layer parameters
        for name, param in [("image_proj", self.image_proj), ("mode_embeddings", self.mode_embeddings)]:
            # Count total parameters for this layer
            total = sum(p.numel() for p in param.parameters())
            # Count trainable parameters for this layer
            trainable = sum(p.numel() for p in param.parameters() if p.requires_grad)

            custom_total_params += total
            custom_trainable_params += trainable
            print(f"{name}: Trainable params={trainable:,} | Total params={total:,} | Trainable ratio={trainable / total:.2%}")
        # Count caption_start parameters
        caption_start_total = self.caption_start.numel()
        caption_start_trainable = caption_start_total if self.caption_start.requires_grad else 0
        custom_total_params += caption_start_total
        custom_trainable_params += caption_start_trainable
        print(f"caption_start: Trainable params={caption_start_trainable:,} | Total params={caption_start_total:,} | Trainable ratio={caption_start_trainable / caption_start_total:.2%}")
        # Print total custom layer parameters
        print(f"Total custom layers: Trainable params={custom_trainable_params:,} | Total params={custom_total_params:,} | Trainable ratio={custom_trainable_params / custom_total_params:.2%}")


    def apply_history_dropout(self, caption_embeds: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        """
        caption_embeds: [B, T, D]
        caption_ids:    [B, T]
        Apply dropout to valid token embeddings only during training, without modifying token ids (safe for Qwen2.5).
        """
        if (not self.training) or self.history_dropout_p <= 0:
            return caption_embeds
        if self.tokenizer is None:
            return caption_embeds

        pad_id = self.tokenizer.pad_token_id
        # Valid tokens (non-pad)
        valid = (caption_ids != pad_id)  # [B,T] bool

        # Generate keep mask: True=keep, False=drop
        keep = (torch.rand(caption_ids.shape, device=caption_ids.device) > self.history_dropout_p) & valid

        # Protect first N tokens from being dropped (for stability)
        if self.history_dropout_protect and self.history_dropout_protect > 0:
            keep[:, :self.history_dropout_protect] = True

        keep_f = keep.unsqueeze(-1).to(dtype=caption_embeds.dtype)  # [B,T,1]

        if self.history_dropout_noise_std and self.history_dropout_noise_std > 0:
            # Add noise to dropped positions (instead of zeroing out)
            noise = torch.randn_like(caption_embeds) * self.history_dropout_noise_std
            dropped = caption_embeds + noise
            caption_embeds = caption_embeds * keep_f + dropped * (1 - keep_f)
        else:
            # Zero out dropped positions (most common)
            caption_embeds = caption_embeds * keep_f

        return caption_embeds


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                att_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None,
                caption: Optional[torch.Tensor] = None,
                ):
        image_embeds = self.image_proj(input_ids) 
        B, H, D = image_embeds.shape

        caption_embeds = self.lm.model.model.embed_tokens(caption)  # B,60,D
        # caption_embeds = self.apply_history_dropout(caption_embeds, caption)

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


def visualize_caption_token_attention(attention_weights, image, caption_tokens=None, save_path=None,
                                      heads_to_show=None, aggregate_heads=True, aggregation_method='mean',
                                      show_all_tokens=True, token_filter=None, aggregate_tokens=False):
    """
    Visualize caption token attention heatmaps on images
    
    Args:
        attention_weights: Attention weights with shape (batch_size, num_heads, caption_seq_len, L_img)
        image: Original image, supporting PIL Image, numpy array, or torch tensor
        caption_tokens: List of caption tokens for heatmap titles
        save_path: Save path, if None, no saving
        heads_to_show: List of attention head indices to display
        aggregate_heads: Whether to aggregate all attention heads
        aggregation_method: Aggregation method, options: 'mean', 'sum', or 'max'
        show_all_tokens: Whether to show heatmaps for all tokens, False to show only tokens filtered by token_filter
        token_filter: Function to filter tokens, only tokens returning True will be displayed
        aggregate_tokens: Whether to aggregate attention from all tokens, showing overall attention distribution
    
    Returns:
        matplotlib figure object
    """
    attn = attention_weights[0]  # (num_heads, caption_seq_len, L_img)
    
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        image = np.array(image)

    img_h, img_w = image.shape[:2]
    
    patch_size = int(np.sqrt(attn.shape[-1]))  
    
    if aggregate_heads:
        attn_aggregated = getattr(np, aggregation_method)(attn, axis=0)  # (caption_seq_len, L_img)
    else:
        if heads_to_show is None:
            heads_to_show = list(range(min(8, attn.shape[0])))
        else:
            heads_to_show = [h for h in heads_to_show if 0 <= h < attn.shape[0]]
    
    token_indices = range(attn.shape[1])
    if not show_all_tokens and token_filter is not None:
        filtered_indices = []
        for i in range(attn.shape[1]):
            token = caption_tokens[i].replace('Ġ', '') if caption_tokens and i < len(caption_tokens) else f"Token {i}"
            if token_filter(token):
                filtered_indices.append(i)
        if filtered_indices:
            token_indices = filtered_indices
        else:
            print("Warning: All tokens have been filtered out, showing all tokens instead")
    

    
    if aggregate_tokens:
        if aggregate_heads:
            all_tokens_attn = getattr(np, aggregation_method)(attn_aggregated[list(token_indices)], axis=0)  # (L_img)
        else:
            head_aggregated = getattr(np, aggregation_method)(attn, axis=0)  # (caption_seq_len, L_img)
            all_tokens_attn = getattr(np, aggregation_method)(head_aggregated[list(token_indices)], axis=0)  # (L_img)

        
        attn_reshaped = all_tokens_attn.reshape(patch_size, patch_size)
  
        attn_image = Image.fromarray(attn_reshaped.astype(np.float32))
        attn_resized = np.array(attn_image.resize((img_w, img_h), Image.BICUBIC))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        im = ax.imshow(attn_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        ax.set_title("Attention distribution of all tokens")
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('weight', fontsize=12)
    
    elif aggregate_heads:
        num_tokens = len(token_indices)
        if num_tokens == 0:
            num_tokens = 1
        
        fig, axs = plt.subplots(1, num_tokens, figsize=(5 * num_tokens, 5))
        if num_tokens == 1:
            axs = [axs]
        
        for ax_idx, i in enumerate(token_indices):
            ax = axs[ax_idx]
            token_attn = attn_aggregated[i]  # (L_img)
            token_attn_reshaped = token_attn.reshape(patch_size, patch_size)  # (patch_size, patch_size)
            attn_image = Image.fromarray(token_attn_reshaped.astype(np.float32))
            attn_resized = np.array(attn_image.resize((img_w, img_h), Image.BICUBIC))
            
            ax.imshow(image)
            im = ax.imshow(attn_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
            if caption_tokens and i < len(caption_tokens):
                token = caption_tokens[i].replace('Ġ', '')
                ax.set_title(f"Token {i}: {token}", fontsize=10)
            else:
                ax.set_title(f"Token {i}", fontsize=10)
            ax.axis('off')
        
        cbar = plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        cbar.set_label('weight', fontsize=12)
    
    else:
        num_heads = len(heads_to_show)
        num_tokens = len(token_indices)
        
        fig, axs = plt.subplots(num_heads, num_tokens, figsize=(5 * num_tokens, 5 * num_heads))
        if num_heads == 1:
            axs = [axs]
        if num_tokens == 1:
            axs = [[ax] for ax in axs]
        
        for h, head_idx in enumerate(heads_to_show):
            for ax_idx, t in enumerate(token_indices):
                token_attn = attn[head_idx, t]  # (L_img)
                token_attn_reshaped = token_attn.reshape(patch_size, patch_size)  
                attn_image = Image.fromarray(token_attn_reshaped.astype(np.float32))
                attn_resized = np.array(attn_image.resize((img_w, img_h), Image.BICUBIC))
                
                axs[h][ax_idx].imshow(image)
                im = axs[h][ax_idx].imshow(attn_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
                if caption_tokens and t < len(caption_tokens):
                    token = caption_tokens[t].replace('Ġ', '')
                    axs[h][ax_idx].set_title(f"Head {head_idx}, Token {t}: {token}", fontsize=8)
                else:
                    axs[h][ax_idx].set_title(f"Head {head_idx}, Token {t}", fontsize=8)
                axs[h][ax_idx].axis('off')
        
        cbar = plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
        cbar.set_label('weight', fontsize=12)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    return fig


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        multimodal_model = model.module if hasattr(model, 'module') else model

        custom_state_dict = {
            "image_proj": {k: v.detach().cpu() for k, v in multimodal_model.image_proj.state_dict().items()},
            "mode_embeddings": {k: v.detach().cpu() for k, v in multimodal_model.mode_embeddings.state_dict().items()},
            "caption_start": multimodal_model.caption_start.detach().cpu(),
        }

        checkpoint_step = self.state.global_step
   
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{checkpoint_step}")
   
        save_path = os.path.join(checkpoint_dir, "custom_layers.bin")
        torch.save(custom_state_dict, save_path)

        
        if isinstance(multimodal_model.lm, PeftModel):
            multimodal_model.lm.save_pretrained(checkpoint_dir)

    def prediction_step(
            self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss

        if prediction_loss_only:
            return (loss, None, inputs.get("labels"))
        return (loss, inputs, inputs.get("labels"))

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        if self.args.max_grad_norm > 0:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, self.args.max_grad_norm)
        
        return loss


class MultimodalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        labels = []
        mask2 = []
        caption = []
        for f in features:
            label_tensor = torch.tensor(f["label"])
            caption.append(label_tensor.clone())
            label_tensor[label_tensor == self.pad_id] = -100
            label_mask_tensor = label_tensor.clone()
            valid_len = (label_mask_tensor != -100).sum().item()
            label_mask_tensor[valid_len] = self.eos_id
            mask2.append((label_tensor!=-100).long())
            labels.append(label_mask_tensor)
        labels = torch.stack(labels)
        mask2 = torch.stack(mask2)
        caption = torch.stack(caption)

        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
            "labels": labels,
            "att_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
            "mask2": mask2,
            "caption": caption
        }



def compute_metrics(eval_pred):
 
    tokenizer = globals()['tokenizer']
    model = globals()['model']
    
    original_use_cache = model.lm.config.use_cache
    
    model.eval()
    model.lm.config.use_cache = True
    if hasattr(model.lm, 'gradient_checkpointing_disable'):
        model.lm.gradient_checkpointing_disable()
    with torch.no_grad(): 
        pass

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    inputs, labels = eval_pred
    input_ids = inputs['input_ids']

    decoded_preds = []
    decoded_labels = []
    with torch.no_grad():
        batch_size = input_ids.shape[0]
        generated_sequences = []
        batch_start = 0
        batch_size_gen = 8 
        while batch_start < batch_size:
            batch_end = min(batch_start + batch_size_gen, batch_size)
            batch_input_ids = torch.tensor(input_ids[batch_start:batch_end]).to(model.device)
            img_embeds = model.image_proj(batch_input_ids)
            B, L_img, D = img_embeds.shape
            caption_start = model.caption_start.to(dtype=model.lm.dtype, device=model.device).expand(B, -1, -1)
            inputs_embeds = torch.cat([img_embeds, caption_start], dim=1)
         
            attention_mask_gen = torch.ones(
                (B, inputs_embeds.shape[1]),
                dtype=torch.long,
                device=model.device
            )
           
            outputs = model.lm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_gen,
                max_new_tokens=30,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                early_stopping=True,
                use_cache=True,
                do_sample=False,
                num_beams=5,
                no_repeat_ngram_size=2,
            )
            generated_sequences.extend(outputs)
            batch_start = batch_end

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    for seq, lbl in zip(generated_sequences, labels):

        pred_text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        decoded_preds.append(pred_text if pred_text else "no caption generated")

        lbl = lbl[(lbl != -100) & (lbl != pad_id)] 
        lbl_text = tokenizer.decode(lbl, skip_special_tokens=True).strip()
        decoded_labels.append(lbl_text)

    gts = {idx: [txt] for idx, txt in enumerate(decoded_labels)}
    res = {idx: [txt] for idx, txt in enumerate(decoded_preds)}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Meteor(), "METEOR"),
    ]
    metrics = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                metrics[m] = round(sc, 4)
        else:
            metrics[method] = round(score, 4)
    metrics['combined_score'] = metrics['CIDEr'] * 0.7 + metrics['Bleu_4'] * 0.3

    model.lm.config.use_cache = original_use_cache
    if hasattr(model.lm, 'gradient_checkpointing_enable') and not original_use_cache:
        model.lm.gradient_checkpointing_enable()
    
    return metrics


def compute_custom_loss(model: MultimodalModel, inputs: Dict, return_outputs: bool = False,
                        num_items_in_batch: Optional[int] = None):

    

    input_ids = inputs.get("input_ids")
    att_mask = inputs.get("att_mask")
    labels = inputs.get("labels")
    mask2 = inputs.get("mask2")
    caption = inputs.get("caption")

    outputs = model(
        input_ids=input_ids,
        att_mask=att_mask,
        labels=labels,
        mask2=mask2,
        caption=caption
    )
    weighted_loss = outputs.loss

    return (weighted_loss, outputs) if return_outputs else weighted_loss




def evaluate_image_captioning(model, test_dataset, tokenizer, device, batch_size=5, 
                              visualize_attention=False, attention_output_dir="attention_maps",
                              num_visualize=100):
  
    model.eval()

    if visualize_attention:
        os.makedirs(attention_output_dir, exist_ok=True)
        print(f"注意力热力图将保存到: {attention_output_dir}")

    original_model = None

    image_references = {}  
    image_features = {}  

    for i in tqdm(range(len(test_dataset)), desc="Collecting reference descriptions"):
        sample = test_dataset[i]
        if sample.get('type') != 'image':
            continue

        input_ids = np.array(sample['input_ids'])
        attention_mask = np.array(sample['attention_mask'])

        img_id = sample['image_name']
        ref_caption = tokenizer.decode(np.array(sample['label']), skip_special_tokens=True).strip()

        if img_id not in image_references:
            image_references[img_id] = []
        image_references[img_id].append(ref_caption)

        if img_id not in image_features:
            image_features[img_id] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'index': i
            }

    print(f"Reference descriptions collected, found {len(image_features)} unique images")

    # --------------Batch caption generation --------------
    gts = {}  
    res = {}  
    img_ids = list(image_features.keys())

    visualize_count = 0  

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(img_ids), batch_size), desc="Batch captions generation"):
            batch_ids = img_ids[batch_start:batch_start + batch_size]
            batch_input_ids = []
            batch_attention_mask = []
            
            for img_id in batch_ids:
                feat = image_features[img_id]
                batch_input_ids.append(torch.from_numpy(feat['input_ids']).to(device))
                batch_attention_mask.append(torch.from_numpy(feat['attention_mask']).to(device))
            
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_mask = torch.stack(batch_attention_mask)
            
            need_attention = visualize_attention and visualize_count < num_visualize
            
            if need_attention:
                batch_captions, batch_attentions = generate_response(
                    model,
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_attentions=True
                )
                
                if original_model is not None:
                    original_batch_captions, original_batch_attentions = generate_response(
                        original_model,
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        return_attentions=True
                    )
            else:
                batch_captions = generate_response(
                    model,
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_attentions=False
                )
            
            for idx, img_id in enumerate(batch_ids):
                pred_caption = batch_captions[idx].strip()
                
                gts[img_id] = image_references[img_id]
                res[img_id] = [pred_caption] 
                
                if need_attention and visualize_count < num_visualize:
                    image_path = os.path.join("Flickr8k_dataset/Flickr8k_images", img_id)
                    
                    image = Image.open(image_path).convert("RGB")
                    
                    if batch_attentions is not None:
                        sample_attentions = batch_attentions[idx:idx+1]  # (1, num_heads, seq_len)

                        save_path = os.path.join(
                            attention_output_dir,
                            f"{img_id}_attention_count{visualize_count}.png"
                        )

                        caption_tokens = model.tokenizer.tokenize(pred_caption) if model.tokenizer else None

                        def is_content_token(token):
                            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                                         'is', 'are', 'was', 'were', 'be', 'been', 'being', 'and', 'or', 'but',
                                         'if', 'because', 'as', 'while', 'where', 'when', 'which', 'who', 'whom',
                                         'this', 'that', 'these', 'those', 'am', 'is', 'are', 'has', 'have', 'had'}
                            return token.lower() not in stop_words and len(token) > 1

                        visualize_caption_token_attention(
                            attention_weights=sample_attentions,
                            image=image,
                            caption_tokens=caption_tokens,
                            save_path=save_path.replace('.png', '_content_tokens.png'),
                            aggregate_heads=True,
                            aggregation_method='mean',
                            show_all_tokens=False,
                            token_filter=is_content_token
                        )

                        visualize_caption_token_attention(
                            attention_weights=sample_attentions,
                            image=image,
                            caption_tokens=caption_tokens,
                            save_path=save_path.replace('.png', '_aggregated.png'),
                            aggregate_heads=True,
                            aggregation_method='mean',
                            aggregate_tokens=True
                        )

                    
                        caption_file = os.path.join(
                            attention_output_dir,
                            f"{img_id}_caption_count{visualize_count}.txt"
                        )
                        with open(caption_file, 'w', encoding='utf-8') as f:
                            f.write(f"Image ID: {img_id}\n")
                            f.write(f"Generated Caption: {pred_caption}\n")
                            f.write(f"Reference Captions:\n")
                            for ref in image_references[img_id]:
                                f.write(f"  - {ref}\n")

                        visualize_count += 1


    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),  
        (Spice(), "SPICE")  
    ]
    metrics = {}
    print("\nStart calculating indicators...")
    for scorer, method in scorers:
        # for i, (scorer, method) in enumerate(scorers):
        print(f"Calculate indicators: {method}")
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                metrics[m] = round(sc, 4)
                print(metrics[m])
        else:
            metrics[method] = round(score, 4)
            print(metrics[method])

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics





def generate_response(model: MultimodalModel,
                      input_ids: Optional[torch.Tensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      return_attentions: bool = False
                      ) -> List[str] or tuple:
    with torch.no_grad():
        model.eval()
        model.lm.eval()
        model_device = model.module.device if hasattr(model, 'module') else model.device
        input_dtype = model.image_proj.weight.dtype
        input_ids = input_ids.to(dtype=input_dtype, device=model_device)

        inputs_embeds = model.image_proj(input_ids)
        img_embeds = inputs_embeds
        B, L_img, D = inputs_embeds.shape
        caption_start = model.caption_start.to(dtype=input_dtype, device=model_device).expand(B, -1, -1)
        inputs_embeds = torch.cat([img_embeds, caption_start], dim=1)

        attention_mask = torch.ones(
            (B, inputs_embeds.shape[1]),
            dtype=torch.long,
            device=model_device
        )

        outputs = model.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=30,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            early_stopping=True,
            use_cache=True,
            output_attentions=return_attentions,
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        
        generated_texts = []
        sequences = outputs.sequences

        if return_attentions:
            generated_caption_ids = sequences
            generated_caption_embeds = model.lm.model.model.embed_tokens(generated_caption_ids).to(dtype=input_dtype)
            full_inputs_embeds = torch.cat([inputs_embeds, generated_caption_embeds], dim=1)

            full_attention_mask = torch.ones(
                (B, full_inputs_embeds.shape[1]),
                dtype=torch.long,
                device=model_device
            )
            forward_outputs = model.lm(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                output_attentions=True
            )
            attentions = None
            if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions is not None:
                last_layer_attentions = forward_outputs.attentions[-1]  # (batch_size, num_heads, seq_len_q, seq_len_kv)
                caption_start_idx = L_img + 1  
                all_caption_attentions = last_layer_attentions[:, :, caption_start_idx:, :]  # (batch_size, num_heads, caption_seq_len, seq_len_kv)
                image_attentions = all_caption_attentions[:, :, :, 1:L_img]  # (batch_size, num_heads, caption_seq_len, L_img)
                attentions = image_attentions.cpu().detach().numpy()


        for output in sequences:
            text = model.tokenizer.decode(output, skip_special_tokens=True).strip()
            generated_texts.append(text if text else "no caption generated")

        return (generated_texts, attentions) if return_attentions else generated_texts


#
def evaluate_saved_model(model_path=None, base_model="Qwen/Qwen2.5-7B", checkpoint_dir=None, use_quantized=True):
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if use_quantized else None

    use_checkpoint = checkpoint_dir is not None and os.path.exists(checkpoint_dir)
    load_path = checkpoint_dir if use_checkpoint else model_path
    

    print("Loading tokenizer...")
    if use_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # model.set_attn_implementation('eager')

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,  # base model
        load_path,  
    )

    # model = model.merge_and_unload()
    
    print("Creating multimodal model...")
    multimodal_model = MultimodalModel(model, tokenizer)

    print("Loading custom layers...")
    custom_layers_path = os.path.join(load_path, "custom_layers.bin")
    custom_weights = torch.load(custom_layers_path, map_location=DEVICE)
    multimodal_model.image_proj.load_state_dict(custom_weights["image_proj"])
    multimodal_model.caption_start.data.copy_(custom_weights["caption_start"])

    multimodal_model.model_dtype = next(multimodal_model.parameters()).dtype

    test_dataset = load_from_disk('test_dataset_flickr_vit_clean')

    if test_dataset:
        evaluate_image_captioning(multimodal_model, test_dataset, tokenizer, DEVICE)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multimodal model training and evaluation parameters')
    parser.add_argument('--evaluate_only', type=str, choices=['True', 'False'], default='False',
                        help='Only evaluate the model, no training')
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Path to the model to evaluate')
    parser.add_argument('--resume_from_checkpoint', type=str, choices=['True', 'False'], default='False',
                        help='Resume training from checkpoint, if True')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=None,
                        help='Path to the checkpoint directory to load or resume')
    parser.add_argument('--use_quantized', type=str, choices=['True', 'False'], default='False',
                        help='Use 4-bit quantized model, if True, False otherwise')
    args = parser.parse_args()

    model_name = "Qwen/Qwen2.5-7B"

    if args.evaluate_only.lower() == 'true':
        use_quantized = args.use_quantized.lower() == 'true'
        evaluate_saved_model(model_path=args.model_path, base_model=model_name, checkpoint_dir=args.checkpoint_dir, use_quantized=use_quantized)
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={'': DEVICE},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Configure LoRA
    lora_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = prepare_model_for_kbit_training(model)
    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  
    print("Initializing model...")

    multimodal_model = MultimodalModel(model, tokenizer)
    

    globals()['multimodal_model'] = multimodal_model
    globals()['tokenizer'] = tokenizer

    if hasattr(multimodal_model.lm, 'gradient_checkpointing_enable'):
        multimodal_model.lm.gradient_checkpointing_enable()
    multimodal_model.lm.config.use_cache = False

    model = multimodal_model

    print("\n===== All trainable parameters validation =====")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters parameters: {trainable_params:,}")
    print(f"Trainable parameters ratio: {trainable_ratio:.4%}")
    
    model.print_trainable_parameters()  
    print("Initializing model...")

    print('Loading datasets...')
    train_dataset = load_from_disk('train_dataset_flickr_vit_clean')
    val_dataset = load_from_disk('val_dataset_flickr_vit_clean')
    test_dataset = load_from_disk('test_dataset_flickr_vit_clean')
    print('Datasets loaded.')

    collator = MultimodalCollator(tokenizer=tokenizer)
    today = 260101
    lr = 1.5e-4 # 1e-4 1.5e-4 2e-4
    batch_size = 16
    epochs = 15
    mode = '1'
    modelname = model_name.split('/')[-1]


    resume_from_checkpoint = args.resume_from_checkpoint.lower() == 'true'
    checkpoint_dir = args.checkpoint_dir
    if resume_from_checkpoint and checkpoint_dir and os.path.isdir(checkpoint_dir):
        print(f"Resume training from checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = None
        print("No checkpoint enabled")

    checkpoints_dir = f"./checkpoints/{modelname}_{mode}_{today}_lr{lr}_bs{batch_size}_epochs{epochs}_rank{lora_config.r}"
    os.makedirs(checkpoints_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        load_best_model_at_end=True,
        optim="adamw_torch",
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.015,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        logging_dir="./logs/{}_{}_{}_lr{}_bs{}_epochs{}_rank{}".format(modelname, mode, today, lr, batch_size, epochs, lora_config.r),
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        save_total_limit=1,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        save_safetensors=False,
        save_on_each_node=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=8,
        dataloader_pin_memory=True
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    trainer.compute_loss = lambda model, inputs, return_outputs=False, num_items_in_batch=None: compute_custom_loss(
        model, inputs, return_outputs, num_items_in_batch)


    if resume_from_checkpoint and checkpoint_dir:
        import re

        step_match = re.search(r'(\d+)', os.path.basename(checkpoint_dir))
        step = int(step_match.group(1))
        print(f"Checkpoint step: {step}")

        custom_layer_file = os.path.join(checkpoint_dir, "custom_layers.bin")

        custom_state_dict = torch.load(custom_layer_file, map_location=trainer.model.device)

        model_to_load = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_load.image_proj.load_state_dict(custom_state_dict["image_proj"], strict=True)
        model_to_load.mode_embeddings.load_state_dict(custom_state_dict["mode_embeddings"], strict=True)
        model_to_load.caption_start.data.copy_(
            custom_state_dict["caption_start"].to(device=model_to_load.caption_start.device,
                                            dtype=model_to_load.caption_start.dtype))

        print(f"Custom layers image_proj / mode_embeddings / caption_start restored from checkpoint {step}.")
        print(f"Resume training from checkpoint: {checkpoint_dir}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("Start new training session...")
        train_result = trainer.train()
        print(
            f"Training will save checkpoints at every {training_args.save_steps} steps, up to {training_args.save_total_limit} checkpoints.")
        print(f"Checkpoints will be saved in: {training_args.output_dir}")

    print("Training completed...")

    save_dir = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model.lm, PeftModel):
        model.lm.save_pretrained(save_dir)
    else:
        print("LoRA weights not saved")
    custom_weights = {
        "image_proj": model.image_proj.state_dict(),
        "mode_embeddings": model.mode_embeddings.state_dict(),
        "caption_start": model.caption_start.data
    }
    torch.save(custom_weights, os.path.join(save_dir, "custom_layers.bin"))
    tokenizer.save_pretrained(save_dir)
    print(f"Final model saved to: {save_dir}")

    print("\nStart evaluating model on test dataset...")
    image_metrics = evaluate_image_captioning(model, test_dataset, tokenizer, DEVICE)
    print("Image caption evaluation completed")
