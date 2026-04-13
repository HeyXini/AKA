# -*- coding: utf-8 -*-
import os
import sys
import builtins
import socket
import re
original_print = builtins.print

def filtered_print(*args, **kwargs):
    for arg in args:
        if isinstance(arg, str) and "socket.socket fd=" in arg:
            return
        if hasattr(arg, '__str__'):
            arg_str = str(arg)
            if "socket.socket fd=" in arg_str:
                return
    original_print(*args, **kwargs)

builtins.print = filtered_print

socket_original_str = socket.socket.__str__
def socket_filtered_str(self):
    return "<socket.socket>"
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
import random
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")







class MultimodalModel(nn.Module):

    def __init__(self, model, tokenizer=None,
                 history_dropout_p: float = 0.3,
                 history_dropout_protect: int = 2,
                 history_dropout_noise_std: float = 0,
                 ):
        super().__init__()
        self.lm = model
        self.tokenizer = tokenizer
        self.device = DEVICE

        self.history_dropout_p = history_dropout_p
        self.history_dropout_protect = history_dropout_protect
        self.history_dropout_noise_std = history_dropout_noise_std

        hidden_dim = model.config.hidden_size
        self.image_proj = nn.Linear(768, hidden_dim).to(self.device, dtype=model.dtype)
        self.question_start = nn.Parameter(torch.randn(1, 1, hidden_dim, device=self.device, dtype=model.dtype))


    def print_trainable_parameters(self):
        # 1. Count LoRA layer parameters (call PeftModel method)
        # print("\n===== LoRA Layer Trainable Parameters =====")
        # self.lm.print_trainable_parameters()

        # 2. Manually count custom layer parameters (image_proj + mode_embeddings + question_start)
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
        # Count question_start parameters
        question_start_total = self.question_start.numel()
        question_start_trainable = question_start_total if self.question_start.requires_grad else 0
        custom_total_params += question_start_total
        custom_trainable_params += question_start_trainable
        print(f"question_start: Trainable params={question_start_trainable:,} | Total params={question_start_total:,} | Trainable ratio={question_start_trainable / question_start_total:.2%}")
        # Print total custom layer parameters
        print(f"Total custom layers: Trainable params={custom_trainable_params:,} | Total params={custom_total_params:,} | Trainable ratio={custom_trainable_params / custom_total_params:.2%}")


    def apply_history_dropout(self, question_embeds: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        """
        question_embeds: [B, T, D]
        question_ids:    [B, T]
        """
        if (not self.training) or self.history_dropout_p <= 0:
            return question_embeds
        if self.tokenizer is None:
            return question_embeds

        pad_id = self.tokenizer.pad_token_id
        valid = (question_ids != pad_id)  # [B,T] bool

        keep = (torch.rand(question_ids.shape, device=question_ids.device) > self.history_dropout_p) & valid

        if self.history_dropout_protect and self.history_dropout_protect > 0:
            keep[:, :self.history_dropout_protect] = True

        keep_f = keep.unsqueeze(-1).to(dtype=question_embeds.dtype)  # [B,T,1]

        if self.history_dropout_noise_std and self.history_dropout_noise_std > 0:
            noise = torch.randn_like(question_embeds) * self.history_dropout_noise_std
            dropped = question_embeds + noise
            question_embeds = question_embeds * keep_f + dropped * (1 - keep_f)
        else:
            question_embeds = question_embeds * keep_f

        return question_embeds


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                att_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None,
                question: Optional[torch.Tensor] = None,
                answer_types: Optional[torch.Tensor] = None,
                answer_start_indices: Optional[torch.Tensor] = None,
                ):
        """Model forward pass"""
        image_embeds = self.image_proj(input_ids)
        B, H, D = image_embeds.shape
        question_answer_embeds = self.lm.model.model.embed_tokens(caption)
        # question_answer_embeds = self.apply_history_dropout(question_answer_embeds, question)

        inputs_embeds = torch.cat((image_embeds, self.question_start.expand(B, -1, -1), question_answer_embeds), dim=1)

        labels = torch.cat([
            torch.full((B, H + 1), -100, device=DEVICE, dtype=torch.long),
            labels,
        ], dim=1)

        attention_mask = torch.cat([
            att_mask,
            torch.ones((B, 1), device=DEVICE, dtype=torch.long),
            mask2
        ], dim=1)

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=self.lm.config.use_cache
        )


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        multimodal_model = model.module if hasattr(model, 'module') else model

        custom_state_dict = {
            "image_proj": {k: v.detach().cpu() for k, v in multimodal_model.image_proj.state_dict().items()},
            "mode_embeddings": {k: v.detach().cpu() for k, v in multimodal_model.mode_embeddings.state_dict().items()},
            "question_start": multimodal_model.question_start.detach().cpu(),
        }

        checkpoint_step = self.state.global_step
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{checkpoint_step}")
        save_path = os.path.join(checkpoint_dir, "custom_layers.bin")
        torch.save(custom_state_dict, save_path)

        print(f"Custom layer weights saved: {save_path}")
        
        if isinstance(multimodal_model.lm, PeftModel):
            multimodal_model.lm.save_pretrained(checkpoint_dir)
            print(f"LoRA adapter saved to: {checkpoint_dir}")

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
        batch_size = len(features)
        
        qa_ids = [f["qa_ids"] for f in features]
        qa_masks = [f["qa_attention_mask"] for f in features]
        answer_start_indices = [f["answer_start_idx"] for f in features]
        answer_types = [f["answer_type"] for f in features]
        
        qa_ids = torch.tensor(np.array(qa_ids), dtype=torch.long)
        qa_masks = torch.tensor(np.array(qa_masks), dtype=torch.long)
        answer_start_indices = torch.tensor(answer_start_indices, dtype=torch.long)
        
        labels = qa_ids.clone()
        for i in range(batch_size):
            start_idx = answer_start_indices[i].item()
            labels[i, :start_idx] = -100
            labels[i, qa_masks[i] == 0] = -100

        answer_types_tensor = torch.tensor(answer_types)
        img_lens = [len(f["input_ids"]) for f in features]
        assert len(set(img_lens)) == 1, "Image token lengths in the same batch are inconsistent"
        
        return {
            "input_ids": torch.tensor(np.array([f["input_ids"] for f in features]), dtype=torch.float),
            "labels": labels,
            "att_mask": torch.tensor(np.array([f["attention_mask"] for f in features]), dtype=torch.long),
            "mask2": qa_masks,
            "question": qa_ids,
            "answer_start_indices": answer_start_indices,
            "answer_types": answer_types_tensor
        }





def compute_custom_loss(model: MultimodalModel, inputs: Dict, return_outputs: bool = False,
                        num_items_in_batch: Optional[int] = None):

    input_ids = inputs.get("input_ids")
    att_mask = inputs.get("att_mask")
    labels = inputs.get("labels")
    mask2 = inputs.get("mask2")
    question = inputs.get("question")
    answer_types = inputs.get("answer_types")
    answer_start_indices = inputs.get("answer_start_indices")

    outputs = model(
        input_ids=input_ids,
        att_mask=att_mask,
        labels=labels,
        mask2=mask2,
        question=question,
        answer_types = answer_types,
        answer_start_indices = answer_start_indices,
    )
    weighted_loss = outputs.loss

    return (weighted_loss, outputs) if return_outputs else weighted_loss




def process_punctuation(in_text):
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile("(\d)(\,)(\d)")
    
    out_text = in_text
    for p in punct:
        if (p + ' ' in in_text or ' ' + p in in_text) or (re.search(comma_strip, in_text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub("", out_text, re.UNICODE)
    return out_text

def process_digit_article(in_text):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}
    manual_map = { 'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    articles = ['a', 'an', 'the']
    
    out_text = []
    temp_text = in_text.lower().split()
    for word in temp_text:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            out_text.append(word)
        else:
            pass
    for word_id, word in enumerate(out_text):
        if word in contractions:
            out_text[word_id] = contractions[word]
    out_text = ' '.join(out_text)
    return out_text

def compute_vqa_accuracy(pred_answer, gt_answers):
    gt_answers = list(gt_answers)
    for i in range(len(gt_answers)):
        gt_answers[i] = gt_answers[i].replace('\n', ' ')
        gt_answers[i] = gt_answers[i].replace('\t', ' ')
        gt_answers[i] = gt_answers[i].strip()
    
    pred_answer = pred_answer.replace('\n', ' ')
    pred_answer = pred_answer.replace('\t', ' ')
    pred_answer = pred_answer.strip()
    
    if len(set(gt_answers)) > 1:
        for i in range(len(gt_answers)):
            gt_answers[i] = process_punctuation(gt_answers[i])
            gt_answers[i] = process_digit_article(gt_answers[i])
        pred_answer = process_punctuation(pred_answer)
        pred_answer = process_digit_article(pred_answer)
    
    gt_acc = []
    for i, gt_ans in enumerate(gt_answers):
        other_gt_ans = [item for j, item in enumerate(gt_answers) if j != i]
        matching_ans = [item for item in other_gt_ans if item == pred_answer]
        acc = min(1.0, float(len(matching_ans))/3.0)
        gt_acc.append(acc)
    
    avg_gt_acc = float(sum(gt_acc))/len(gt_acc)
    return avg_gt_acc

def evaluate_vqa(model, test_dataset, tokenizer, device, batch_size=5):
    """
    Evaluate VQA task
    Args:
        model: Multimodal model
        test_dataset: Test dataset
        tokenizer: Tokenizer
        device: Device
        batch_size: Batch size
    """
    model.eval()

    image_questions = {}
    image_features = {}


    print(f"Starting to process image samples, total samples: {len(test_dataset)}")
    for i in tqdm(range(len(test_dataset)), desc="Collecting questions and answers"):
        sample = test_dataset[i]
        input_ids = sample['input_ids']
        img_id = sample['image_name'].split('.')[0]
        # image_questions[img_id] = input_ids
        question = sample["question"]
        answer = sample["answer"]
        all_answers = sample.get("all_answers", [])
        image_id = int(sample["image_id"])

        if img_id not in image_questions:
            image_questions[img_id] = []
        
        image_questions[img_id].append({
            "question": question,
            "answer": answer,
            "all_answers": all_answers,
            "qa_ids": sample["qa_ids"],
            "answer_start_idx": sample["answer_start_idx"],
            "image_id": image_id,
            "answer_type": sample["answer_type"],
        })


        if img_id not in image_features:
            image_features[img_id] = {
                'input_ids': input_ids
            }

    print(f"Question and answer collection completed, found {len(image_features)} unique images")

    total_accuracy = 0.0
    total = 0
    
    # Category statistics
    yes_no_accuracy = 0.0
    yes_no_total = 0
    number_accuracy = 0.0
    number_total = 0
    other_accuracy = 0.0
    other_total = 0

    with torch.no_grad():
        for img_id in tqdm(image_features.keys(), desc="Generating answers one by one"):
            feat = image_features[img_id]
            input_ids = torch.tensor(feat['input_ids']).to(device).unsqueeze(0)
            
            for qa in image_questions[img_id]:
                question_ids = torch.tensor([qa['qa_ids'][:qa['answer_start_idx']]], dtype=torch.long, device=device)
                
                pred_answer = generate_response(
                    model,
                    input_ids=input_ids,
                    question_ids=question_ids,
                    return_attentions=False
                )[0].strip()

                all_answers = qa["all_answers"]
                acc = compute_vqa_accuracy(pred_answer, all_answers)
                total_accuracy += acc
                total += 1
                
                # Category statistics
                answer_type = qa.get('answer_type', 2)
                if answer_type == 0:
                    yes_no_accuracy += acc
                    yes_no_total += 1
                elif answer_type == 1:
                    number_accuracy += acc
                    number_total += 1
                else:
                    other_accuracy += acc
                    other_total += 1

    print(f"Generation completed, total questions: {total}")

    # Calculate accuracy
    accuracy = total_accuracy / total if total > 0 else 0
    yes_no_acc = yes_no_accuracy / yes_no_total if yes_no_total > 0 else 0
    number_acc = number_accuracy / number_total if number_total > 0 else 0
    other_acc = other_accuracy / other_total if other_total > 0 else 0
    
    print(f"\nVQA task evaluation results (using official VQA evaluation method):")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Yes/No accuracy: {yes_no_acc:.4f} (total {yes_no_total} questions)")
    print(f"Number accuracy: {number_acc:.4f} (total {number_total} questions)")
    print(f"Other accuracy: {other_acc:.4f} (total {other_total} questions)")

    return {
        "accuracy": accuracy,
        "yes_no_acc": yes_no_acc,
        "number_acc": number_acc,
        "other_acc": other_acc,
        "yes_no_count": yes_no_total,
        "number_count": number_total,
        "other_count": other_total
    }


def generate_response(model: MultimodalModel,
                      input_ids: Optional[torch.Tensor] = None,
                      question_ids: Optional[torch.Tensor] = None,
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
        question_start = model.question_start.to(dtype=input_dtype, device=model_device).expand(B, -1, -1)

        question_ids = question_ids.to(device=model_device)
        question_embeds = model.lm.model.embed_tokens(question_ids)
        inputs_embeds = torch.cat([img_embeds, question_start, question_embeds], dim=1)


        attention_mask = torch.ones(
            (B, inputs_embeds.shape[1]),
            dtype=torch.long,
            device=model_device
        )

        outputs = model.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            early_stopping=True,
            use_cache=True,
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=3,
        )
        
        generated_texts = []
        sequences = outputs.sequences

        for output in sequences:
            text = model.tokenizer.decode(output, skip_special_tokens=True).strip()
            generated_texts.append(text if text else "no answer generated")

        return generated_texts


def evaluate_saved_model(model_path=None, base_model="Qwen/Qwen2.5-7B", checkpoint_dir=None, use_quantized=True):
    """
    Args:
        model_path: Model save path, default to use the latest output_dir/final_model
        base_model: Base model name
        checkpoint_dir: Checkpoint directory path, used to load model from intermediate checkpoint
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if use_quantized else None

    use_checkpoint = checkpoint_dir is not None and os.path.exists(checkpoint_dir)
    load_path = checkpoint_dir if use_checkpoint else model_path
    
    print(f"Using {'checkpoint' if use_checkpoint else 'final model'} path: {load_path}")

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

    print("Mounting LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        load_path,
    )

    multimodal_model = MultimodalModel(model, tokenizer)

    custom_layers_path = os.path.join(load_path, "custom_layers.bin")
    custom_weights = torch.load(custom_layers_path, map_location=DEVICE)
    multimodal_model.image_proj.load_state_dict(custom_weights["image_proj"])
    multimodal_model.mode_embeddings.load_state_dict(custom_weights["mode_embeddings"])
    multimodal_model.question_start.data.copy_(custom_weights["question_start"])

    multimodal_model.model_dtype = next(multimodal_model.parameters()).dtype

    test_dataset = load_from_disk('test_dataset_vqa_vit_clean_6k')

    # Execute evaluation
    if test_dataset:
        print("\nStarting test set evaluation...")
        evaluate_vqa(multimodal_model, test_dataset, tokenizer, DEVICE)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multimodal VQA model training and evaluation parameters')
    parser.add_argument('--evaluate_only', type=str, choices=['True', 'False'], default='False',
                        help='Only perform evaluation, no training')
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Model path to use for evaluation')
    parser.add_argument('--resume_from_checkpoint', type=str, choices=['True', 'False'], default='False',
                        help='Whether to resume training from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=None,
                        help='Specify checkpoint directory path to load or resume from')
    parser.add_argument('--use_quantized', type=str, choices=['True', 'False'], default='True',
                        help='Whether to use quantized model, True for 4-bit quantization, False for non-quantization')
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
        r=16,
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

    train_dataset = load_from_disk('train_dataset_vqa_vit_clean_20k')
    val_dataset = load_from_disk('val_dataset_vqa_vit_clean_4k')
    test_dataset = load_from_disk('test_dataset_vqa_vit_clean_6k')

    output_dir = f"./checkpoints/Qwen2.5-7B_vqa_lora_{time.strftime('%y%m%d')}_20k_proj_ft"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.015,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        fp16=True,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    collator = MultimodalCollator(tokenizer=tokenizer)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint.lower() == 'true' and args.checkpoint_dir)

    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    tokenizer.save_pretrained(final_model_dir)

    if isinstance(model.lm, PeftModel):
        model.lm.save_pretrained(final_model_dir)

    custom_state_dict = {
        "image_proj": {k: v.detach().cpu() for k, v in model.image_proj.state_dict().items()},
        "mode_embeddings": {k: v.detach().cpu() for k, v in model.mode_embeddings.state_dict().items()},
        "question_start": model.question_start.detach().cpu(),
    }
    torch.save(custom_state_dict, os.path.join(final_model_dir, "custom_layers.bin"))


    evaluate_vqa(model, test_dataset, tokenizer, DEVICE)
