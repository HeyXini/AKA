# Multimodal Model Project

## Project Overview

This project implements a multimodal model based on Qwen2.5, supporting both image captioning and visual question answering (VQA) tasks. It includes scripts for model fine-tuning, cross-modal feature similarity analysis, and dataset processing.

## Directory Structure

```
├── Qwen/                # Qwen2.5 model files
├── dataset/             # Dataset processing scripts
├── vit-mae-model/       # ViT-MAE model for image feature extraction
├── cross_modal_similarity.py  # Cross-modal feature similarity analysis
├── image-qwen2.5.py     # Image captioning model fine-tuning
├── vqa-qwen2.5.py       # Visual question answering model
├── vqa.py               # VQA utility functions
├── vqaEval.py           # VQA evaluation utilities
├── vqa_data_process.py  # VQA data processing
└── image_data_process.py # Image data processing
```



## Main Features

### 1. Cross-Modal Feature Similarity Analysis

The `cross_modal_similarity.py` script implements:
- Extraction of FFN features from both images and text
- Calculation of cosine similarity between image and text features
- Visualization of similarity distributions
- Comparison of different pooling strategies and layers

### 2. Image Captioning

The `image-qwen2.5.py` script provides:
- Fine-tuning of Qwen2.5 for image captioning
- Attention visualization for caption tokens
- Evaluation using BLEU, ROUGE, CIDEr, and METEOR metrics

### 3. Visual Question Answering (VQA)

The `vqa-qwen2.5.py` script includes:
- Fine-tuning of Qwen2.5 for VQA tasks
- Evaluation using VQA official evaluation metrics
- Support for different answer types (yes/no, number, other)

## Model Architecture

The multimodal model architecture consists of:

1. **Image Encoder**: ViT-MAE for image feature extraction
2. **Text Encoder**: Qwen2.5 language model
3. **Cross-Modal Fusion**: Dedicated projection layers QKVO
4. **LoRA Adaptation**: Parameter-efficient fine-tuning for multimodal tasks

## Usage

### Fine-tuning for Image Captioning

```bash
python image-qwen2.5.py --evaluate_only False --use_quantized True
```

### Fine-tuning for VQA

```bash
python vqa-qwen2.5.py --evaluate_only False --use_quantized True
```

### Cross-Modal Similarity Analysis

```bash
python cross_modal_similarity.py
```

### Evaluating Saved Models

For image captioning:
```bash
python image-qwen2.5.py --evaluate_only True --model_path ./checkpoints/final_model
```

For VQA:
```bash
python vqa-qwen2.5.py --evaluate_only True --model_path ./checkpoints/final_model
```

## Evaluation Metrics

### Image Captioning
- BLEU (1-4)
- ROUGE-L
- CIDEr
- METEOR
- SPICE (requires Java)

### VQA
- Overall accuracy
- Yes/No accuracy
- Number accuracy
- Other accuracy


## Notes

- The model uses 4-bit quantization by default for memory efficiency
- Training requires a GPU with sufficient memory (recommended: 16GB+)
- Datasets should be preprocessed and stored in the appropriate format
- Attention visualization requires the model to use eager attention implementation

## License

This project is for research purposes only. Please refer to the original Qwen2.5 license for commercial use.

## Acknowledgments

- Qwen2.5 model from Alibaba Cloud
- ViT-MAE for image feature extraction
- COCO evaluation metrics for image captioning
- VQA evaluation metrics for visual question answering