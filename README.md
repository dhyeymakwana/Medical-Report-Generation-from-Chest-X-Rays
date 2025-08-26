# Multimodal Medical Report Generation from Chest X-Rays

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚧 Project Status: Ongoing 🚧

This project is a deep-learning pipeline to automatically generate radiology reports from chest X-ray images. The core architecture is complete and functional, and the model is currently being trained and evaluated. Future work will focus on implementing more advanced techniques to improve factual accuracy and clinical robustness.

## 📖 Overview
This project presents a sophisticated deep learning pipeline for the automatic generation of radiological reports from chest X-ray images. By harnessing a state-of-the-art multimodal architecture, this work aims to bridge the gap between visual diagnostic data and clinical narrative, offering a powerful tool to assist radiologists, reduce workload, and enhance the consistency of medical reporting.

The core of this project lies in the synergy between a **Vision Transformer (ViT)** for nuanced image interpretation and a pre-trained **Large Language Model (LLM)** for articulate and context-aware text generation. This serves as a robust proof-of-concept for next-generation AI-assisted diagnostic tools, with a focus on clinical relevance and factual accuracy.

## ✨ Key Features
- **Advanced Vision Encoder**: Utilizes a pre-trained Vision Transformer (`vit_base_patch16_224_in21k`) to capture intricate patterns and clinical indicators from chest radiographs, transforming visual information into a rich, sequential feature representation.  
- **Intelligent Language Model**: Employs the powerful and efficient `TinyLlama-1.1B-Chat-v1.0` as the core text generation engine, providing a strong foundation in natural language understanding and generation.  
- **Efficient Fine-Tuning with LoRA**: Implements Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. This allows the LLM to be specialized for the medical domain without the computational expense of retraining the entire model.  
- **Automated Data Pipeline**: Comprehensive data preprocessing, cleaning, and augmentation for both images and corresponding textual reports.  
- **State-of-the-Art Training Techniques**: AdamW optimizer, cosine learning rate scheduler, and mixed-precision (AMP) training for efficiency and stability.  

## 🔧 Getting Started

### Prerequisites
- Python 3.11+  
- PyTorch 2.1+  
- NVIDIA GPU with CUDA support (recommended for training)  

### Installation
Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Set up a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Download the Dataset
This project uses the **Indiana University Chest X-Ray dataset**. Please download it from Kaggle and place its contents into the `data/raw/` directory.  

🔗 [IU X-Ray Dataset on Kaggle](https://www.kaggle.com/datasets)

---

## 🚀 Project Workflow
The project is structured into modular scripts that should be run from the project root.

### Data Preprocessing
Cleans raw data, filters corrupted images/incomplete reports, and prepares analysis-ready dataset.
```bash
python -m src.run_preprocessing
```

### Model Training
Orchestrates the full training loop with multimodal architecture and saves the best LoRA adapter to `models/`.
```bash
python -m src.train
```

### Model Evaluation
Evaluates the trained model on the test set using BLEU and ROUGE metrics.
```bash
python -m src.run_evaluation
```

### Inference: Generate a Report
Generate a clinical report from a new chest X-ray.
```bash
python -m src.inference --image_path /path/to/your/image.png
```

---

## 🔮 Future Directions
- [ ] **Reinforcement Learning from Human Feedback (RLHF)**: Further refine factual accuracy and clinical alignment.  
- [ ] **Retrieval-Augmented Generation (RAG)**: Ground generations in trusted medical knowledge bases.  
- [ ] **Scaling Up Models**: Experiment with larger vision and language backbones.  

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
