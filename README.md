# Multimodal Medical Report Generation from Chest X-Rays

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚧 Project Status: Ongoing 🚧

This project is a deep-learning pipeline to automatically generate radiology reports from chest X-ray images. The core architecture is complete and functional, and the model is currently being trained and evaluated. Future work will focus on implementing more advanced techniques to improve factual accuracy and clinical robustness.

## 📜 Description

This project implements a state-of-the-art multimodal architecture that combines a Vision Transformer (ViT) for image understanding and a pre-trained Large Language Model (LLM) for text generation. The system is designed to analyze a radiological image and produce a coherent, clinically relevant draft report. This serves as a proof-of-concept for AI-assisted diagnostic tools aimed at reducing radiologist workload and standardizing report quality.

## ✨ Key Features

- **Vision Encoder**: Uses a pre-trained Vision Transformer (`vit_base_patch16_224_in21k`) to extract a rich sequence of features from chest X-ray images.
- **Language Model**: Leverages a powerful, pre-trained LLM (`TinyLlama-1.1B-Chat-v1.0`) as the text generation engine.
- **Efficient Fine-Tuning**: Employs **LoRA (Low-Rank Adaptation)** to efficiently fine-tune the LLM for the medical domain without retraining the entire model.
- **Robust Data Pipeline**: Includes automated preprocessing, cleaning, and augmentation for both image and text data.
- **Advanced Training**: The training loop uses modern techniques like the **AdamW optimizer**, a **cosine learning rate scheduler**, and **automatic mixed-precision** for stable and efficient training.

## 🔧 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    Download the Indiana University Chest X-Ray dataset from Kaggle and place its contents in the `data/raw/` directory.
    - **Link**: [IU X-Ray Dataset](https://www.kaggle.com/datasets/raddar/indiana-university-chest-x-rays)

## 🚀 Usage / Workflow

The project is executed via a sequence of scripts. Run them from the main project directory.

1.  **Prepare the Data:**
    This script cleans the data, filters out corrupt images and short reports, and creates the final dataset.
    ```bash
    python -m src.run_preprocessing
    ```

2.  **Train the Model:**
    This script loads the preprocessed data, builds the model, and runs the full training and validation loop, saving the best model adapter to the `models/` directory.
    ```bash
    python -m src.train
    ```

3.  **Evaluate the Model:**
    After training, run this script to get performance metrics (BLEU, ROUGE) on the test set.
    ```bash
    python -m src.run_evaluation
    ```

4.  **Generate a Report (Inference):**
    Use this script to generate a report for a single image.
    ```bash
    python -m src.inference --image_path /path/to/your/image.png
    ```

## 🔮 Future Work

- [ ] Implement advanced training methodologies like Reinforcement Learning (from Section III of the research plan).
- [ ] Integrate a Retrieval-Augmented Generation (RAG) component to further reduce model hallucination (from Section 2.4).
- [ ] Experiment with larger vision and language models.

## 📄 License

This project is licensed under the MIT License.
