# Resource-Efficient LLM Fine-Tuning with Alignment

This project provides scripts, configs, and notebooks for **resource-efficient fine-tuning of large language models (LLMs)** using:
- **Supervised Fine-Tuning (SFT)**
- **Direct Preference Optimization (DPO)**
- **DeepSpeed** for memory-efficient training

## 📂 Project Structure
- `train_sft.py` – run supervised fine-tuning
- `train_dpo.py` – run preference optimization
- `custom_dpo.py` – custom DPO trainer implementation
- `configs/` – YAML/JSON configs for datasets, models, trainers, and DeepSpeed
- `sagemaker/` – example notebooks for running SFT & DPO on AWS SageMaker
- `requirements.txt` – dependencies

## 🚀 Quick Start
Install dependencies:
```bash
pip install -r requirements.txt
```

Run SFT:
```bash
python train_sft.py
```

Run DPO:
```bash
python train_dpo.py
```
