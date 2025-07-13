# ATRIE: Automating Legal Concept Interpretation with LLMs

[![ACL 2025](https://img.shields.io/badge/ACL%202025-Main%20Conference-blue)](https://2025.aclweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.01743-b31b1b.svg)](https://arxiv.org/abs/2501.01743)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/KcLuo/ATRIE)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains the official implementation of the paper "**Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation**", accepted to the main conference of ACL 2025.

## 📖 Dataset

Our Legal Concept Entailment dataset has been released and is now available on Hugging Face.

You can access it here: 👉 [**KcLuo/ATRIE**](https://huggingface.co/datasets/KcLuo/ATRIE)

## 🚀 Quick Start

**Installation**

```bash
git clone https://github.com/lkc233/ATRIE.git
cd ATRIE
conda env create -f environment.yml
```

**Reproducing Paper Results**

```bash
bash src/bash_scripts/run.sh
```

## 📂 Code Structure

- **`src/bash_scripts`**: Contains shell scripts to run experiments.
  - `run.sh`: The main script to reproduce the results in the paper. 
  - `start_qwen_*_servers.sh` / `stop_qwen_*_servers.sh`: Scripts to start and stop the Qwen model servers.
  - `vllm_Qwen*.sh`: Scripts for running specific vLLM servers.
- **`src/configs`**: Configuration files for the llms.
- **`src/data_preparation`**: Code for data preparation.
  - `data_split.py`: Splits the data into training and test sets.
  - `mongodb_data_filter.py`: Code for the step 1 of our Interpreter. (You may need a mongodb server of case database running.)
  - `llm_data_filter.py`: Using LLMs to filter the relevant cases. (Step 2 of our Interpreter.)
  
- **`src/evaluate`**: Script for evaluating the model's performance.
  - `evaluate.py`: The main script for evaluation.
- **`src/generate_interpretation`**: Scripts for generating interpretations from the models.
  - `generate_interpretation.py`: Generates interpretations for legal concepts.
  - `extract_reason.py`: Extracts reasons from court views (Step 2 of our Interpreter).
- **`src/judgement_pred`**: Script for predicting judgments.
  - `judgement_pred.py`: Code for performing the LCE task.
- **`src/my_scripts`**: Utility scripts.
  - `utils.py`: Utility functions.
  - `chat.py`: A script for interactive chat with the llms.

🤝 **Citation**

If you use this code or find our work helpful, please cite our paper:

```bibtex
@misc{luo2025automatinglegalconceptinterpretation,
      title={Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation}, 
      author={Kangcheng Luo and Quzhe Huang and Cong Jiang and Yansong Feng},
      year={2025},
      eprint={2501.01743},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.01743}, 
}
```
