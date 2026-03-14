# Baseline Experiment 3: Zero-Shot Large Language Models (TableCoT)

This section outlines the implementation of our third baseline category, which explores the zero-shot capabilities of massive pre-trained Large Language Models (LLMs) on the Indonesian TableQA task without any task-specific fine-tuning. 

## Overview

For this experiment, we adopted the in-context learning framework described by Chen et al. ([TableCoT](https://github.com/TIGER-AI-Lab/TableCoT)). We evaluated the following models to investigate their ability to solve the task without prior exposure to the training set:
* **GPT-3** (`davinci-003`)
* **Gemma-3-4B-IT**
* **Meta-Llama-3-8B-Instruct**

## Methodology & Prompt Design

To effectively prompt these LLMs, we structured the inputs using the following approach:
* **Table Linearization:** The tabular data is linearized into a text-based **Markdown** format to serve as the context.
* **Prompt Construction:** The final input prompt is constructed by concatenating the linearized Markdown table with the corresponding natural language question.
* **System Instructions:** We included a specific Indonesian system instruction to define the model's role and strictly constrain the output format. This ensures the model focuses solely on extracting the correct answer from the provided table context.

## Inference Configuration

To ensure strict reproducibility, all LLM inferences were conducted using the models' default generation configurations. The specific generation hyperparameters are set as follows:
* **Sampling:** Enabled
* **Temperature:** `0.7`
* **Max New Tokens:**

## Usage

The inference script (`prompt.py`) is designed to evaluate Large Language Models (LLMs) on the IndoHiTab dataset using in-context learning. It supports both zero-shot and few-shot evaluations, automatic prompt construction with Markdown table linearization, and batched inference.

### Prerequisites
Before running the script, ensure you have the required libraries installed and your Hugging Face access token configured. 

**Important Note on Gated Models & Access Tokens:**
The script requires an `HF_ACCESS_TOKEN` because certain models evaluated in this baseline (such as `Meta-Llama-3`) are **gated models**. To give authors more control over how their models are used, the Hugging Face Hub requires users to explicitly request access and agree to share their contact information or accept specific terms before downloading the model files. 

* **Action Required:** You must first visit the respective model page on Hugging Face (e.g., Meta-Llama-3) and request access using your own account. Once granted, replace the hardcoded `HF_ACCESS_TOKEN` in `prompt.py` with your own valid Hugging Face token before running the script.

### Command-Line Arguments

You can customize the evaluation using the following arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | Hugging Face model ID to evaluate. | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `--dataset_path` | Path to the local huggingface dataset. | `datasets_with_shots` |
| `--num_shots` | Number of few-shot examples to include in the prompt. | `0` (Zero-shot) |
| `--temperature` | Sampling temperature. `0` implies greedy decoding. | `0.7` |
| `--batch_size` | Number of samples processed per batch. | `4` |
| `--chat_model` | Flag to format prompts using the model's specific chat template. | `False` |
| `--seed` | Random seed for reproducibility. | `42` |
| `--dry_run` | Flag to print the generated prompt and exit without running inference. | `False` |
| `--start` / `--end` | Indices to slice the test set for partial evaluations. | `None` |
| `--max_input_tokens` | Truncate prompts exceeding this token limit to avoid OOM errors. | `None` |

### Example Commands

**1. Basic Zero-Shot Evaluation (Default)**
Run the default Llama-3 model in a zero-shot setting (as described in our paper):
```
python prompt.py 
```