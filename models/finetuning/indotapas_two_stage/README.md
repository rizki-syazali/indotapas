---
language: 
  - id
tags:
  - table-question-answering
  - tapas
  - indonesian
  - indohitab
  - extractive-qa
datasets:
  - IndoHiTab
  - IndoHiTab-EXT-MT
metrics:
  - exact_match
  - f1
---

# Model Card for IndoTaPas (Two-Stage Fine-tuning)

## Model Details

### Model Description

**IndoTaPas (Two-Stage)** is our state-of-the-art TaPas-based model specifically adapted and fine-tuned for the Table Question Answering (TQA) task in the Indonesian language. It is designed to extract precise answers from structured tabular data based on natural language questions. 

This specific variant is our **best-performing model**, achieving an Exact Match (EM) of 45.22%. It was trained using a **two-stage fine-tuning strategy**:
1. **Stage 1 (Augmentation):** Fine-tuned on `IndoHiTab-EXT-MT` (automatically translated data) to build broad reasoning capabilities.
2. **Stage 2 (Adaptation):** Further fine-tuned on the high-quality, manually translated `IndoHiTab` dataset for precise domain adaptation.

- **Developed by:** Muhammad Rizki Syazali & Evi Yulianti
- **Model type:** Table Parser (TaPas) for Extractive Question Answering
- **Language(s) (NLP):** Indonesian (`id`)
- **Finetuned from model:** IndoTaPas MaskedLM -> IndoTapas One Stage

### Model Sources

- **Repository:** [GitHub - IndoTaPas](https://github.com/rizki-syazali/indotapas)
- **Paper:** "IndoTaPas: A TaPas-Based Model for Indonesian Table Question Answering" (Expert Systems with Applications, 2026)

## Uses

### Direct Use

The model is intended to be used for **extractive table question answering** in Indonesian. Given a flattened, 1-dimensional table and a corresponding question, the model will output the coordinates of the cell(s) containing the correct answer. 

### Out-of-Scope Use

- The model is **not** generative; it cannot synthesize new text or generate conversational responses. It only extracts existing cell values.
- Due to architectural constraints applied during the dataset filtering phase, the model is not optimized for questions that strictly require **header selection** as the final answer.

## Bias, Risks, and Limitations

- **"All-or-Nothing" Decoding:** When the model fails to predict the exact complete set of cell coordinates, its current decoding mechanism defaults to returning an empty array. This results in no partial overlap, meaning the Exact Match (EM) and F1 scores are identical.
- **Domain Limitation:** While pre-trained on diverse Wikipedia tables, its fine-tuning is heavily localized to the characteristics of the IndoHiTab (StatCan, ToTTo, NSF) data distributions.

## How to Get Started with the Model

You can load the model using the `transformers` library:

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

model_name = "rizki-syazali/tapasid_finetuned_hitab_to_itqa"
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base") # using base tokenizer with custom vocab
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Example Table and Question
data = {'Nama': ['Budi', 'Siti'], 'Umur': ['25', '30']}
table = pd.DataFrame.from_dict(data)
queries = ["Berapa umur Siti?"]

inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
outputs = model(**inputs)

# Predict answer coordinates
predicted_answer_coordinates, = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach())
print(predicted_answer_coordinates)
```

## Training Details

### Training Data

The model was fine-tuned using a combination of machine-translated and human-translated datasets:
1. **IndoHiTab-EXT-MT:** 2,914 instances of automatically translated data (used in Stage 1).
2. **IndoHiTab (Manual):** 2,057 instances of high-quality, human-translated data (used in Stage 2).

The "Flattened" version of the tables was used, where multi-level hierarchical headers were concatenated into single-level headers.

### Training Procedure

#### Training Hyperparameters (Applied to both stages)

- **Training regime:** fp16 mixed precision
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Epochs:** 4 (per stage)
- **Batch Size:** 32
- **Scheduler:** Linear (with 0 warmup steps)

## Evaluation

### Testing Data & Metrics

#### Testing Data
The model was evaluated on the unseen test split of the **IndoHiTab** dataset, comprising **502** high-quality, manually translated question-table pairs.

#### Metrics
- **Exact Match (EM):** The primary metric measuring whether the predicted cell coordinates exactly match the ground truth coordinates.
- **F1 Score:** Due to the decoding mechanism mentioned in the limitations, the F1 score mirrors the EM score exactly for this model.

### Results

| Model Variant | Fine-Tuning Strategy | Exact Match (EM) | F1 Score |
| :--- | :--- | :---: | :---: |
| **IndoTaPas (Two-Stage)** | Stage 1 (MT Data) + Stage 2 (Manual Data) | **45.22%** | **45.22%** |

#### Summary
The Two-Stage IndoTaPas model achieves the state-of-the-art Exact Match (EM) score of 45.22% on the Indonesian TQA task. This performance significantly outperforms early neural semantic parsers, UnifiedSKG variants, and even surpasses massive zero-shot generative models like Meta-Llama-3-8B-Instruct (44.02% EM) in exact retrieval precision.

<!-- ## Citation 

If you use this model or the IndoHiTab dataset, please cite our paper:

**BibTeX:**
```bibtex
@article{syazali2026indotapas,
  title={IndoTaPas: A TaPas-Based Model for Indonesian Table Question Answering},
  author={Syazali, Rizki and Co-authors},
  journal={Expert Systems with Applications},
  year={2026}
} 
```-->
