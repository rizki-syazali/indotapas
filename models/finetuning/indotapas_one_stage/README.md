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
metrics:
  - exact_match
  - f1
---

# Model Card for IndoTaPas (One-Stage Fine-tuning)

## Model Details

### Model Description

**IndoTaPas (One-Stage)** is a TaPas-based model specifically adapted and fine-tuned for the Table Question Answering (TQA) task in the Indonesian language. It is designed to extract precise answers from structured tabular data based on natural language questions. 

This specific variant was fine-tuned using a **one-stage strategy**, meaning it was trained directly on the high-quality, manually translated **IndoHiTab** dataset without prior augmentation. 

- **Developed by:** Muhammad Rizki Syazali & Evi Yulianti
- **Model type:** Table Parser (TaPas) for Extractive Question Answering
- **Language(s) (NLP):** Indonesian (`id`)
- **Finetuned from model:** IndoTaPas MaskedLM (pre-trained from scratch on 1.6M Indonesian WikiTableText pairs)

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

model_name = "rizki-syazali/tapasid_finetuned_itqa"
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

The model was fine-tuned on the **IndoHiTab** dataset, which consists of manually translated English-to-Indonesian table-question pairs. Specifically, the "Flattened" version of the tables was used, where multi-level hierarchical headers were concatenated into single-level headers.
- **Train Set Size:** 2,057 instances.

### Training Procedure

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision
- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Epochs:** 4
- **Batch Size:** 32
- **Scheduler:** Linear (with 0 warmup steps)

## Evaluation

### Testing Data & Metrics

#### Testing Data
The model was evaluated on the unseen test split of the **IndoHiTab** dataset, comprising **502** question-table pairs.

#### Metrics
- **Exact Match (EM):** The primary metric measuring whether the predicted cell coordinates exactly match the ground truth coordinates.
- **F1 Score:** Due to the decoding mechanism mentioned in the limitations, the F1 score mirrors the EM score exactly for this model.

### Results

| Model Variant | Fine-Tuning Strategy | Exact Match (EM) | F1 Score |
| :--- | :--- | :---: | :---: |
| **IndoTaPas (One-Stage)** | Manual Data Only (IndoHiTab) | **37.25%** | **37.25%** |

#### Summary
The one-stage IndoTaPas model achieves a strong baseline of 37.25% EM, significantly outperforming early neural semantic parsers (LatentAlignment at 19.12%) and remaining highly competitive against zero-shot generative LLMs on the Indonesian TQA task.

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
``` -->