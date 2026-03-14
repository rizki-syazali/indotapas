---
language: 
  - id
tags:
  - table-language-model
  - tapas
  - indonesian
  - masked-language-modeling
  - pretraining
license: apache-2.0
datasets:
  - IndoWikiTableText
---

# Model Card for IndoTaPas (MaskedLM Pre-training)

## Model Details

### Model Description

**IndoTaPas (MaskedLM)** is the foundational, pre-trained TaPas (Table Parser) model for the Indonesian language. It was pre-trained from scratch to understand the structural and semantic alignment between natural language text and tabular data in Indonesian. 

The model was trained using a Masked Language Modeling (MLM) objective with Whole-Word Masking on a massive corpus of Indonesian Wikipedia text-table pairs. This model serves as a strong starting point for various tabular downstream tasks in Indonesian, such as Table Question Answering (TQA), Table Fact Verification, and Table-based Text Generation.

- **Developed by:** Muhammad Rizki Syazali & Evi Yulianti
- **Model type:** Table Parser (TaPas) for Masked Language Modeling
- **Language(s) (NLP):** Indonesian (`id`)
- **Finetuned from model:** Pre-trained from scratch using `google/tapas-base` configuration and an Indonesian-specific vocabulary (IndoBERT).

### Model Sources

- **Repository:** [GitHub - IndoTaPas](https://github.com/rizki-syazali/indotapas)
- **Paper:** "IndoTaPas: A TaPas-Based Model for Indonesian Table Question Answering" (Expert Systems with Applications, 2026)

## Uses

### Direct Use

As a pre-trained base model, **it is not intended for direct use in end-user applications**. It is designed to be fine-tuned on downstream tabular tasks. You can use this model directly only for masked word prediction within a table/text context.

### Downstream Use

This model is intended to be fine-tuned for tasks such as:
- **Table Question Answering (Extractive):** e.g., fine-tuning on the IndoHiTab dataset (see our one-stage and two-stage models).
- **Table Entailment / Fact Verification:** Verifying if a statement is true
