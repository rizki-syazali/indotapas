# IndoTaPas: A TaPas-Based Model for Indonesian Table Question Answering
We introduce IndoTaPas, a TaPaS-based language model pre-trained on the Indonesian Wikipedia dataset, comprising 1,636,656 text-table pairs. To further adapt the model to the TQA task, we fine-tuned it using an annotated Indonesian TQA dataset consisting of 2,507 question–table pairs featuring hierarchical structures and complex reasoning types

# Models
The pre-trained and fine-tuned models are provided directly within this repository in the `models/` directory. For ease of integration with the `transformers` library, all models also available via the Hugging Face Model Hub.

| Model Variant | Training Strategy / Description | Exact Match (EM) | Local Path in Repo | Hugging Face Link |
| :--- | :--- | :---: | :--- | :--- |
| **IndoTaPas (Pre-trained)** | pre-trained from scratch using a masked language modeling objective on a massive corpus of 1.6 million Indonesian text-table pairs. | - | [`models/pretraining/indotapas_masklm`](./models/pretraining/indotapas_masklm) | [link](https://huggingface.co/rizki-syazali/tapas_masklm_id_3.0) |
| **IndoTaPas (One-Stage)** | Fine-tuned directly on the IndoHiTab manual dataset. | 37.25% | [`models/finetuning/indotapas_one_stage`](./models/finetuning/indotapas_one_stage) | [Link](https://huggingface.co/rizki-syazali/tapasid_finetuned_itqa) |
| **IndoTaPas (Two-Stage)** | 1st fine-tuning on HiTab (automatic), 2nd fine-tuning on IndoHiTab (manual). | **45.22%** | [`models/finetuning/indotapas_two_stage`](./models/finetuning/indotapas_two_stage) | [Link](https://huggingface.co/rizki-syazali/tapasid_finetuned_hitab_to_itqa) |


## Pretraining

To build the foundational knowledge of **IndoTaPas**, the model is pre-trained from scratch using a Masked Language Modeling (MLM) objective. Information about the pre-taining dataset can be found [here](./datasets/IndoWikiTableText/).

*Note:* you can skip pre-training and just use the pre-trained checkpoints provided above.


### Vocabulary
The model uses an Indonesian-specific vocabulary to ensure optimal tokenization. We utilize the IndoBERT vocabulary file located at `vocab_file/indobert_vocab.txt`.

### Usage

You can start the pretraining process by running this pretraining script:
```
python pretraining.py \
    --masklm_model google/tapas-base \
    --vocab_file vocab_file/indobert_vocab.txt \
    --model_version v1.0 \
    --resume False
```

## Fine-tuning

To adapt the pre-trained IndoTaPas model for the Table Question Answering (TQA) task, we fine-tune it using the `TapasForQuestionAnswering` architecture. To answer our core research questions, we explore two distinct fine-tuning strategies using the **IndoHiTab** dataset and its machine-translated variants. Information about this dataset can be found [here](./datasets/IndoHiTab/).

### Fine-tuning Strategies
1. **One-Stage Fine-tuning:** The model is trained directly on the manually translated `IndoHiTab` dataset.
2. **Two-Stage Fine-tuning (Best Performance):** The model is first fine-tuned on an automatically translated augmentation dataset (`IndoHiTab-EXT-MT`), followed by a second fine-tuning stage on the high-quality, manually translated `IndoHiTab` dataset. This approach yields our state-of-the-art Exact Match (EM) score of 45.22%.

### Usage

To run the fine-tuning script, you must specify the paths to your pre-trained model, vocabulary file, and dataset directory. The script expects the dataset directory to contain the pre-processed `train.pkl` and `test.pkl` files.

**Example 1: One-Stage Fine-tuning**
```
python finetune.py \
    --masklm_model models/pretraining/indotapas_masklm \
    --vocab_file vocab_file/indobert_vocab.txt \
    --input_dir data/IndoHiTab \
    --output_dir models/finetuning/indotapas_one_stage \
    --epoch 4 \
    --batch_size 32
```

## Citation

If you plan to use  `IndoTaPas`, `IndoWikiTableText`, or `IndoHiTab` in your project, please consider citing our [paper]():

``` 

```