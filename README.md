# IndoTaPas: A TaPas-Based Model for Indonesian Table Question Answering
We introduce IndoTaPas, a TaPaS-based language model pre-trained on the Indonesian Wikipedia dataset, comprising 1,636,656 text-table pairs. To further adapt the model to the TQA task, we fine-tuned it using an annotated Indonesian TQA dataset consisting of 2,507 question–table pairs featuring hierarchical structures and complex reasoning types

# Models
Due to the large size of the models, we are unable to host them directly in this repository. Instead, all models are available via the [Hugging Face Model Hub](https://huggingface.co/), allowing easy integration with the `transformers` library.

|  Fine-tuning Strategy | Description | Exact Match (%) | Hugging Face Link |
|:----------------------:|:----------------------|:-----------------:|:-------------------:|
| one-stage | fine-tuning on IndoHiTab manual | 37.25 |  [link](rizki-syazali/tapasid_finetuned_itqa) |
| two-stage | 1st fine-tuning on HiTab automatic , and 2nd fine-tuning on IndoHiTab manual ) | 45.22 |[link](rizki-syazali/tapasid_finetuned_hitab_to_itqa) |
