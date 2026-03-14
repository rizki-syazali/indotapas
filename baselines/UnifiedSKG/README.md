# Baseline Experiment 2: UnifiedSKG

This repository section details the implementation of our second baseline, which represents modern state-of-the-art generative approaches for the Table Question Answering (TableQA) task. For this experiment, we adopted the [UnifiedSKG](https://github.com/xlang-ai/UnifiedSKG) framework.

## Overview

UnifiedSKG is a unified text-to-text architecture originally designed for English structured knowledge grounding tasks. To adapt this framework effectively for the Indonesian context using the **IndoHiTab** dataset, we implemented the following modifications:

* **Model Adaptation:** We replaced the original English-centric T5 base model with two relevant pre-trained language models:
  * `mT5-base`: A multilingual language model.
  * `idT5-base`: An Indonesian-specific language model.
* **Data Representation:** The tables are linearized into a sequential text format. The models take the concatenated linearized table and the natural language question as input.
* **Generative Training:** Both variants (UnifiedSKG-mT5-base and UnifiedSKG-idT5-base) are fine-tuned to directly generate the final answer string from the input, bypassing the need to generate intermediate logical forms or programs.

## Setup
In order to include third-party dependencies in baseline repository, make sure to clone recursively, e.g.:

```
git clone --recurse-submodules https://github.com/xlang-ai/UnifiedSKG.git
cd UnifiedSKG
```

To establish the environment run this code in the shell:

``````
conda env create -f py3.7pytorch1.8.yaml
conda activate py3.7pytorch1.8new
pip install datasets==1.14.0

# The following line to be replaced depending on your cuda version.
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
``````

### Applying IndoHiTab Configurations and Data

To adapt the UnifiedSKG framework for our experiment, copy the custom files and directories provided in this repository directly into the root of your cloned `UnifiedSKG` directory. 

Ensure your final directory structure includes the following additions:

**1. Configurations (`configure/`)**
Merge our custom configuration files into the existing `configure` directory:
* `configure/META_TUNING/indohitab.cfg`
* `configure/Salesforce/idT5_base_finetune_indohitab.cfg`
* `configure/Salesforce/mT5_base_finetune_indohitab.cfg`

**2. Local Datasets (`local_datasets/`)**
Place the IndoHiTab dataset splits inside the `local_datasets` folder:
* `local_datasets/indohitab/train.json`
* `local_datasets/indohitab/validation.json`
* `local_datasets/indohitab/test.json`

**3. Pretrained Models (`pretrained_models/`)**
Create a `pretrained_models` directory to store the downloaded pre-trained model weights:
* `pretrained_models/idt5-base/`
* `pretrained_models/mt5-base/`

## Usage

### Environment setup
Activate the environment by running
``````shell
conda activate py3.7pytorch1.8new
``````

### WandB setup

Setup [WandB](https://wandb.ai/) for logging (registration needed):
``````shell
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_TEAM_NAME
``````

### Training

idT5-base finetuning on IndoHiTab (1 GPUs, 128 batch size)
``````shell
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 \
  train.py \
  --seed 42 \
  --cfg Salesforce/idT5_base_finetune_indohitab.cfg \
  --run_name idT5_base_finetune_indohitab \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 400 \
  --adafactor true \
  --learning_rate 5e-5 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --generation_num_beams 4 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --ddp_find_unused_parameters true
``````


idT5-base finetuning on IndoHiTab (1 GPUs, 128 batch size)
``````shell
python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 \
  train.py \
  --seed 42 \
  --cfg Salesforce/mT5_base_finetune_indohitab.cfg \
  --run_name mT5_base_finetune_indohitab \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --metric_for_best_model avr \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 400 \
  --adafactor true \
  --learning_rate 5e-5 \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 16 \
  --generation_num_beams 4 \
  --generation_max_length 128 \
  --input_max_length 1024 \
  --ddp_find_unused_parameters true
``````

If you want to resume training, remove the ``--overwrite_output_dir`` flag from the above command

