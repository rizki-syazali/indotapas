from datasets import load_dataset,load_from_disk, get_dataset_split_names, Dataset
from transformers import default_data_collator, DataCollatorForWholeWordMask
from transformers import TapasTokenizer,TapasConfig,TapasForMaskedLM
from transformers import TrainingArguments, Trainer as HFTrainer
from transformers import get_scheduler
import multiprocessing
import torch
import pandas as pd
import math
import os
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os
import argparse


class Trainer():
    def __init__(self, args):
        self.model_version = args.model_version
        self.masklm_model = args.masklm_model
        self.vocab_file = args.vocab_file
        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base", vocab_file=self.vocab_file)
        self.data_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.model = TapasForMaskedLM.from_pretrained(self.masklm_model) if args.resume else TapasForMaskedLM(TapasConfig())


    def whole_word_masking_data_collator(self,batch):
        # batch format is list of dict
        # print(batch)
        if "input_ids" not in batch[0]:
            #for training
            new_inputs = []
            for input in batch:
                table = pd.DataFrame(data = input["table"]["data"], columns = input["table"]["header"]).astype(str)
                table = table.applymap(str.lower)
                table.columns = table.columns.str.lower()

                tokenized_input = self.tokenizer(
                    table=table,
                    queries=input["text"].lower(),
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    max_row_id = 256,
                    max_column_id = 256,
                    max_question_length = 256,
                )
                new_inputs.append(tokenized_input)
            
            # convert again into dict of list
            new_features = {key: [input[key] for input in new_inputs] for key in new_inputs[0]}
            masked_inputs = self.data_collator(new_features["input_ids"]) # return value in tensor format

            new_features["input_ids"] = masked_inputs["input_ids"]
            new_features["labels"] = masked_inputs["labels"]
            new_features["attention_mask"] = torch.tensor(new_features["attention_mask"])
            new_features["token_type_ids"] = torch.tensor(new_features["token_type_ids"])

            return new_features
        else:
            #for validation
            return default_data_collator(batch)

    def train(self):
        #load dataset
        dir = os.path.abspath("data/dataset-pretraining")
        dataset = load_from_disk(dir)

        batch_size = 32
        epoch = 50
        training_args = TrainingArguments(
            output_dir= "model/tapas_masklm_id",
            overwrite_output_dir = True,
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            logging_strategy='epoch',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            seed = 42,
            learning_rate= 5e-5,
            num_train_epochs= epoch,
            weight_decay= 0.01,
            fp16= True,
	    use_cpu=False,
            remove_unused_columns= False,
            push_to_hub=True,
            hub_model_id=f"rizki-syazali/tapas_masklm_id_{self.model_version}"
        )
        trainer = HFTrainer(
            model= self.model,
            args= training_args,
            train_dataset= dataset["train"],
            eval_dataset= dataset["test"],
            data_collator= self.whole_word_masking_data_collator
        )
        #torch.cuda.empty_cache()
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--masklm_model", required=True, type=str)
    parser.add_argument("--vocab_file", default="vocab_file/indobert_vocab.txt", type=Path)
    parser.add_argument("--resume", required=True, type=bool)
    parser.add_argument("--model_version", required=True, type=str)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


