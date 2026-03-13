import argparse
from pathlib import Path
from transformers import TapasForQuestionAnswering, TapasTokenizer
import pandas as pd
from utils.dataset_utils import TableQADataset
from torch.utils.data import DataLoader
from utils.data_utils import save_to_json
from transformers import get_scheduler
import torch
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, args):

        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.masklm_model = args.masklm_model
        self.vocab_file = args.vocab_file

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda")

        self.train_df = None
        self.eval_df = None
        self.test_df = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.test_dataloader = None

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.optimizer = None
        self.scheduler = None

        self.train_logs = []
        self.eval_logs = []

        self.set_tokenizer()
        self.set_model()
        self.set_data()
        self.set_optimizer()
        self.set_scheduler()
        self.train()

    def set_data(self):
        self.train_df = pd.read_pickle(f"{self.input_dir}/train.pkl")
        self.eval_df = pd.read_pickle(f"{self.input_dir}/test.pkl")

        # create Dataset and DataLoader
        train_dataset = TableQADataset(df=self.train_df, tokenizer=self.tokenizer)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)

        eval_dataset = TableQADataset(df=self.eval_df, tokenizer=self.tokenizer)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size)

    def set_model(self):
        self.model = TapasForQuestionAnswering.from_pretrained(self.masklm_model)

    def set_tokenizer(self):
        self.tokenizer = TapasTokenizer.from_pretrained(
            "google/tapas-base", vocab_file=self.vocab_file
        )

    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def set_scheduler(self):
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_training_steps = self.epoch * num_update_steps_per_epoch

        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

    def train(self):
        # empty cache
        torch.cuda.empty_cache()
        self.model.to(self.device)

        for epoch in range(self.epoch):
            with tqdm(self.train_dataloader, unit="batch", ncols=110) as tepoch:
                self.model.train()
                train_loss = []
                for step, batch in enumerate(tepoch):
                    # update tqdm
                    tepoch.set_description(f"Epoch {epoch}")

                    inputs = batch["encoding"]
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    train_loss.append(loss.item())
                    # update tqdm
                    tepoch.set_postfix(train_loss=f"{loss.item():.3f}", eval_loss=None, exact_match=None)

                    if step % 100 == 0:
                        self.train_logs.append(
                            {
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": mean(train_loss),
                            }
                        )

                    # each last step do evaluation
                    if step == len(tepoch) - 1:
                        self.eval(epoch, loss.item(), tepoch)

    def eval(self, epoch, train_loss, progress_bar):
        self.model.eval()
        val_loss = []
        total_correct = 0
        total_instances = 0
        for batch in self.eval_dataloader:
            index = batch["index"].tolist()
            inputs = batch["encoding"]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs.loss
            val_loss.append(loss.item())

            logits = outputs.logits
            predicted_answer_coordinates, = self.tokenizer.convert_logits_to_predictions(batch["encoding"], logits.detach().cpu())

            for i, data_index in enumerate(index):
                y_true = self.eval_df.iloc[data_index].answer_coordinates
                y_pred = predicted_answer_coordinates[i]

                if y_pred == y_true:
                    total_correct += 1

            total_instances += len(predicted_answer_coordinates)

        exact_match = float(total_correct / total_instances) * 100
        progress_bar.set_postfix(train_loss=f'{train_loss:.3f}', eval_loss=f'{mean(val_loss):.3f}',exact_match=f'{exact_match:.2f}%')
        self.eval_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": mean(val_loss),
                "exact_match": f"{exact_match:.2f}",
            }
        )

    def save_log(self):
        log = {"train": self.train_logs, "eval": self.eval_logs}
        save_to_json(f"{self.output_dir}/logs.json", log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masklm_model", required=True, type=Path)
    parser.add_argument("--vocab_file", required=True, type=Path)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoch", default=4, type=int)

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.save_log


