import torch
import pandas as pd

class TableQADataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # TapasTokenizer expects the table data to be text only
        table = item.table
        if item.position != 0:
          # use the previous table-question pair to correctly set the prev_labels token type ids
          previous_item = self.df.iloc[idx-1]
          encoding = self.tokenizer(table=table,
                                    queries=[previous_item.question, item.question],
                                    answer_coordinates=[previous_item.answer_coordinates, item.answer_coordinates],
                                    answer_text=[previous_item.answer_text, item.answer_text],
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
          )
          # use encodings of second table-question pair in the batch
          encoding = {key: val[-1] for key, val in encoding.items()}
        else:
          # this means it's the first table-question pair in a sequence
          encoding = self.tokenizer(table=table,
                                    queries=item.question,
                                    answer_coordinates=item.answer_coordinates,
                                    answer_text=item.answer_text,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
          )
          # remove the batch dimension which the tokenizer adds
          encoding = {key: val.squeeze(0) for key, val in encoding.items()}


        return {'encoding':encoding, 'index':idx}

    def __len__(self):
        return len(self.df)
