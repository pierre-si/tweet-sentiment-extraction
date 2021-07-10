# 1st level models of the winning team (character level models)
#%%
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import torch
from torch.utils.data import random_split, Subset
from transformers import DistilBertTokenizerFast, AutoModelForQuestionAnswering
from transformers import Trainer, TrainingArguments

#%%
def read_tweets(path):
    data = pd.read_csv(path, skip_blank_lines=False)
    data.dropna(0, "any", inplace=True)

    contexts = data["text"].to_list()
    ids = data["textID"].to_list()
    sentiments = data["sentiment"].to_list()
    questions = ["Sentiment"] * len(contexts)
    answers = [
        {"text": context, "answer_start": context.find(answer)}
        for context, answer in zip(contexts, data["selected_text"].values)
    ]
    for answer in answers:
        answer["answer_end"] = answer["answer_start"] + len(answer["text"])

    return contexts, questions, answers, ids, sentiments


contexts, questions, answers, ids, sentiments = read_tweets(
    "../input/tweet-sentiment-extraction/train.csv"
)  # 27500 examples
# %%
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(contexts, questions, truncation=True, padding=True)
# %%
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


add_token_positions(encodings, answers)
# %%
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):  # , ids, sentiments):
        self.encodings = encodings
        # self.ids = ids
        # self.sentiments = sentiments

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item.update({'textID': self.ids[idx]})
        # item.update({'sentiment': self.sentiments[idx]})
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = TweetDataset(encodings)
#%%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="no",
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=200,
)
#%%
def cross_validate(model_name, training_args, dataset, n_splits=5):
    folds = KFold(n_splits)  # StratifiedKFold(n_splits=n_splits)
    n_samples = len(dataset)

    starts = []
    ends = []
    losses = []
    for i, (train_idx, test_idx) in enumerate(
        folds.split(np.zeros(n_samples))  # , dataset.sentiments)
    ):
        print("\nFold", i)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, test_idx)

        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        (start, end), _, metrics = trainer.predict(val_dataset)
        starts.append(start)
        ends.append(end)
        losses.append(metrics["test_loss"])

    start_logits = np.concatenate(starts)
    end_logits = np.concatenate(ends)
    return start_logits, end_logits, losses


#%%
start_logits, end_logits, losses = cross_validate(
    "distilbert-base-uncased", training_args, dataset
)
#%%
with open("distilbert-base-uncased_pred_off_start", "wb") as f:
    pickle.dump(start_logits, f)
with open("distilbert-base-uncased_pred_off_end", "wb") as f:
    pickle.dump(end_logits, f)
