# 1st level models of the winning team (character level models)
#%%
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
)
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
    output_dir="./results",  # output directory
    num_train_epochs=0.1,  # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=128,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,
)
#%%
def cross_validate(model_name, training_args, dataset, n_splits=5):
    folds = KFold(n_splits)  # StratifiedKFold(n_splits=n_splits)
    n_samples = len(dataset)

    predictions = []
    losses = []
    for i, (train_idx, test_idx) in enumerate(
        folds.split(np.zeros(n_samples))  # , dataset.sentiments)
    ):
        print("Fold", i)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, test_idx)

        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
        )

        trainer.train()

        preds, _, metrics = trainer.predict(val_dataset)
        predictions.append(preds)
        losses.append(metrics["test_loss"])

    return predictions, losses


#%%
predictions, losses = cross_validate("distilbert-base-uncased", training_args, dataset)

#%%
# train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*.8), int(len(dataset)*.2)])
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)

trainer.train()
