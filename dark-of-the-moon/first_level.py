# 1st level models of the winning team (token level models)
#%%
import os
from pathlib import Path
import pickle
from statistics import mean

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import torch
from torch.utils.data import random_split, Subset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import Trainer, TrainingArguments

#%%
loc = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
if loc == "Interactive" or loc == "Localhost":
    conf = {
        "batch_size": 32,
        "epochs": 0.1,
        "eval_steps": 200,
        "learning_rate": 2e-5,
    }
# When it is run after an api push.
elif loc == "Batch":
    conf = {
        "batch_size": 128,
        "epochs": 3,
        "eval_steps": 100,
        "learning_rate": 1e-4,
    }
#%%
def read_tweets(path):
    data = pd.read_csv(path, skip_blank_lines=False)
    data.dropna(0, "any", inplace=True)

    contexts = data["text"].to_list()
    # contexts = [" ".join(context.split()) for context in contexts]
    ids = data["textID"].to_list()
    questions = data["sentiment"].to_list()
    answers = [
        {"text": answer}
        for answer in data["selected_text"].values
        # {"text": " ".join(answer.split())} for answer in data["selected_text"].values
    ]
    for answer, context in zip(answers, contexts):
        answer["answer_start"] = context.find(answer["text"])
        answer["answer_end"] = answer["answer_start"] + len(answer["text"])

    return contexts, questions, answers, ids


contexts, questions, answers, ids = read_tweets(
    "../input/tweet-sentiment-extraction/train.csv"
)  # 27500 examples
#%%
# length = 40
# contexts = contexts[:length]
# questions = questions[:length]
# answers = answers[:length]
# ids = ids[:length]
# %%
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# the length of the tokenized sentences (max: 112) does not seem to exceed the model’s default max length.
encodings = tokenizer(contexts, questions, padding=True)
# encodings = tokenizer(contexts, questions, padding=False)
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
    def __init__(self, encodings):  # , ids):
        self.encodings = encodings
        # self.ids = ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item.update({'textID': self.ids[idx]})
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = TweetDataset(encodings)
#%%
def cross_validate(model_name, training_args, dataset, n_splits=5):
    folds = KFold(n_splits)  # StratifiedKFold(n_splits=n_splits)
    n_samples = len(dataset)

    starts = []
    ends = []
    losses = []
    for i, (train_idx, test_idx) in enumerate(folds.split(np.zeros(n_samples))):
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

        # train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)

    start_logits = np.concatenate(starts)
    end_logits = np.concatenate(ends)
    return start_logits, end_logits, losses


#%%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=conf["epochs"],
    per_device_train_batch_size=conf["batch_size"],
    per_device_eval_batch_size=4 * conf["batch_size"],
    learning_rate=conf["learning_rate"],
    lr_scheduler_type="constant",
    # warmup_steps=500,
    weight_decay=0.001,
    label_smoothing_factor=0.1,
    save_strategy="no",
    evaluation_strategy="steps",
    report_to="tensorboard",
    logging_dir="./logs",
    logging_steps=conf["eval_steps"],
)

start_logits, end_logits, losses = cross_validate(model_name, training_args, dataset)
#%%
with open(model_name + "_pred_off_start", "wb") as f:
    pickle.dump(start_logits, f)
with open(model_name + "_pred_off_end", "wb") as f:
    pickle.dump(end_logits, f)
#%%
# preds_path = Path("../output/constant-lr")

# with open(preds_path / "distilbert-base-uncased_pred_off_start", "rb") as f:
#     start_logits = pickle.load(f)
# with open(preds_path / "distilbert-base-uncased_pred_off_end", "rb") as f:
#     end_logits = pickle.load(f)
logits = np.stack((start_logits, end_logits), axis=2)
# %%
def logits_to_string(logits, encoding, text):
    """Transforms logit predictions to a text selection
    Assumes that logits[i] corresponds to text[i].
    """
    # TODO option to truncate the end_logits to [start_index:] to force end_index to be > start_index

    start_idx, end_idx = logits.argmax(axis=0)
    if end_idx < start_idx:
        return text, 1
    else:
        # tokenizer.decode adds extra spaces…
        # return tokenizer.decode(encoding.ids[start_idx:end_idx+1])
        # offset[1] is the index of the character after the last on in the token
        return text[encoding.offsets[start_idx][0] : encoding.offsets[end_idx][1]], 0


def words_jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


#%%
def logits_to_string_bert(logits, encoding, text):
    """Same as above but following the procedure from the BERT paper"""

    start_logits = logits[:, 0]
    end_logits = logits[:, 1]

    best_start_idx = len(start_logits) - 1
    best_end_idx = len(start_logits) - 1
    best_span_score = start_logits[best_start_idx] + end_logits[best_end_idx]
    # evaluates all spans starting from the end of the sequence
    for start_idx in range(len(start_logits) - 1, 0, -1):
        # best_end_idx: idx greater than or equal to start_idx with the biggest score
        if end_logits[start_idx] > end_logits[best_end_idx]:
            best_end_idx = start_idx
        # score of the best span starting from start_idx
        span_score = start_logits[start_idx] + end_logits[best_end_idx]
        # score of the best span starting from or after start_idx
        if span_score >= best_span_score:
            best_span_score = span_score
            best_start_idx = start_idx

    return text[encoding.offsets[best_start_idx][0] : encoding.offsets[best_end_idx][1]]


#%%
errors = 0
predictions = []
for i in range(len(logits)):
    pred, whole = logits_to_string(logits[i], encodings[i], contexts[i])
    predictions.append(pred)
    errors += whole
print(errors)
# 99,96% of the predictions have end_idx >= start_idx
#%%
predictions_b = [
    logits_to_string_bert(logits[i], encodings[i], contexts[i])
    for i in range(len(logits))
]
#%%
jaccards = [
    words_jaccard(prediction, answer["text"])
    for prediction, answer in zip(predictions, answers)
]
jaccards_b = [
    words_jaccard(prediction, answer["text"])
    for prediction, answer in zip(predictions_b, answers)
]
print(mean(jaccards), mean(jaccards_b))
#%%
# baselines = [
#     words_jaccard(context, answer["text"])
#     for context, answer in zip(contexts, answers)
# ]
# print(mean(baselines)) 0.589
#%%
# for idx in range(70):
#     if jaccards[idx] < .8:
#         print(idx, jaccards[idx], predictions[idx], "||",  answers[idx]["text"], logits[idx].argmax(0))
#         print(logits[idx])

#%%
# jaccards = np.array(jaccards)
# predictions = np.array(predictions)
# contexts = np.array(contexts)
# # %%
# import matplotlib
# from matplotlib import pyplot as plt
# # %%
# print("There are", len(jaccards), "examples")
# plt.hist(jaccards)
# #%%
# where0 = np.where(jaccards==0)[0]
