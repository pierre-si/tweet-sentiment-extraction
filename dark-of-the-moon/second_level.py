# 2nd level models of the winning team (character level models)
#%%
from pathlib import Path
from copy import deepcopy
import pickle
import random
from statistics import mean

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

device = "cuda" if torch.cuda.is_available() else "cpu"
# %% code by the winning team
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2020
seed_everything(SEED)
#%%
DATA_PATH = "../input/tweet-sentiment-extraction/"
PKL_PATH = "../input/tweet-char-lvl-preds/"

MODELS = [
    ("bert-base-uncased-", "theo"),
    ("bert-wwm-neutral-", "theo"),
    ("distilbert-base-uncased-distilled-squad-", "theo"),
    ("albert-large-v2-squad-", "theo"),
    ("roberta-", "hk"),
    ("distil_", "hk"),
    ("large_", "hk"),
]

add_spaces_to = ["bert_", "xlnet_", "electra_", "bertweet-"]
#%%
def reorder(order_source, order_target, preds):
    #     assert len(order_source) == len(order_target) and len(order_target) == len(preds)
    order_source = list(order_source)
    new_preds = []
    for tgt_idx in order_target:
        new_idx = order_source.index(tgt_idx)
        new_preds.append(preds[new_idx])

    return new_preds


# encoding tried: ISO-8859-1, latin1, utf-8, ISO-8859-3, utf-8-sig
df_train = pd.read_csv(DATA_PATH + "train.csv").dropna().reset_index(drop=True)
df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
order_t = list(df_train["textID"].values)

df_train = pd.read_csv(DATA_PATH + "train.csv").dropna()
df_train = df_train.sample(frac=1, random_state=50898).reset_index(drop=True)
order_hk = list(df_train["textID"].values)

ORDERS = {
    "theo": order_t,
    "hk": order_hk,
}

char_pred_oof_starts = []
char_pred_oof_ends = []

for model, author in tqdm(MODELS):
    with open(PKL_PATH + model + "char_pred_oof_start.pkl", "rb") as fp:  # Pickling
        probas = pickle.load(fp)

        if author != "hk":
            probas = reorder(ORDERS[author], ORDERS["hk"], probas)

        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]

        char_pred_oof_starts.append(probas)

    with open(PKL_PATH + model + "char_pred_oof_end.pkl", "rb") as fp:  # Pickling
        probas = pickle.load(fp)

        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]

        if author != "hk":
            probas = reorder(ORDERS[author], ORDERS["hk"], probas)

        char_pred_oof_ends.append(probas)


# %%
n_models = len(MODELS)

char_pred_oof_start = [
    np.concatenate(
        [char_pred_oof_starts[m][i][:, np.newaxis] for m in range(n_models)], 1
    )
    for i in range(len(df_train))
]

char_pred_oof_end = [
    np.concatenate(
        [char_pred_oof_ends[m][i][:, np.newaxis] for m in range(n_models)], 1
    )
    for i in range(len(df_train))
]

preds = {
    "oof_start": np.array(char_pred_oof_start),
    "oof_end": np.array(char_pred_oof_end),
}

model_names = [a + " : " + m for m, a in MODELS]
combs = [model_names]

# %% reimplementation starts here
# Character level tokenizer from https://huggingface.co/google/reformer-enwik8
# Encoding
def encode(list_of_strings, pad_token_id=0, max_length=None):
    if max_length is None:
        max_length = max([len(string) for string in list_of_strings])
    print("Max length:", max_length)
    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full(
        (len(list_of_strings), max_length), pad_token_id, dtype=torch.long
    )

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            bstring = str.encode(string)
            # characters that take more than 1 byte (such as ï¿½) are truncated
            # they appear in 156 examples
            if len(bstring) != len(string):
                bstring = [str.encode(c)[0] for c in string]
            string = bstring
        input_ids[idx, : len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, : len(string)] = 1

    return input_ids, attention_masks


# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append(
            "".join([chr(x - 2) if x > 1 else "" for x in output_ids])
        )
    return decoded_outputs


# %%
def generate_targets(texts, selected_texts, seq_length):
    y = np.zeros((len(texts), seq_length, 2))
    y_combined = np.zeros((len(texts), seq_length))
    y_start = []
    y_end = []
    for i, (text, selected_text) in enumerate(zip(texts, selected_texts)):
        assert text.find(selected_text) >= 0
        # not correct for the few examples where len(text) != len(characters)
        start = text.find(selected_text)
        # end is the index of the last character
        end = start + len(selected_text) - 1
        y[i, start, 0] = 1
        y[i, end, 1] = 1
        y_start.append(start)
        y_end.append(end)
        y_combined[i, start:end] = 1
    return y, y_combined, y_start, y_end


#%% dataset
class TweetSentimentDataset(Dataset):
    def __init__(
        self,
        sentiments,
        start_probabilities,
        end_probabilities,
        characters,
        targets,
        texts,
        selected_texts,
    ):
        self.sentiments = torch.tensor(sentiments, dtype=torch.int)
        self.start_probabilities = pad_sequence(
            [torch.tensor(p, dtype=torch.float32) for p in start_probabilities],
            batch_first=True,
        )
        self.end_probabilities = pad_sequence(
            [torch.tensor(p, dtype=torch.float32) for p in end_probabilities],
            batch_first=True,
        )
        self.characters = characters
        self.targets = targets
        self.texts = texts
        self.selected_texts = selected_texts

    def __len__(self):
        return len(self.sentiments)

    def __getitem__(self, idx):
        return {
            "start_probabilities": self.start_probabilities[idx],
            "end_probabilities": self.end_probabilities[idx],
            "sentiment": self.sentiments[idx],
            "tokens": self.characters[idx],
            "targets": self.targets[idx],
            "text": self.texts[idx],
            "selected_text": self.selected_texts[idx],
        }


# # for dynamic padding. (do not forget uniform size batching ie sorting by seq_len and to not pad characters and targets before giving it to the dataset)
# def tweet_collate_fn(batch):
#     start_prob = [torch.tensor(p) for p in batch['start_probabilities'])]
#     start_prob = pad_sequence(start_prob, batch_first=True)
#     end_prob = [torch.tensor(p) for p in batch['end_probabilities']]
#     end_prob = pad_sequence(end_prob, batch_first=True)

#%% model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TweetSentimentCNN(nn.Module):
    def __init__(
        self, n_models, n_tokens, emb_dim=16, cnn_dim=16, dropout=0.3, dropout_samples=8
    ):
        super().__init__()

        self.prob_conv = ConvBlock(2 * n_models, emb_dim, kernel_size=3)
        self.char_emb = nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=emb_dim, padding_idx=0
        )
        self.sentiment_emb = nn.Embedding(
            num_embeddings=3, embedding_dim=emb_dim, padding_idx=-1
        )  # there is no padding. 0 is used for a sentiment.

        convs = []
        convs.append(ConvBlock(3 * emb_dim, cnn_dim, kernel_size=3))
        for i in range(1, 4):
            convs.append(
                ConvBlock(cnn_dim * (2 ** (i - 1)), cnn_dim * (2 ** i), kernel_size=3)
            )
        self.convs = nn.Sequential(*convs)

        # embedding wise linear.
        self.lin = nn.Sequential(
            nn.Linear(cnn_dim * (2 ** i), cnn_dim), nn.ReLU(), nn.Linear(cnn_dim, 2)
        )

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_samples = dropout_samples

    def forward(self, start_probabilities, end_probabilities, tokens, sentiment):
        L = start_probabilities.size()[1]
        prob = torch.cat((start_probabilities, end_probabilities), -1).transpose(1, 2)
        prob = self.prob_conv(prob)
        tokens = self.char_emb(tokens)
        sent = self.sentiment_emb(sentiment).unsqueeze(2).repeat((1, 1, L))
        x = torch.cat((prob, tokens.transpose(1, 2), sent), dim=1)
        x = self.convs(x)  # N×C×L
        if self.training:
            x = torch.stack(
                [self.dropout(x) for _ in range(self.dropout_samples)], dim=1
            )  # N×Ds×C×L
            x = torch.mean(self.lin(x.transpose(2, 3)), dim=1)  # N×L×2
        else:
            x = self.dropout(x)
            x = self.lin(x.transpose(1, 2))  # N×L×2
        return x


def logits_to_string(logits, text):
    """Transforms logit predictions to a text selection
    Assumes that logits[i] corresponds to text[i].
    """
    # TODO option to truncate the end_logits to [start_index:] to force end_index to be > start_index

    # start_idx, end_idx = logits[: len(text)].argmax(dim=0).cpu().numpy()
    start_idx, end_idx = logits.argmax(dim=0).cpu().numpy()
    if end_idx <= start_idx:
        return text
    else:
        return text[start_idx : end_idx + 1]
        # while text[start_idx] != " " and start_idx > 0:
        # start_idx -= 1
        # while end_idx < len(text) and text[end_idx] != " ":
        # end_idx += 1
        # return text[start_idx:end_idx]


#%% metrics
# from Huggingface's LabelSmoother
class LabelSmoothingCrossEntropy:
    def __init__(self, eps=0.1, ignore_index=-100):
        self.epsilon = eps
        self.ignore_index = ignore_index

    # logits has dim B×Class(i.e. seq length)×2
    # labels has dim B×2
    def __call__(self, logits, labels):
        log_probs = -torch.nn.functional.log_softmax(logits, dim=1)
        # labels has now dim B×1×2
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(1)
        # nll_loss is logsoftmax(x_class)
        nll_loss = log_probs.gather(dim=1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=1, keepdim=True, dtype=torch.float32)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = labels.numel()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def words_jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def evaluate(model, criterion, dataloader):
    model.eval()
    losses = []
    jaccards = []
    with torch.no_grad():
        for batch in dataloader:
            ypreds = model(
                batch["start_probabilities"].to(device),
                batch["end_probabilities"].to(device),
                batch["tokens"].to(device),
                batch["sentiment"].to(device),
            )
            pred_selections = [
                logits_to_string(logits, text)
                for logits, text in zip(ypreds, batch["text"])
            ]

            y = batch["targets"].to(device)
            true_selections = batch["selected_text"]

            losses.append(float(criterion(ypreds, y)))
            jaccards.append(
                mean(
                    [
                        words_jaccard(pred_selection, true_selection)
                        for pred_selection, true_selection in zip(
                            pred_selections, true_selections
                        )
                    ]
                )
            )

    return mean(losses), mean(jaccards)


# %%
def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader=None,
    epochs=10,
    eval_steps=100,
    log_dir="tensorboard",
):

    if log_dir is not None:
        writer = SummaryWriter(log_dir)

    step = 0
    for e in range(epochs):
        print("Epoch ", e)
        for batch in train_loader:
            model.train()
            ypreds = model(
                batch["start_probabilities"].to(device),
                batch["end_probabilities"].to(device),
                batch["tokens"].to(device),
                batch["sentiment"].to(device),
            )
            loss = criterion(ypreds, batch["targets"].to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % eval_steps == 0:
                train_loss, train_jaccard = evaluate(model, criterion, train_loader)
                print("Step", step, end="\n")
                print("Train loss:", "{:.3f}".format(train_loss), end=" ")
                print("Train jaccard:", "{:.3f}".format(train_jaccard), end="\n")
                if log_dir is not None:
                    writer.add_scalar("train/loss", train_loss, step)
                    writer.add_scalar("train/jaccard", train_jaccard, step)
                if val_loader is not None:
                    val_loss, val_jaccard = evaluate(model, criterion, val_loader)
                    print("Valid loss:", "{:.3f}".format(val_loss), end=" ")
                    print("Valid jaccard:", "{:.3f}".format(val_jaccard), end="\n")
                    if log_dir is not None:
                        writer.add_scalar("eval/loss", val_loss, step)
                        writer.add_scalar("eval/jaccard", val_jaccard, step)

    if log_dir is not None:
        writer.close()


#%%
def cross_validate(model, dataset, epochs=10, n_splits=5):
    folds = StratifiedKFold(n_splits=n_splits)
    n_samples = len(dataset)

    models = []
    val_losses = []
    val_jaccard_scores = []
    # stratify based on sentiment.
    # criterion = CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(eps=0.1)
    for i, (train_idx, test_idx) in enumerate(
        folds.split(np.zeros(n_samples), dataset.sentiments)
    ):
        print("Fold", i)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size=128)
        val_loader = DataLoader(val_dataset, batch_size=512)

        m = model(n_models, ids.max().item() + 1, cnn_dim=32, dropout=0.3).to(device)
        lr = 4e-3
        optimizer = optim.AdamW(m.parameters(), lr=lr)
        opt = SWA(optimizer, swa_start=5, swa_freq=50, swa_lr=None)
        train(
            m,
            opt,
            criterion,
            train_loader,
            val_loader,
            epochs=epochs,
            eval_steps=50,
            log_dir=Path("tensorboard") / ("fold" + str(i)),
        )
        opt.swap_swa_sgd()
        val_loss, val_jaccard = evaluate(m, criterion, val_loader)
        val_losses.append(val_loss)
        val_jaccard_scores.append(val_jaccard)
        models.append(deepcopy(m))

    print("Average loss:", mean(val_losses))
    print("Average jaccard:", mean(val_jaccard_scores))
    return models


#%%
max_len = max([len(p) for p in preds["oof_start"]])
ids, att_masks = encode(df_train["text"].values, max_length=max_len)
#%%
errors = 0
for i, (text, id, att_mask) in enumerate(zip(df_train["text"], ids, att_masks)):
    try:
        assert len(text) == att_mask.numpy().sum()
    except:
        errors += 1
        print(i, text)
#%%
y, y_combined, y_start, y_end = generate_targets(
    df_train["text"].values, df_train["selected_text"].values, max_len
)
#%%
dataset = TweetSentimentDataset(
    df_train["sentiment"].astype("category").cat.codes.values,
    preds["oof_start"],
    preds["oof_end"],
    ids,
    torch.tensor((y_start, y_end)).T,
    df_train["text"],
    df_train["selected_text"],
)
#%%
model = TweetSentimentCNN(n_models, ids.max().item() + 1).to(device)
lr = 4e-3
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()
train_loader = DataLoader(dataset, batch_size=16)
#%%
train(model, optimizer, criterion, train_loader, log_dir=None)
evaluate(model, criterion, train_loader)
#%%
models = cross_validate(TweetSentimentCNN, dataset, n_splits=10)
