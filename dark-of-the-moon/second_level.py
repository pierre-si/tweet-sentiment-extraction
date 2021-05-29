# 2nd level models of the winning team (character level models)
#%%
import pickle
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# %%
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
    ('bert-base-uncased-', 'theo'),
    ('bert-wwm-neutral-', 'theo'),
    ("roberta-", 'hk'),
    ("distil_", 'hk'),
    ("large_", 'hk'),
]

add_spaces_to = ["bert_", 'xlnet_', 'electra_', 'bertweet-']
#%%

# def create_input_data(models):
#     char_pred_test_starts = []
#     char_pred_test_ends = []

#     for model, _ in models:
#         with open(model + 'char_pred_test_start.pkl', "rb") as fp:   #Pickling
#             probas = pickle.load(fp)  

#             if model in add_spaces_to:
#                 probas = [np.concatenate([np.array([0]), p]) for p in probas]

#             char_pred_test_starts.append(probas)

#         with open(model + 'char_pred_test_end.pkl', "rb") as fp:   #Pickling
#             probas = pickle.load(fp)

#             if model in add_spaces_to:
#                 probas = [np.concatenate([np.array([0]), p]) for p in probas]

#             char_pred_test_ends.append(probas)
            
#     char_pred_test_start = [np.concatenate([char_pred_test_starts[m][i][:, np.newaxis] for m in range(len(models))], 
#                                            1) for i in range(len(char_pred_test_starts[0]))]

#     char_pred_test_end = [np.concatenate([char_pred_test_ends[m][i][:, np.newaxis] for m in range(len(models))], 
#                                          1) for i in range(len(char_pred_test_starts[0]))]
    
#     return char_pred_test_start, char_pred_test_end
#%%
# char_pred_test_start, char_pred_test_end = create_input_data(MODELS)
#%%
def reorder(order_source, order_target, preds):
#     assert len(order_source) == len(order_target) and len(order_target) == len(preds)
    order_source = list(order_source)
    new_preds = []
    for tgt_idx in order_target:
        new_idx = order_source.index(tgt_idx)
        new_preds.append(preds[new_idx])
        
    return new_preds


df_train = pd.read_csv(DATA_PATH + 'train.csv').dropna().reset_index(drop=True)
df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
order_t = list(df_train['textID'].values)

df_train = pd.read_csv(DATA_PATH + 'train.csv').dropna()
df_train = df_train.sample(frac=1, random_state=50898).reset_index(drop=True)
order_hk = list(df_train['textID'].values)

ORDERS = {
    'theo': order_t,
    'hk': order_hk,
}

char_pred_oof_starts = []
char_pred_oof_ends = []

for model, author in tqdm(MODELS):
    with open(PKL_PATH + model + 'char_pred_oof_start.pkl', "rb") as fp:   #Pickling
        probas = pickle.load(fp)
        
        if author != 'hk':
            probas = reorder(ORDERS[author], ORDERS['hk'], probas)
        
        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]
            
        char_pred_oof_starts.append(probas)

    with open(PKL_PATH + model + 'char_pred_oof_end.pkl', "rb") as fp:   #Pickling
        probas = pickle.load(fp)
        
        if model in add_spaces_to:
            probas = [np.concatenate([np.array([0]), p]) for p in probas]
        
        if author != 'hk':
            probas = reorder(ORDERS[author], ORDERS['hk'], probas)
            
        char_pred_oof_ends.append(probas)



# %%
n_models = len(MODELS)

char_pred_oof_start = [np.concatenate([char_pred_oof_starts[m][i][:, np.newaxis] for m in range(n_models)], 
                                      1) for i in range(len(df_train))]

char_pred_oof_end = [np.concatenate([char_pred_oof_ends[m][i][:, np.newaxis] for m in range(n_models)], 
                                      1) for i in range(len(df_train))]

# %%
preds = {
#    'test_start': np.array(char_pred_test_start),
#    'test_end': np.array(char_pred_test_end),
    'oof_start': np.array(char_pred_oof_start),
    'oof_end': np.array(char_pred_oof_end),
}

model_names = [a + ' : ' + m for m, a in MODELS]
combs = [model_names]

print('Using models : ', combs)
# %% Character level tokenizer from https://huggingface.co/google/reformer-enwik8
# Encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(str.encode(string)) for string in list_of_strings])
    print("Max length:", max_length)
    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        #print(len(string), len(torch.tensor([x+2 for x in string])))
        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks

# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs
# %%
ids, att_masks = encode(df_train['text'].values)
#%%
errors = 0
i = 0
for text, id, att_mask in zip(df_train['text'], ids, att_masks):
    try:
        assert len(text) == att_mask.numpy().sum()
    except:
        errors+=1
        #print(text, id)
    i+=1
# for 156 examples, len(text) is different than len(encoded_characters), because of characters such as Cafï¿½ .
# %%
def generate_targets(texts, selected_texts, seq_length):
    targs = np.zeros((len(texts), seq_length))
    for i, (text, selected_text) in enumerate(zip(texts, selected_texts)):
        assert text.find(selected_text) >= 0
        # not correct for the few examples where len(text) != len(characters)
        targs[i, text.find(selected_text):len(selected_text)] = 1
    return targs
#%%
'''Second-level models
input:
1. unprocessed (not cleaned) sentences embedded at character level
2. sentiment
3. probabilities (transformers' outputs using cleaned text as input)
training process:
Adam optimizer
Linear learning rate, no warmup
smoothed cross entropy loss
multi sample dropout
5 epochs or 5 + 5 with Stochastic Weighted Average (average of the model weights during the last 5 epochs)
Pseudo labeling?
TODO: CNN and eventually Wavenet.
'''
#%% dataset
class TweetSentimentDataset(Dataset):
    def __init__(self, sentiments, start_probabilities, end_probabilities, characters, targets):
        self.sentiments = sentiments
        self.start_probabilities = start_probabilities
        self.end_probabilities = end_probabilities
        self.characters = characters
        self.targets = targets
    def __len__(self):
        return len(self.sentiments)
    def __getitem__(self, idx):
        return {
            'start_probabilities': self.start_probabilities[idx],
            'end_probabilities': self.end_probabilities[idx],
            'characters': self.characters[idx],
            'targets': self.targets[idx]
        }
        # return torch.cat((self.sentiments[idx], self.start_probabilities[idx], self.end_probabilities[idx], self.characters[idx])), self.targets[idx]

#%%
y = generate_targets(df_train['text'].values, df_train['selected_text'].values, 159)
#%%
train_dataset = TweetSentimentDataset(df_train['sentiment'].astype('category').cat.codes.values, preds['oof_start'], preds['oof_end'], ids, y)
#%% model

class TweetSentimentCNN(nn.Module):
    def __init__(self, n_models, max_token, dim=16):
        self.conv_prob = nn.Conv1d(in_channels=n_models, out_channel=dim, kernel_size=3)
        self.batchnorm = nn.BatchNorm1d(num_features=dim)

        self.char_emb = nn.Embedding(num_embeddings=max_token, embedding_dim=dim, padding_idx=0)
        self.sentiment_emb = nn.Embedding(num_embeddings=3, embedding_dim=dim, padding_idx=3) # there is no padding. 0 is used for a sentiment.

        convs = []
        for i in range(4):
            convs.append(nn.Conv1d(in_channels=dim, out_channel=dim, kernel_size=3))
            convs.append(nn.BatchNorm1d(num_features=dim))
        self.convs = nn.Sequential(*convs)

    def forward(self, probabilities, characters, sentiment):
        pass