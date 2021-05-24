#%%
import pickle
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
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

def create_input_data(models):
    char_pred_test_starts = []
    char_pred_test_ends = []

    for model, _ in models:
        with open(model + 'char_pred_test_start.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)  

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_starts.append(probas)

        with open(model + 'char_pred_test_end.pkl', "rb") as fp:   #Pickling
            probas = pickle.load(fp)

            if model in add_spaces_to:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_ends.append(probas)
            
    char_pred_test_start = [np.concatenate([char_pred_test_starts[m][i][:, np.newaxis] for m in range(len(models))], 
                                           1) for i in range(len(char_pred_test_starts[0]))]

    char_pred_test_end = [np.concatenate([char_pred_test_ends[m][i][:, np.newaxis] for m in range(len(models))], 
                                         1) for i in range(len(char_pred_test_starts[0]))]
    
    return char_pred_test_start, char_pred_test_end
#%%
char_pred_test_start, char_pred_test_end = create_input_data(MODELS)
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
