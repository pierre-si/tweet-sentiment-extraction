#%%
import pickle

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
train = pd.read_csv('../data/train.csv', dtype={'sentiment':'category'})
test = pd.read_csv('../data/test.csv')
# 0: bug, 1: feature, 2: question
# %%
df = train.drop(['selected_text', 'textID'], axis=1)
df = df.dropna().reset_index(drop=True)
# %%
# Removing \r, \n, contiguous whitespaces, 's, "
df['Content_Parsed'] = df['text'].str.replace("\r", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("\n", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace(" +", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("'s", "")
df['Content_Parsed'] = df['Content_Parsed'].str.replace('"', '')
# To downcase
df['Content_Parsed'] = df['Content_Parsed'].str.lower()
# Removing punctuation
punctuation_signs = list("?:!.,;")
for punct_sign in punctuation_signs:
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(punct_sign, '')
# Lemmatization
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    lemmatized_list = []
    text = df.loc[row]['Content_Parsed']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos='v'))

    lemmatized_text = " ".join(lemmatized_list)
    lemmatized_text_list.append(lemmatized_text)
df['Content_Parsed'] = lemmatized_text_list
# STOP Words
nltk.download('stopwords')
# missing a few stopwords such as they've
stop_words = list(stopwords.words('english'))
stop_words.sort(key=len, reverse=True)
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(regex_stopword, '')
# supprimer éventuellement les "'", "-", 
df['Content_Parsed'] = df['Content_Parsed'].str.replace(" +", " ")
# %%
df.to_csv("../data/datasets/contents.csv", index=False)
# %% 
# Train Test split
#df = pd.read_csv('data/datasets/contents.csv', dtype={'Category':'category'})
random_state = 10
X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'], df['sentiment'].cat.codes, test_size=0.1, random_state=random_state, stratify=df['sentiment'].cat.codes)
with open('../data/datasets/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
with open('../data/datasets/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
with open('../data/datasets/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
with open('../data/datasets/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

# %%
# TFIDF
ngram_range = (1, 2) # unigram et bigram
min_df = 10 # ignores terms with df lower than 10 (int)
max_df = 1. # ignores terms with df larger than 100% (float)
max_features = 300

tfidf = TfidfVectorizer(
    encoding="utf-8",
    ngram_range=ngram_range,
    stop_words=None,
    lowercase=False,
    max_df=max_df,
    min_df=min_df,
    max_features=max_features,
    norm='l2',
    sublinear_tf=True) # → replaces tf with 1+log(tf)

# with open("data/datasets/X_train.pickle", 'rb') as f:
#     X_train = pickle.load(f)
# with open("data/datasets/X_test.pickle", 'rb') as f:
#     X_test = pickle.load(f)
# with open("data/datasets/y_train.pickle", 'rb') as f:
#     y_train = pickle.load(f)
# with open("data/datasets/y_test.pickle", 'rb') as f:
#     y_test = pickle.load(f)

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
with open('../data/datasets/tfidf/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)
with open('../data/datasets/tfidf/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)
with open('../data/datasets/tfidf/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)
with open('../data/datasets/tfidf/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)


# %%