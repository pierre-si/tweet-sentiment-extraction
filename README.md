# Tweet Sentiment Extraction Kaggle Competition
Reimplementation of the winning model.

This repo hosts a reimplementation of a second level model of the [winning team](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477) (private score: 0.7362).
This model uses as inputs the 1st level models' outputs shared by the winning team.

The aim of the competition was to predict, given a text and a sentiment (positive, neutral, negative), the substring "selected_text" which reflects the sentiment.

The model is a character-level 1-dimensional CNN which uses multi sample dropout, stochastic weighted average and a smooth cross-entropy loss.

The score is a word level jaccard index.

Results (5 folds cross validation):
| CV average   |  Loss | Jaccard |
|:-------------|:------|--------:|
| CNN | 1.7912 | 0.7303 |

Graphs of the metrics during training of a 10 folds cross-validation:
![Train loss](train_loss.png "Training loss")
![Train jaccard](train_jaccard.png "Training jaccard")
![Eval loss](eval_loss.png "Eval loss")
![Eval jaccard](eval_jaccard.png "Eval jaccard")
