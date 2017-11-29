#%% imports
import os
import sys
from __future__ import print_function
sys.path.append('lib')

from collections import Counter
from itertools import count
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from gensim import corpora, models

from corpus import CsvCorpusReader, Levels, generate_processor, process_text
from plot import plot_classes_scatter, plot_roc
from utility import nested_partial_print, nested_to_list

##################################################################
# Preparing Corpus
##################################################################
#%% create corpus
corpus = CsvCorpusReader(".\data", ["data.csv"],
                         encoding="utf8",
                         default_text_selector=lambda row: row["Text"])

#%% create data frame
df = corpus.rows_as_dataframe(columns=["Id", "Text", "Class"])
df.head()

#%% define processor
stop_words = ['would', 'like', 'mcdonald']
text_processor = generate_processor(keep_alpha_only=True, to_lower=True, stopwords_langs=['english'], add_stopwords=stop_words, stemmer_langs=['english'])
docs_factory = lambda: corpus.words(keep_levels=Levels.Nothing, **text_processor)
nested_partial_print(docs_factory())

#%% build word counter dictionary
word_frequencies = Counter((word for doc in docs_factory() for word in doc))
pprint(word_frequencies.most_common(20))

#%% drop infrequent words
min_word_freq = 3
docs = [
    [
        word
        for word in doc if word_frequencies[word] >= min_word_freq
     ] for doc in docs_factory()
]
nested_partial_print(docs)

#%% create folder for models and transformations if not exist
model_path = "model"
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print("Created [{}]".format(model_path))

preprocessing_path = os.path.join(model_path, 'preprocessing')
if not os.path.exists(preprocessing_path):
    os.makedirs(preprocessing_path)
    print("Created [{}]".format(preprocessing_path))

#%% convert to Bag of Words representation
dictionary_path = os.path.join(preprocessing_path, 'dictionary.bin')

if os.path.exists(dictionary_path):
    dictionary = corpora.Dictionary.load(dictionary_path)
    print("Using existing transform from [{}]".format(dictionary_path))
else:
    dictionary = corpora.Dictionary(docs)
    dictionary.save(dictionary_path)
    print("Created [{}]".format(dictionary_path))

docs_bow = [dictionary.doc2bow(doc) for doc in docs]
nested_partial_print(docs_bow)

#%% convert to tf-idf representation
tfidf_path = os.path.join(preprocessing_path, 'tfidf.bin')

if os.path.exists(tfidf_path):
    model_tfidf = models.TfidfModel.load(tfidf_path)
    print("Using existing transform from [{}]".format(tfidf_path))
else:
    model_tfidf = models.TfidfModel(docs_bow)
    model_tfidf.save(tfidf_path)
    print("Created [{}]".format(tfidf_path))

docs_tfidf = nested_to_list(model_tfidf[docs_bow])
nested_partial_print(docs_tfidf)

#%% train and convert to LSI representation
lsi_path = os.path.join(preprocessing_path, 'lsi.bin')
lsi_num_topics = 500

if os.path.exists(lsi_path):
    model_lsi = models.LsiModel.load(lsi_path)
    print("Using existing transform from [{}]".format(lsi_path))
else:
    model_lsi = models.LsiModel(docs_tfidf, id2word=dictionary, num_topics=lsi_num_topics)
    model_lsi.save(lsi_path)
    print("Created [{}]".format(lsi_path))

docs_lsi = model_lsi[docs_tfidf]
nested_partial_print(docs_lsi)

#%% convert from sparse to dense feature space
feature_size = lsi_num_topics

docs_features = []
for doc in docs_lsi:
    doc_features = [0] * feature_size
    for p, v in doc:
        doc_features[p] = v

    docs_features.append(doc_features)
  
nested_partial_print(docs_features)

#%% enumerate class occurance
class_values = np.hstack(df[["Class"]].apply(lambda row: row["Class"].split("\n"), axis=1).values)
class_unique, class_counts = np.unique(class_values, return_counts=True)
dict(zip(class_unique, class_counts))

#%% create target
class_to_find = "SlowService"
df["Target"] = df.apply(lambda row: 1 if class_to_find in row["Class"] else 0, axis=1)
df.groupby(by=["Target"]).count()

#%% create features and targets dataset
features = pd.DataFrame(docs_features, columns=["F" + str(i) for i in range(len(model_lsi.show_topics()))])
notnul_idx = features.notnull().all(axis=1)
features = features[notnul_idx]
df_notnull = df[notnul_idx]
target = df_notnull[["Target"]]
plot_classes_scatter(features.values, target["Target"].values)

#%% split dataset to train and test
train_idx, test_idx = train_test_split(df_notnull.index.values, test_size=0.3, random_state=56)
df_train = df_notnull.loc[train_idx]
features_train = features.loc[train_idx]
target_train = target.loc[train_idx]
df_test = df_notnull.loc[test_idx]
features_test = features.loc[test_idx]
target_test = target.loc[test_idx]

#%% train logistic classifier
classifier = LogisticRegression()
classifier.fit(features_train, target_train)

#%% score on train
scores_train = classifier.predict_proba(features_train)[:, 1]
(tp_train, fp_train, tsh) = plot_roc(target_train, scores_train)

#%% score on test
scores_test = classifier.predict_proba(features_test)[:, 1]
(tp_test, fp_test, tsh) = plot_roc(target_test, scores_test)

#%% plot treshold values
pd.DataFrame(nested_to_list(zip(tsh, tp_test, fp_test, fp_test-tp_test)), columns=['Threshold', 'True Positive Rate', 'False Positive Rate', 'Difference']).plot(x='Threshold')
plt.xlim(0, 0.4)
plt.ylim([0,1])
plt.grid()
plt.show()

#%% check example (check threshold value two steps above)
threshold = 0.25
ex = "It took me 30 minutes to get my soda"
processor_lsi = lambda w: model_lsi[model_tfidf[dictionary.doc2bow(process_text(w, keep_levels=Levels.Nothing, **text_processor))]]
ex_score = classifier.predict_proba([c[1] for c in processor_lsi(ex)])[0,1]

"{0} with score {1}".format(ex_score > threshold, ex_score)

#%% check samples from test
for i in np.where(scores_test > threshold)[0]:
    print("[{0:0.3f} real: {1}] {2}".format(scores_test[i], df_test.iloc[i]["Target"], df_test.iloc[i]["Text"]))

#%% save model
model_filename = 'class_{0}_thresh_{1}.bin'.format(class_to_find, threshold)
joblib.dump(classifier, os.path.join(model_path, model_filename))
