#%% imports
import os
import sys

# simulate Azure Functions Runtime in IPython
if "IPython" in sys.modules and 'Score' not in os.getcwd():
    os.environ['inputcsv'] = os.path.join('debug', 'input.csv')    
    os.environ['scoredcsv'] = os.path.join('debug', 'input.scores.csv')
    os.environ['unscorablecsv'] = os.path.join('debug', 'input.unscorable.csv')
    os.chdir('Score')

sys.path.append(os.path.join('..', 'lib'))

import pandas as pd
from sklearn.externals import joblib

from corpus import CsvCorpusReader, Levels, generate_processor, process_text
from gensim import corpora, models
from utility import nested_to_list

#%% create corpus 
input_path = os.environ['inputcsv']
input_dir = os.path.dirname(input_path)
input_name = os.path.basename(input_path)

corpus = CsvCorpusReader(input_dir, [input_name],
                         encoding="utf8",
                         default_text_selector=lambda row: row["Text"])

#%% define processor
stop_words = ['would', 'like', 'mcdonald']
text_processor = generate_processor(keep_alpha_only=True, to_lower=True, stopwords_langs=['english'], add_stopwords=stop_words, stemmer_langs=['english'])
docs = nested_to_list(corpus.words(keep_levels=Levels.Nothing, **text_processor))

#%% score words
dictionary = corpora.Dictionary.load(os.path.join('..', 'model', 'preprocessing', 'dictionary.bin'))
docs_bow = [dictionary.doc2bow(doc) for doc in docs]

model_tfidf = models.TfidfModel.load(os.path.join('..', 'model', 'preprocessing', 'tfidf.bin'))
docs_tfidf = model_tfidf[docs_bow]

model_lsi = models.LsiModel.load(os.path.join('..', 'model', 'preprocessing', 'lsi.bin'))
docs_lsi = model_lsi[docs_tfidf]

docs_features = [[c[1] for c in doc] for doc in docs_lsi]
features = pd.DataFrame(docs_features, columns=["F" + str(i) for i in range(len(model_lsi.show_topics()))])

#%% handle null
ids = corpus.rows_as_dataframe(columns=["Id"])
notnul_idx = features.notnull().all(axis=1)
features_notnull = features[notnul_idx]

unscorable_path = os.environ['unscorablecsv']
ids_null = ids[~notnul_idx]
if not(ids_null.empty):
    ids_null.to_csv(unscorable_path, index=False)
    print('INFO: {0} out of {1} rows are unscorable. Ids are saved to *.unscorable.csv'.format(len(ids_null), len(ids)))
else:
    print('INFO: All {0} rows are scorable'.format(len(ids)))

result = ids[notnul_idx].copy()

#%% score models
model_paths = [path for path in os.listdir(os.path.join('..', 'model')) if path.startswith('class_') ]

for model_path in model_paths:
    model = joblib.load(os.path.join('..', 'model', model_path))
    res = model.predict_proba(features_notnull)[:, 1]

    class_name = model_path.split('_')[1]
    threshold = float(model_path.rsplit('.', 1)[0].split('_')[-1])

    result.loc[:, "class_" + class_name] = res > threshold
    result.loc[:, "class_" + class_name + "_score"] = res

#%% save results
output_path = os.environ['scoredcsv']
result.to_csv(output_path)

print('INFO: {0} rows scores. Values are saved to *.scores.csv'.format(len(result)))