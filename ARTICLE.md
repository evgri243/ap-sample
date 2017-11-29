# Cost Effective Python ML Model Operationalization with Azure Functions

Recently, we had a project with [Action Press]() with a common, but rather popular email-classification task. Action Press LLC is a part of Action-Media company - the largest supplier of professional media in Russia. Since 2000, Action Press LLC is focused solely on the online subscription business with more than 100,000 companies as clients and offers more than 40 titles of professional magazines and e-books. To inform customers about news and stimulate subscriptions Action Press has developed an internal outbound marketing system that regularly send emails to the customer base. System has used full-text search with a basic automation which was not so effective as expected, as users were required to manually check and sort replies on the emails, forwarding them to corresponding departments. Main types of consumers responses were autoreplies, requests to remove from the mailing list, clarifying questions and requests for new subscriptions. The core idea of the engagement with us was in development of a machine learning model to make the process as much automatic as possible and then operationalize it to be used with other parts of the system in production. In this article I'd like to give a short review of ML model building process and then show a way Azure Functions can be used as a host for your python models in a simple and cost effective way.

## Machine Learning Model

Speaking honestly, there is nothing really big to share on ML side of the question. A set of simple binary Logistic Regressions classifiers showed very promising results, shifting core focus from the model itself to data preparation and text embedding. But the repository itself was already used as a basis for 3 other independent projects and proved to be useful in some qualifying classification experiments and as a jump start for development. Hence, the core idea of this section is not to show any "_know-how_", but to layout a required basis for a subsequent operationalization section, share code and experience and give some guidance for anyone interested in playing with or reusing the code.

> Due to privacy concerns the original partner's dataset was replaced by a similar public one for [McDonalds review]() classification. You can see it in [data/data.csv]() file.

The data itself was in format of CSVs with 3 columns: _Id_, _Text_ and _Class_. And as NLTK has no built-in reader support for csv data, we wrote our own, allowing to read files in a folder as a single pandas DataFrame or extract text in NLTK-native lists of paragraph, sentences, words and etc. Below is a code to initialize that custom `CsvCorpusReader` with customer's data. You can have a look on class implementation in [`lib\corpus.py`)]() file. I strongly recommend you to have a look at [_Experiments\TrainingExperiment.py_]() while reading this section. 

```
#%% create corpus
corpus = CsvCorpusReader(".\data", ["data.csv"],
                         encoding="utf8",
                         default_text_selector=lambda row: row["Text"])
```

The next step is to extract words from documents and normalize them. In our case, after some experimenting we've decided to wrap the whole process in a set of helper functions, which hides calls to NLTK and Gensim libraries under some simple to use configuration layer. Below we instructs the extractor to return documents as a list of alphanumeric-only words, dropping any paragraph or sentence structure (see, `keep_levels=Levels.Nothing`). Then we cast each word to a lower case, skip any stop words and stem the result. At the final step we remove infrequent words, assuming that they are just typos or have no real impact on classification. Please note, that the code below was adopted for English-only data sample, while the original version used Russian lemmatization using PyMorphy2, that allowed to achieve better classification accuracy for Russian language. 

```
#%% tokenize the text
stop_words = ['would', 'like', 'mcdonald']
text_processor = generate_processor(keep_alpha_only=True,
                                    to_lower=True,
                                    stopwords_langs=['english'],
                                    add_stopwords=stop_words,
                                    stemmer_langs=['english'])
docs_factory = lambda: corpus.words(keep_levels=Levels.Nothing, **text_processor)

word_frequencies = Counter((word for doc in docs_factory() for word in doc))
min_word_freq = 3
docs = [
    [
        word
        for word in doc if word_frequencies[word] >= min_word_freq
     ] for doc in docs_factory()
]
```

Once we have our corpus tokenized, the next step is to build text embedding. The idea of the code below is to convert every document to a row of meaningless numbers to be used in classifier. We tried a couple of different approaches (including BoW, TF-IDF, LSI, RP and w2v), but the classic [LSI]() model with 500 extracted topics showed the best results ([AUC]() = 0.98) in our case. At first, the code checks if a serialized model already exists in the shared directory. If not, it trains it with previously prepared data and saves the result to disk. Otherwise, it just loads existing one into memory. And then the code transforms the dataset  and repeats the flow with the next embedding.

There can be different possible reasons for LSI to outperform much more powerful _word2vec_ and other complex approaches. The most obvious one based on the idea that email types we were trying to look for have some predictable and repeatable word patterns, as in the case of outoreplyes (e.g., _"... thanks for your email ... I am out of office now ... in case of urgency ..."_). Therefore, they can be efficiently processed with something as simple as TF-IDF. LSI, while maintaining a common ideology, can be thought as a way to add synonyms tolerable to the processing. At the same time, word2vec, trained on Wikipedia, probably makes unnecessary noise by complex synonymous structures, thereby smoothing out the patterns expressed in the messages and therefore lowering classification accuracy. This approach showed that old and fairly simple methods are still worth trying even in the era of word2vec and recurrent neural networks.

```
#%% convert to Bag of Words representation
dictionary_path = os.path.join(preprocessing_path, 'dictionary.bin')
if os.path.exists(dictionary_path):
    dictionary = corpora.Dictionary.load(dictionary_path)
else:
    dictionary = corpora.Dictionary(docs)
    dictionary.save(dictionary_path)

docs_bow = [dictionary.doc2bow(doc) for doc in docs]
nested_partial_print(docs_bow)

#%% convert to tf-idf representation
tfidf_path = os.path.join(preprocessing_path, 'tfidf.bin')
if os.path.exists(tfidf_path):
    model_tfidf = models.TfidfModel.load(tfidf_path)
else:
    model_tfidf = models.TfidfModel(docs_bow)
    model_tfidf.save(tfidf_path)

docs_tfidf = nested_to_list(model_tfidf[docs_bow])

#%% train and convert to LSI representation
lsi_path = os.path.join(preprocessing_path, 'lsi.bin')
lsi_num_topics = 500
if os.path.exists(lsi_path):
    model_lsi = models.LsiModel.load(lsi_path)
else:
    model_lsi = models.LsiModel(docs_tfidf, id2word=dictionary, num_topics=lsi_num_topics)
    model_lsi.save(lsi_path)

docs_lsi = model_lsi[docs_tfidf]
```

The next code is an inevitable boilerplate required to prepare the data for skit-learn to process correctly. As noted above, we use a set of binary classifiers instead of a single multi-class one. And that's the reason we create a binary target for one of the classes ("SlowService" in case of that sample). You can change the value of `class_to_find` variable and rerun the code below in order to train every single class classifier individually. The scoring script works fine with multiple models and will load them automatically. Finally, the data is split into training and test datasets and rows with nulls are dropped entirely.

```
#%% create target
class_to_find = "SlowService"
df["Target"] = df.apply(lambda row: 1 if class_to_find in row["Class"] else 0, axis=1)
df.groupby(by=["Target"]).count()

#%% create features and targets dataset
features = pd.DataFrame(docs_features, columns=["F" + str(i) for i in range(lsi_num_topics)])
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
```

Now we are about to train the classifier (Logistic Regression in our case) and then save the model to the same shared directory, that was used for embedding transforms previously. As you might have noted in a code below we are using a special model name format `class_{0}_thresh_{1}.bin` to reuse information about the class name and the corresponding threshold later while scoring. 

The last note before we continue. For development I found Visual Studio Code to be really useful. It's a simple and lightweight editor that even allows to get a basic IntelliSense (code completion and hints) for such a dynamic language as Python. But at the same time [Jupyter]() and [Python]() extensions in conjunction with IPython kernel allows to run the code on a cell by cell basis and visualize the result without any need to rerun the script, that is always handy in ML tasks. Yes, it feels like normal Jupyter, but with IntelliSence and pure code/git oriented experience. I recommend you to give it a try at least while playing with the sample, as it uses a lot of others VS Code-oriented features for productive development.

In case of a code below, the cell named `plot ROC threshold values` is one of the examples of _Jupyter_ extension usage. You can click a special `Run cell` button above the cell to see TP and FP rate vs Threshold plots in the results pane to the right. We intensively used this chart while working with a partner, as due to the strong dataset imbalance, the optimal cut-off level was always about 0.04 instead of the usual 0.5. In case you are not able to use VS Code for testing, you can just run the script using the normal python tools and after seeings the results in a separate windows adjust the filename directly.

```
#%% train logistic classifier
classifier = LogisticRegression()
classifier.fit(features_train, target_train)

#%% score on test
scores_test = classifier.predict_proba(features_test)[:, 1]

#%% plot ROC threshold values
pd.DataFrame(nested_to_list(zip(tsh, tp_test, fp_test, fp_test-tp_test)), columns=['Threshold', 'True Positive Rate', 'False Positive Rate', 'Difference']).plot(x='Threshold')
plt.xlim(0, 1)
plt.ylim([0,1])
plt.grid()
plt.show()

#%% save model
threshold = 0.25
model_filename = 'class_{0}_thresh_{1}.bin'.format(class_to_find, threshold)
joblib.dump(classifier, os.path.join(model_path, model_filename))
```

Now it's time to have a look at a scoring script at [_Score\run.py_](). It's almost the same and reuses a lot of code from the original training experiment, explained earlier. Have a look at it from the [GitHub repo](). It accepts an input in the form of a csv file to score and outputs 2 different files one with scored classes and another with ids of unscorable rows. We'll explain the reason for file usage later, when we talk about operationalization.

At the end of this section let me explain the idea of using a set of binary classifiers instead of a single multiclass one. At first, it's much easier to start with, work through and optimize performance on classes independently. It even allows to use different math models for different classes or embeddings, as in a case of autoreplies that often have rather rigid structure and can be handled with simple bag-of-words. At the same time from IT Pro perspective with something like a code below it allows to plug-in new or change existing models by putting them at the same directory and without a need to disturb others. 

```
model_paths = [path for path in os.listdir(os.path.join('..', 'model')) if path.startswith('class_') ]

for model_path in model_paths:
    model = joblib.load(os.path.join('..', 'model', model_path))
    res = model.predict_proba(features_notnull)[:, 1]

    class_name = model_path.split('_')[1]
    threshold = float(model_path.rsplit('.', 1)[0].split('_')[-1])

    result.loc[:, "class_" + class_name] = res > threshold
    result.loc[:, "class_" + class_name + "_score"] = res
```

You can try the code even now with your own data from your local desktop without any operationalization needed. Clone the [repository](), follow the instructions to setup local Anaconda environment and install Visual Studio Code with required extension. Then put you data in the compatible format to _data\data.csv_ and click through _Experiment\TrainingExperiment.py_ to train the model for every class you want to score. Don't forget to delete the whole _model_ folder beforehand, as otherwise the code will try to reuse transforms and models from the sample. Then switch to _Score\run.py_ replace the data in _Score\debug\input.csv_ with yours and run the script line by line with help of _Jupyter_ extension. You can even go to VS Code's _Debug (Ctrl+Alt+D)_ section, select _Score (Python)_ as a configuration and by clicking green _Start Debugging_ button debug the code line by line using the editor. Once done, you will find the scoring results in _input.scores.csv_ and _input.unscorable.csv_ files from _Score\debug_ folder. 


## Operationalization

So, at that stage we had 2 scripts. One for model training [_Experiments\TraintExperiment.py_](), that saves transformation and trained model to shared directory and is supposed to be run on a local machine once needed. And the other [Score\run.py]() to run daily, scoring new emails in batches as they arrive. In this section we will talk about a way to operationalize it with help of Azure Functions. Functions are simple to use, allow to bind a script to a set of different triggers (HTTP, queues, storage blobs, WebHooks and etc.), provide a set of automatic output bindings and at the same time do it in a very cost affective way -- consumption plan costs only $0.000016 for every GB RAM used for every second of execution. But that price comes with a limit: your function cannot be executed for more then 10 minutes and use more than 1.5 GB of RAM. If it doesn't fit you needs you can always switch to a dedicated App Service plan, still having access to other advantages of serverless approach. But in our case with simple logistic regression and bathes with hundreds of emails it looked optimal.

From a programmatic point of view, Function is a folder, named after function itself (in our case just ["Score"]()), with 2 different files:
* `function.json`, containing Function binding configuration and other options (you can read about its format [here]())
* `run.py`, with a script to execute once trigger is hit

[`function.json`]() can be written in hand or configured using Azure Portal. In out case we ended up with something like a code below. The first binding named _inputcsv_ triggers a script every time a file with the name matching to the pattern `mail-classify/input/{input_file_name}.csv` appears in default[] Azure Blob Storage. Other two binding specify there to put output files once the function run successfully. Here, they are saved to the independent _output_ folder with the name consisting of the name of the incoming file with additional _"scored"_ and _"unscorable"_ suffixes. This allows you to put a file with any identifying name like GUID to _input_ and wait for others two to appear automagically with the same GUID in the _output_ folder.

```
{
  "bindings": [
    {
      "name": "inputcsv",
      "type": "blobTrigger",
      "path": "mail-classify/input/{input_file_name}.csv",
      "connection": "apmlstor",
      "direction": "in"
    },
    {
      "name": "scoredcsv",
      "type": "blob",
      "path": "mail-classify/output/{input_file_name}.scored.csv",
      "connection": "apmlstor",
      "direction": "out"
    },
    {
      "name": "unscorablecsv",
      "type": "blob",
      "path": "mail-classify/output/{input_file_name}.unscorable.csv",
      "connection": "apmlstor",
      "direction": "out"
    }
  ],
  "disabled": false
}
```

The `run.py` script for our Functions is almost identical to our initial nonoperationalized version. The only change is determined by the way Functions pass incoming and outgoing data streams. Regardless of input and output type you choose (e.g., HTTP Request, queue message, BLOB file, etc), the content will be held in a temporary file and its path will be written to an environment variable with the name of the corresponding binding. For example, in our case on every function run there will be created a file with a name like _"...\Binding\\[GUID\]\inputcsv"_ and that path will be kept in _inputcsv_ environment variable. The same will be done for each outgoing file. In accordance with that logic, we made a couple minor changes to the script.

```
# read file
input_path = os.environ['inputcsv']
input_dir = os.path.dirname(input_path)
input_name = os.path.basename(input_path)

corpus = CsvCorpusReader(input_dir, [input_name],
                         encoding="utf8",
                         default_text_selector=lambda row: row["Text"])

[...]

# write unscorables
unscorable_path = os.environ['unscorablecsv']
ids_null.to_csv(unscorable_path, index=False) # pandas DataFrame

[...]

# write scored emails
output_path = os.environ['scoredcsv']
result.to_csv(output_path) # pandas DataFrame
```

These are all the changes to the script, required to get a service triggered by the csv file in a BLOB storage and returning the other two with the results of processing. Honestly, we tried other triggers, but found that the most powerful Python feature becomes its curse in a serverless framework. In Python, module is not a static library to link as in many other languages, but a code to execute on every run. It is almost unnoticeable in such long-term solutions like services, but adds a rather big overhead in Function, assuming full execution of script every time. That complicates the use of HTTP triggers with Python, but batch scoring with BLOBs (popular in ML scenarios) allows to lower the that overhead per data row to sensible minimum. In case you still need to use real-time row-based triggering with Python you can try to switch to a dedicated App Service plan, as it allows to significantly increase the amount of the host's CPU resources and decrease import times. In our case, the ease of implementation and low price of Consumption plan overcame[] the need for quick execution.

Before continue, let's have a look at a way to simplify developer experience with Visual Studio Code. At the time of writing [Functions CLI]() was able to make initial Python scaffolding, but no debugging was available. However, the runtime environment is not that difficult to simulate using built-in VS Code features. File called [`.vscode\launch.json`]() is here to help, as it allows to configure debugging settings. As you can see from the json below we instructs VS Code to debug `${workspaceRoot}/Score/run.py` with working directory in `${workspaceRoot}/Score` and environment variables set to 3 mocked binding files. It's completely the way it looks like when executed with Functions runtime (_keep in mind current working directory while developing you script_). With this settings in place just go to VS Code's _Debug (Ctrl+Alt+D)_ section, select _Score (Python)_ as a configuration and by clicking green _Start Debugging_ button debug the code line by line using the editor.

```
[...]
{
    "name": "Score (Python)",
    "type": "python",
    "request": "launch",
    "stopOnEntry": true,
    "pythonPath": "${config:python.pythonPath}",
    "console": "integratedTerminal",
    "program": "${workspaceRoot}/Score/run.py",
    "cwd": "${workspaceRoot}/Score",
    "env": {
        "inputcsv": "${workspaceRoot}/Score/debug/input.csv",
        "outputcsv": "${workspaceRoot}/Score/debug/output.csv",
        "unscorablecsv": "${workspaceRoot}/Score/debug/unscorable.csv"
    },
    "debugOptions": [
        "RedirectOutput",
        "WaitOnAbnormalExit"
    ]
}
[...]
```

In case you want to use Jupyter extension for interactive cell by cell development and execution, you need such a configuration in code of a Function. After some experiments we stuck with something like a code below. It executes only in IPython environment and is ignored while normal execution with runtime or debugging.

```
if "IPython" in sys.modules and 'Score' not in os.getcwd():
    os.environ['inputcsv'] = os.path.join('debug', 'input.csv')    
    os.environ['scoredcsv'] = os.path.join('debug', 'input.scores.csv')
    os.environ['unscorablecsv'] = os.path.join('debug', 'input.unscorable.csv')
    os.chdir('Score')
```

## Environment Configuration

Once we have the model ready and Functions code prepared, it's time to configure required Azure infrastructure. At the time of writing Python support in Functions is still in preview and hence some additional configuration steps are required. By default, the runtime comes with a Python version 2.7 preinstalled. In order to switch to a more popular version 3.6 according to [official wiki](https://github.com/Azure/azure-webjobs-sdk-script/wiki/Using-a-custom-version-of-Python) you need to get any redistributable python package (you can use prepared environment) and put it to _D:\\home\\site\\tools_ directory. The way it works is simple. This directory precedes the one with default Python 2.7 in PATH variable.

You can do it manually using built-in Kudu UI as it shown in the article, but I found that a special function for it is more convenient.[`Setup`]() function shows the way we did it during the project. At first the Function checks if the installed version is 3.6, if not it downloads a preconfigured zip archive with Python and extracts it to the _D:\\home\\site\\tools_. 

```
tools_path = 'D:\\home\\site\\tools'
if not sys.version.startswith('3.6'):

    # in python 2.7
    import urllib
    print('Installing Python Version 3.6.3')

    from zipfile import ZipFile

    if not os.path.exists(tools_path):
        os.makedirs(tools_path)
        print("Created [{}]".format(tools_path))

    python_url = 'https://apmlstor.blob.core.windows.net/wheels/python361x64.zip'
    python_file = os.path.join(tools_path, 'python.zip')
    urllib.urlretrieve(python_url, python_file)
    print("Downloaded Python 3.6.3")

    python_zip = ZipFile(python_file, 'r')
    python_zip.extractall(tools_path)
    python_zip.close()
    print("Extracted Python to [{}]".format(tools_path))

    print("Please rerun this function again to install required pip packages")
    sys.exit(0)
```

Next, it's time to install required pip packages. Pip provides built-in API for Python, so it's as easy to use from Python as with normal command prompt. As you can see in a code below, normal python-only packages (_langid_, _pymorphy_) install just fine without any additional work needed. The only problem is packages build with C++. There is no Visual Studio compiler installed on the platform, so the only option here is to use precompiled wheels. Some of them already exist in the pip repository (you can check it [here](https://pythonwheels.com/)), for other ML specific ones [_Unofficial Windows Binaries for Python Extension Packages_](https://www.lfd.uci.edu/~gohlke/pythonlibs/) proved to be very useful. In my case I used my Azure Storage Container for making this packages available to Function, you are free to reuse that links or move them you private storage.

```
def install_package(package_name):
    pip.main(['install', package_name])

install_package('https://apmlstor.blob.core.windows.net/wheels/numpy-1.13.1%2Bmkl-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/pandas-0.20.3-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/scipy-0.19.1-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/scikit_learn-0.18.2-cp36-cp36m-win_amd64.whl')


install_package('https://apmlstor.blob.core.windows.net/wheels/gensim-2.3.0-cp36-cp36m-win_amd64.whl')
install_package('https://apmlstor.blob.core.windows.net/wheels/nltk-3.2.4-py2.py3-none-any.whl')
install_package('langid')
install_package('pymorphy2')
```

This approach has also proven to be usefull in implementation of additional post-configuration steps. For our solution to work, we needed two NLTK-specific corpuses to be installed afterwards. Below is a code that executes just after a set of `install_packages` commands.

```
import nltk;
nltk_path = os.path.abspath(os.path.join('..', 'lib', 'nltk_data'))
if not os.path.exists(nltk_path):
    os.makedirs(nltk_path)
    print("INFO: Created {0}".format(nltk_path))
    
nltk.download('punkt', download_dir=os.path.join('..', 'lib', 'nltk_data'))
nltk.download('stopwords', download_dir=os.path.join('..', 'lib', 'nltk_data'))
```

As all the code in `Setup` Function is idempotent, you can add other configuration steps or install additional packages without any worry. Please note, _for the first time this Function should run twice in order to switch to Python 3.6 and then use it to install required packages._

## Conslusion

Even in spite of the preconfiguration complexity and limitations of preview, the Azure Functions proved to be quite a convenient and effective way to operationalize Python ML models. In case of the original project, it was deployed to production and with ML model in place significantly improved result of exsing mechanical aproach. You are ecouraged to check, play with and reuse the code in the [sample repository]().


