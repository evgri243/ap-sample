"""
Corpus module
"""

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus.reader.api import CorpusReader
from nltk.stem.snowball import SnowballStemmer
from utility import flatten_levels
import csv
from functools import reduce
import pandas as pd
import langid
from pymorphy2 import MorphAnalyzer 

# set field maxlength to bigger values enough to load answers
csv.field_size_limit(500 * 1024 * 1024)


class Levels:
    Para = [0]
    Sent = [1]

    Nothing = []
    All = Para + Sent


class CsvCorpusReader(CorpusReader):
    """
    A corpus reader for CSV files
    """

    def __init__(self, rootpath, fileids, encoding="utf8",
                 default_text_selector=lambda row: None, **kwargs):
        """
        Initialize CSV corpus reader

        Arguments:
        rootpath (str) - path to folder with corpus files (see NLTK CorpusReader for more info)
        fileids (list str) - names of files in root (see NLTK CorpusReader for more info)
        default_test_selector (lambda) - default selector that will be used to extract text from corpus
        **kwargs (named arguments) - arguemnts passed to csv.DictReader (see csv.DictReader for more info)
        """

        # Initialize base NLTK corpus reader object
        CorpusReader.__init__(self, rootpath, fileids, encoding=encoding)

        # Initialize default selector
        self.__default_text_selectors = default_text_selector

        # Save csv parser params
        self.csv_kwargs = kwargs

    def rows(self, fileids=None, columns=None):
        """
        Returns the complete corpus source in the form of list of dictionaries

        Arguments:
        fileids (list str) - fileids to use
        columnss (list str) - list of columns to select (default: all)
        """

        fileids = fileids if fileids is not None else self.fileids()

        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with open(path, encoding=self._encoding) as f:
                reader = csv.DictReader(f, **self.csv_kwargs)
                for row in reader:
                    columns = columns if columns is not None else row.keys()
                    yield {k: row[k] for k in columns}

    def rows_as_dataframe(self, fileids=None, columns=None, **dataframe_params):
        """
        Return corpus content as a pandas dataframe

        Arguemnts:
        filesid (list str) - fileids to use
        **dataframe_params (named arguments) - any named arguments to pass to Dataframe constructor
        """

        return pd.DataFrame(self.rows(fileids), columns=columns, **dataframe_params)

    def docs(self, fileids=None, text_selector=None, keep_langs_only=[]):
        """
        Return curpus text as a list of strings

        Arguments:
        filesid (list str) - fileids to use
        default_test_selector (lambda) - function to select text from dictionary returned by content (default: default_text_selector)
        keep_langs_only (list str) - languages to filter (e.g. 'en', 'ru'), use [] to skip filtering (default: [])
        """

        text_selector = text_selector if text_selector is not None else self.__default_text_selectors

        for row in self.rows(fileids):
            text = text_selector(row)
            if not keep_langs_only or langid.classify(text)[0] in keep_langs_only:
                yield text

    def paras(self, fileids=None, text_selector=None, keep_langs_only=[]):
        """
        Returns paragraphs as a nested list of strings

        Arguemnts:
        filesid (list str) - fileids to use
        default_test_selector (lambda) - function to select text from dictionary returned by content (default: default_text_selector)
        keep_langs_only (list str) - languages to filter (e.g. 'en', 'ru'), use [] to skip filtering (default: [])
        """

        docs = (
            (
                para for para in doc.split('\n') if para
            ) for doc in self.docs(fileids, text_selector, keep_langs_only=keep_langs_only)
        )

        return docs

    def sents(self, fileids=None, text_selector=None, keep_levels=Levels.Para, keep_langs_only=[]):
        """
        Returns paragraphs as a nested list of strings

        Arguemnts:
        filesid (list str) - fileids to use
        default_test_selector (lambda) - function to select text from dictionary returned by content (default: default_text_selector)
        keep_levels (lint int) - levels to keep (list of docs = 0, list of paras = 1 and etc; default: keep docs + para)        
        keep_langs_only (list str) - languages to filter (e.g. 'en', 'ru'), use [] to skip filtering (default: [])
        """

        docs = (
            (
                (
                    sent for sent in sent_tokenize(para) if sent
                ) for para in doc.split('\n') if para
            ) for doc in self.docs(fileids, text_selector, keep_langs_only=keep_langs_only)
        )

        levels = set(Levels.Para) - set(keep_levels)
        return (flatten_levels(doc, levels) for doc in docs)

    def words(self, fileids=None, text_selector=None,
              keep_levels=Levels.All, keep_langs_only=[],
              word_predicate=(lambda w: True), word_processor=(lambda w: w)):
        """
        Returns paragraphs as a nested list of strings

        Arguemnts:
        filesid (list str) - fileids to use
        default_test_selector (lambda) - function to select text from dictionary returned by content (default: default_text_selector)
        keep_levels (lint int) - levels to keep (list of docs = 0, list of paras = 1 and etc; default: keep docs + para)
        keep_langs_only (list str) - languages to filter (e.g. 'en', 'ru'), use [] to skip filtering (default: [])
        word_predicate (lambda bool) - function to filter words (default: lambda w: True)
        word_predicate (lambda bool) - function to apply on exery word (default: lambda w: w)
        """

        docs = (
            (
                process_text(doc, keep_levels=keep_levels,
                             word_predicate=word_predicate, word_processor=word_processor)
            ) for doc in self.docs(fileids, text_selector, keep_langs_only=keep_langs_only)
        )

        return docs


def generate_processor(keep_alpha_only=True, to_lower=True,
                       stopwords_langs=[], add_stopwords=None,
                       stemmer_langs=[],
                       normalize_russian=False):
    """
    Return word predicate filter and processor

    Arguments:
    keep_alpha_only (bool) - keep only alpha symbols
    to_lower (bool) - convert to lower case
    stopwords_langs (list str) - filter stopwords for languages, use [] for no filtering (default: [])
    add_stopwords (list str) - additional stopwords to filter
    stemmer_langs (bool) - stem words using stemmer for specified language, use None for no stemming (default: None)
    normalize_russian (bool) - normalize Russian with PyMorph2 (fefault: False) 
    """

    def idf(w): return w

    def truef(w): return True

    stops = [w.lower() for w in add_stopwords] if add_stopwords is not None else []
    for stopwords_lang in stopwords_langs:
        stops += stopwords.words(stopwords_lang)

    def _stop_func(w):
        return w.lower() not in stops
    
    is_not_stop = _stop_func if stops else truef
    is_alpha = (lambda w: w.isalpha()) if keep_alpha_only else truef

    stemmers = []
    for stemmer_lang in stemmer_langs:
        stemmer = SnowballStemmer(stemmer_lang)
        stemmers += [stemmer.stem]

    analyzer = MorphAnalyzer()
    normalize = (lambda w: analyzer.normal_forms(w)[0]) if normalize_russian else idf

    def stem(w): return reduce((lambda w, s: s(w)), stemmers, w)

    lower = (lambda w: w.lower()) if to_lower else idf

    return {
        "word_predicate": lambda w: is_not_stop(w) and is_alpha(w),
        "word_processor": lambda w: stem(normalize(lower(w)))
    }


def process_text(text, keep_levels=Levels.All,
                 word_predicate=(lambda w: True), word_processor=(lambda w: w)):
    """
    Split text to a sequence of words and apply processor

    keep_levels (list int) - levels to keep (list of docs = 0, list of paras = 1 and etc; default: keep docs + para)
    keep_langs_only (list str) - languages to filter (e.g. 'en', 'ru'), use [] to skip filtering (default: [])
    word_predicate (lambda bool) - function to filter words (default: lambda w: True)
    word_predicate (lambda bool) - function to apply on exery word (default: lambda w: w)
    """

    paras = (
        (
            (
                word_processor(word)
                for word in word_tokenize(sent)
                if word_predicate(word_processor(word))
            ) for sent in sent_tokenize(para) if sent.strip()
        ) for para in text.split('\n') if para.strip()
    )

    levels = set(Levels.All) - set(keep_levels)
    return flatten_levels(paras, levels)
