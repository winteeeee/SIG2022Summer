import pandas as pd


def filtering(string):
    corpus = pd.read_csv("/myproject/Filtering_Module/bad_word_corpus.csv")
    corpus = corpus['badwords'].values

    if string in corpus:
        return "*" * len(string)

    else:
        return string
