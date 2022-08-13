import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from konlpy.tag import Okt

pad_len = 0
model = KeyedVectors.load_word2vec_format("WordVec2.model")
embedding_dim = 100
zero_vector = np.zeros(embedding_dim, dtype=np.float64)


def get_sentence_list(datas):
    okt = Okt()

    tokenized_data = []
    for sentence in datas:
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        tokenized_data.append(tokenized_sentence)

    return tokenized_data


def sentences_to_vectors(sentences):
    result_list = []

    temp = 0
    index = 0
    count = 0
    while index < 100:
        for word in sentences:
            try:
                temp += model.get_vector(word)[index]
            except KeyError:
                temp = 0
                count += 1
        temp /= len(sentences) + count
        result_list.append(temp)
        temp = 0
        count = 0
        index += 1

    return result_list


def get_embedded_sentence(sentence_list):
    embedded_sentence = []
    global pad_len

    for sentence in sentence_list:
        temp_list = sentences_to_vectors(sentence)
        if len(temp_list) > pad_len:
            pad_len = len(temp_list)
        embedded_sentence.append(temp_list)

    return embedded_sentence


def save_model(sentence_list):
    tokenize_data = sentence_list
    model = Word2Vec(sentences=tokenize_data, vector_size=embedding_dim, window=5, min_count=3, workers=8, sg=0)
    model.wv.save_word2vec_format("WordVec2.model")
    return


def get_max_len():
    return embedding_dim


def sv_padding(sentencelist, pad_len):
    for sentence in sentencelist:
        while len(sentence) < pad_len:
            sentence += [0]


def wv_padding(wv, pad_len):
    while len(wv) < pad_len:
        wv += [0]


def get_pad_len():
    return pad_len
