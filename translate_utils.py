"""

IDE: PyCharm
Project: opennmt-samples
Author: Robin
Filename: translate_utils.py
Date: 05.04.2020

"""
import sys

import sentencepiece as spm
from pandas import DataFrame

sp = spm.SentencePieceProcessor()
sp.Load("pretrained_models/sentencepiece.model")


def translate_batch(translator, df: DataFrame, start, idx, max_length):
    label = df[start: idx]["gold_label"].tolist()
    sent1 = df[start: idx]["sentence1"].map(lambda x: sent_tokenize(x, max_length)).tolist()
    sent2 = df[start: idx]["sentence2"].map(lambda x: sent_tokenize(x, max_length)).tolist()

    probs, sent1_trans = translator.translate(sent1, batch_size=len(sent1))
    probs, sent2_trans = translator.translate(sent2, batch_size=len(sent2))

    sent1_trans_detok = [detokenize(x) for x in sent1_trans]
    sent2_trans_detok = [detokenize(x) for x in sent2_trans]

    return [sent1_trans_detok, sent2_trans_detok, label]


def sent_tokenize(text: str, max_length=sys.maxsize):
    """
    Apply sentence tokenization and return tokenized
    sentence as string
    :param text:
    :return:
    """
    if text is not None and type(text) == str and len(text.strip().split()) > 0:
        pieces = sp.EncodeAsPieces(text)[:max_length]
        return " ".join(pieces)
    return str(text)


def detokenize(text: list):
    """
    Converts a sentencepiece tokenized text into normal text
    :param text:
    :return:
    """
    tokens = text[0].split()
    word_start_prefix = "▁"

    words = []
    word = ""
    for token in tokens:
        if token.startswith(word_start_prefix):
            words.append(word)
            word = ""
        token = token.replace(word_start_prefix, "")
        word += token
    words.append(word)

    sentence = " ".join(words).strip()
    return sentence
