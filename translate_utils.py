"""

IDE: PyCharm
Project: opennmt-samples
Author: Robin
Filename: translate_utils.py
Date: 05.04.2020

"""
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("pretrained_models/sentencepiece.model")


def sent_tokenize(text: str):
    """
    Apply sentence tokenization and return tokenized
    sentence as string
    :param text:
    :return:
    """
    pieces = sp.EncodeAsPieces(text)
    return " ".join(pieces)


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
