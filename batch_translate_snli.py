"""

IDE: PyCharm
Project: opennmt-samples
Author: Robin
Filename: batch_translate_snli.py
Date: 05.04.2020

"""
import csv
import os
from argparse import Namespace

import pandas as pd
# load translator
from onmt.translate.translator import build_translator
from pandas import DataFrame
from tqdm import tqdm

from translate_utils import sent_tokenize, detokenize

path = 'averaged-10-epoch.pt'
opt = Namespace(alpha=0.0, batch_type='sents', beam_size=8, beta=-0.0, block_ngram_repeat=0, coverage_penalty='none',
                data_type='text', dump_beam='', fp32=False, gpu=0, ignore_when_blocking=[], length_penalty='none',
                max_length=100, max_sent_length=None, min_length=0, models=['pretrained_models/averaged-10-epoch.pt'],
                n_best=1, output="/log", phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, ratio=-0.0,
                replace_unk=False, report_align=False, report_time=False, seed=829, stepwise_penalty=False, tgt=None,
                verbose=False)
translator = build_translator(opt, report_score=False)


# around 14 translations/second

def translate_batch(df: DataFrame, start, idx):
    label = df[start: idx]["gold_label"].tolist()
    sent1 = df[start: idx]["sentence1"].map(lambda x: sent_tokenize(x)).tolist()
    sent2 = df[start: idx]["sentence2"].map(lambda x: sent_tokenize(x)).tolist()

    probs, sent1_trans = translator.translate(sent1, batch_size=len(sent1))
    probs, sent2_trans = translator.translate(sent2, batch_size=len(sent2))

    sent1_trans_detok = [detokenize(x) for x in sent1_trans]
    sent2_trans_detok = [detokenize(x) for x in sent2_trans]

    return [sent1_trans_detok, sent2_trans_detok, label]


data_path = "data/"
files = list(filter(lambda x: x.endswith(".txt"), os.listdir(data_path)))
batch_size = 64

for file in files:
    file_path = os.path.join(data_path, file)
    out_file_path = "output/" + file + ".csv"
    if os.path.exists(out_file_path):
        os.remove(out_file_path)

    df = pd.read_csv(file_path, sep='\t', header=0, nrows=300000)
    all_data = [["sentenceA", "sentenceB", "label"]]
    min_step = min(len(df), batch_size)
    for idx in tqdm(range(0, len(df), batch_size)):
        data = translate_batch(df, idx, idx + min_step)
        start = idx

        for i in range(len(data[0])):
            all_data.append([data[0][i], data[1][i], data[2][i]])

        with open(out_file_path, "a", encoding="utf8", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerows(all_data)

        all_data = []
