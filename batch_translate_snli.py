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
from tqdm import tqdm

from translate_utils import translate_batch

# define global settings
data_path = "data/"
# files = list(filter(lambda x: x.endswith(".txt"), os.listdir(data_path)))
# files = ["snli_1.0_dev.txt"]
files = ["multinli_1.0_train.txt"]
batch_size = 128
beam_size = 16  # the higher the beam size the longer the translation runs
max_length = 50
path = 'averaged-10-epoch.pt'

# init translation model
opt = Namespace(alpha=0.0, batch_type='sents', beam_size=beam_size, beta=-0.0, block_ngram_repeat=0,
                coverage_penalty='none',
                data_type='text', dump_beam='', fp32=False, gpu=0, ignore_when_blocking=[], length_penalty='none',
                max_length=max_length, max_sent_length=None, min_length=0,
                models=['pretrained_models/averaged-10-epoch.pt'],
                n_best=1, output="/log", phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, ratio=-0.0,
                replace_unk=False, report_align=False, report_time=False, seed=829, stepwise_penalty=False, tgt=None,
                verbose=False)
translator = build_translator(opt, report_score=False)

# iterate over files and translate csv content
for file in files:
    print("Translating file %s" % file)
    file_path = os.path.join(data_path, file)
    out_file_path = "output/" + file[:-4] + ".de.csv"
    if os.path.exists(out_file_path):
        os.remove(out_file_path)

    df = pd.read_csv(file_path, sep='\t', header=0, error_bad_lines=False)
    all_data = [["sentenceA", "sentenceB", "label"]]
    min_step = min(len(df), batch_size)
    for idx in tqdm(range(0, len(df), batch_size)):
        data = translate_batch(translator, df, idx, idx + min_step, max_length)
        start = idx

        for i in range(len(data[0])):
            all_data.append([data[0][i], data[1][i], data[2][i]])

        with open(out_file_path, "a", encoding="utf8", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerows(all_data)

        all_data = []
