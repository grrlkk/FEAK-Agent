import argparse
import os, sys, time
import logging
import pandas as pd
import numpy as np
import torch
import spacy
import re
from torch import nn, optim
from config import *
from apps.morpheme.models.build_model import build_model
from apps.morpheme.data import Multi30k
from apps.morpheme.utils import get_bleu_score, greedy_decode
import apps.morpheme.mecab_main as MeCab

DATASET = Multi30k()


def main(args, target_sentence):
    device = torch.device("cuda:0")

    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device, dr_rate=DROPOUT_RATE)
    # model = build_model(42024, 42091, device, dr_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(args.resume_from, map_location=device)["model_state_dict"])

    # Inference용

    item = target_sentence
    result = DATASET.translate(model, item, greedy_decode)
    result = result.replace("<sos>", "")
    result = result.replace("<unk>", "")
    result = result.replace("<eos>", "")
    print(result)
    return result


""" test용
    target_file=pd.read_csv("/home/compu/Desktop/KDH/Untitled Folder/en.csv",sep="\t")
    result=pd.DataFrame({"en":[],
                         "de":[]})
    
    for idx in range(0, 305):#len(target_file)-1):
        item=target_file.loc[idx]
        print(idx)
        try:
            now_en=item[0]
            now_de=DATASET.translate(model, now_en, greedy_decode)
            now_de=now_de.replace("<sos>","")
            now_de=now_de.replace("<unk>","")
            now_de=now_de.replace("<eos>","")
            now_df=pd.DataFrame({"en":[now_en],
                                    "de":[now_de]})
            result=pd.concat([result,now_df])
            print(now_en)
            print(now_de)
        except:
            continue

    result.to_csv("sentencepiece_실험 결과_1500.csv",encoding="utf-8-sig")
    print("끝")
"""

if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", default="./chk_Mecab/1100.pt")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sentence = "동전을 투자해 본다."
    mecab = MeCab()
    morp = mecab.morphs(sentence)
    sentence = " ".join(morp)
    print(sentence)
    main(args, sentence)
