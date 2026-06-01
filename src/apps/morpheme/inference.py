import os
import sys
import time
import pandas as pd
import logging
import torch
import argparse
from torch import nn, optim
from apps.morpheme.config import *
from apps.morpheme.models.build_model import build_model
from apps.morpheme.data import Multi30k
from apps.morpheme.utils import get_bleu_score, greedy_decode
from konlpy.tag import Mecab

mecab = Mecab()
DATASET = Multi30k()


def inf(src):
    src = mecab.morphs(src)
    src = " ".join(src)
    torch.manual_seed(0)
    
    print(mecab.pos(src))

    resume_from = "./apps/morpheme/chk/1400.pt"

    # 불러오기
    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(resume_from, map_location="cuda:0")["model_state_dict"])

    result = DATASET.translate(model, src, greedy_decode)

    result = result.replace("<sos>", "")
    result = result.replace("<unk>", "")
    result = result.replace("<eos>", "")

    before = result

    # 사용자 사전 적용
    user_dict = pd.read_csv("./apps/morpheme/user_dic.csv", encoding="utf-8-sig")
    user_dict = user_dict.drop_duplicates(subset="morp")

    src_split = src.split(" ")
    
    print(result)
    print(src_split)

    t = before.split(" ")
    morp_list = list()

    for idx, item in enumerate(t):
        try:
            tmp = ""
            if item == "/":
                tmp = (t[idx - 1], t[idx + 1])
                morp_list.append(tmp)
        except:
            ConnectionRefusedError

    print(morp_list)

    for idx, item in enumerate(src_split):
        try:
            if src_split[idx].strip() not in item:
                print(src_split[idx])
                filtered_user_dict = user_dict[user_dict["morp"].str.strip() == src_split[idx].strip()]
                if not filtered_user_dict.empty:
                    replace = (filtered_user_dict.iloc[0]["morp"], filtered_user_dict.iloc[0]["label"])
                    morp_list[idx] = replace
        except:
            continue

    print(morp_list)

    return morp_list


def pos(src):
    return inf(src)
