import collections
import inspect
import json
import logging
import multiprocessing
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from apps.cohesion.essay_scoring.essay_scoring import load_essay_model, score_results
from apps.morph.bareun import bareun
from apps.morph.utagger import utagger
from google.protobuf.json_format import MessageToDict
from keybert import KeyBERT
from transformers import BertModel

from . import TTR, adjacent_overlap, counter, similarity, textpreprocess

logging.basicConfig(level=logging.INFO)


def initialize_models():
    with torch.no_grad():
        global key_model, kw_model, simil_model, device, morph, morph2, bert_model, gru_model, tokenizer
        device = "cuda"

        key_model = BertModel.from_pretrained("skt/kobert-base-v1")
        kw_model = KeyBERT(model=key_model)

        simil_model = similarity.model()
        simil_model.to(device)

        morph = bareun()
        morph2 = utagger()

        # kkma = inference.inf(text)

        bert_model, gru_model, tokenizer = load_essay_model(device)


# 대명사 목록, 지시대명사 -> 인칭대명사 순서
pronounList = [
    "이",
    "그",
    "저",
    "이것",
    "그것",
    "저것",
    "무엇",
    "여기",
    "저기",
    "거기",
    "어디",
    "저희",
    "본인",
    "그대",
    "귀하",
    "너희",
    "당신",
    "여러분",
    "임자",
    "자기",
    "자네",
    "이런",
    "그들",
    "그녀",
    "당신",
    "저희",
    "놈",
    "얘",
    "걔",
    "쟤",
    "누구",
]
morphs_C = [  # 내용
    "NNG",
    "NNP",
    "NNB",
    "NNBC",
    "NP",
    "NR",
    "VV",
    "VA",
    "VX",
    "VCP",
    "VCN",
    "MM",
    "MMA",
    "MMD",
    "MMN",
    "MAG",
    "MAJ",
    "IC",
    "XR",
]


def processTTR(kkma):
    result = collections.defaultdict()
    result = TTR.TTR(kkma).get_TTR()
    return result


def processSimilarity(text):
    # topic & similarity -----------------------------------------------------------------
    result = collections.defaultdict()
    result["avgSentSimilarity"], result["topicConsistency"] = similarity.similar(
        text, simil_model, kw_model, device=device
    )

    return result


def processAdjacency(kkma_list):
    result = collections.defaultdict()

    # All lemmas -------------------------------------------------------------------------
    result["adjacent_sentence_overlap_all_lemmas"] = 0
    result["adjacent_sentence_overlap_all_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_all_lemmas"] = 0

    result["adjacent_two_sentence_overlap_all_lemmas"] = 0
    result["adjacent_two_sentence_overlap_all_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_all_lemmas"] = 0

    # content lemmas ---------------------------------------------------------------------
    result["adjacent_sentence_overlap_content_lemmas"] = 0
    result["adjacent_sentence_overlap_content_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_content_lemmas"] = 0

    result["adjacent_two_sentence_overlap_content_lemmas"] = 0
    result["adjacent_two_sentence_overlap_content_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_content_lemmas"] = 0

    # function lemmas --------------------------------------------------------------------
    result["adjacent_sentence_overlap_function_lemmas"] = 0
    result["adjacent_sentence_overlap_function_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_function_lemmas"] = 0

    result["adjacent_two_sentence_overlap_function_lemmas"] = 0
    result["adjacent_two_sentence_overlap_function_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_function_lemmas"] = 0

    # noun lemmas ------------------------------------------------------------------------
    result["adjacent_sentence_overlap_noun_lemmas"] = 0
    result["adjacent_sentence_overlap_noun_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_noun_lemmas"] = 0

    result["adjacent_two_sentence_overlap_noun_lemmas"] = 0
    result["adjacent_two_sentence_overlap_noun_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_noun_lemmas"] = 0

    # verb lemmas ------------------------------------------------------------------------
    result["adjacent_sentence_overlap_verb_lemmas"] = 0
    result["adjacent_sentence_overlap_verb_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_verb_lemmas"] = 0

    result["adjacent_two_sentence_overlap_verb_lemmas"] = 0
    result["adjacent_two_sentence_overlap_verb_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_verb_lemmas"] = 0

    # adjective lemmas -------------------------------------------------------------------
    result["adjacent_sentence_overlap_adjective_lemmas"] = 0
    result["adjacent_sentence_overlap_adjective_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_adjective_lemmas"] = 0

    result["adjacent_two_sentence_overlap_adjective_lemmas"] = 0
    result["adjacent_two_sentence_overlap_adjective_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_adjective_lemmas"] = 0

    # adverb lemmas ----------------------------------------------------------------------
    result["adjacent_sentence_overlap_adverb_lemmas"] = 0
    result["adjacent_sentence_overlap_adverb_lemmas_normed"] = 0
    result["binary_adjacent_sentence_overlap_adverb_lemmas"] = 0

    result["adjacent_two_sentence_overlap_adverb_lemmas"] = 0
    result["adjacent_two_sentence_overlap_adverb_lemmas_normed"] = 0
    result["binary_adjacent_two_sentence_overlap_adverb_lemmas"] = 0

    # -------------------------------------------------------------------------------------
    for idx in range(len(kkma_list) - 1):
        result[
            "adjacent_sentence_overlap_all_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_all_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_all_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_all_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_all_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_all_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_content_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_content_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_content_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_content_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_content_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_content_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_function_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_function_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_function_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_function_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_function_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_function_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_noun_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_noun_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_noun_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_noun_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_noun_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_noun_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_verb_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_verb_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_verb_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_verb_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_verb_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_verb_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_adjective_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_adjective_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_adjective_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_adjective_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_adjective_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_adjective_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_sentence_overlap_adverb_lemmas"
        ] += adjacent_overlap.adjacent_sentence_overlap_adverb_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "adjacent_sentence_overlap_adverb_lemmas_normed"
        ] += adjacent_overlap.adjacent_sentence_overlap_adverb_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1]
        )
        result[
            "binary_adjacent_sentence_overlap_adverb_lemmas"
        ] += adjacent_overlap.binary_adjacent_sentence_overlap_adverb_lemmas(
            kkma_list[idx], kkma_list[idx + 1]
        )

    # -------------------------------------------------------------------------------------
    for idx in range(len(kkma_list) - 2):
        result[
            "adjacent_two_sentence_overlap_all_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_all_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_all_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_all_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_all_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_all_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_content_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_content_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_content_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_content_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        result[
            "binary_adjacent_two_sentence_overlap_content_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_content_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_function_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_function_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_function_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_function_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_function_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_function_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_noun_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_noun_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_noun_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_noun_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_noun_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_noun_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_verb_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_verb_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_verb_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_verb_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_verb_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_verb_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_adjective_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_adjective_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_adjective_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_adjective_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_adjective_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_adjective_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

        # --------------------------------------------------------------------------------
        result[
            "adjacent_two_sentence_overlap_adverb_lemmas"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_adverb_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "adjacent_two_sentence_overlap_adverb_lemmas_normed"
        ] += adjacent_overlap.adjacent_two_sentence_overlap_adverb_lemmas_normed(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )
        result[
            "binary_adjacent_two_sentence_overlap_adverb_lemmas"
        ] += adjacent_overlap.binary_adjacent_two_sentence_overlap_adverb_lemmas(
            kkma_list[idx], kkma_list[idx + 1], kkma_list[idx + 2]
        )

    return result


def processCohesion(kkma_list, kkma, kkma_by_sent):
    kkma = [morp for morp in kkma if not morp[1].startswith("S")]
    kkma_by_sent = [
        [morp for morp in sent if not morp[1].startswith("S")] for sent in kkma_by_sent
    ]
    result = collections.defaultdict()

    # 반복 형태소 비율 =================================================================================================
    kkma_C = [morp for morp in kkma if morp[1] in morphs_C]
    kkma_N = [morp for morp in kkma if morp[1].startswith("N")]
    kkma_NN = [morp for morp in kkma if morp[1].startswith("NN")]
    kkma_V = [morp for morp in kkma if morp[1].startswith("V")]

    result["C_repRatio"] = len(kkma_C) / len(kkma)
    result["N_repRatio"] = len(kkma_N) / len(kkma)
    result["NN_repRatio"] = len(kkma_NN) / len(kkma)
    result["V_repRatio"] = len(kkma_V) / len(kkma)

    # 반복 형태소 비율 (문장별) ========================================================================================
    result_adj = processAdjacency(kkma_by_sent)
    result.update(result_adj)

    # 반복 형태소 비율 new ============================================================================================
    sent_sets = [set(m for m, t in sent) for sent in kkma_by_sent]
    content_sets = [
        set(m for m, t in sent if t in morphs_C) for sent in kkma_by_sent
    ]  # 내용
    substantive_sets = [
        set(m for m, t in sent if t.startswith("N")) for sent in kkma_by_sent
    ]  # 체언
    noun_sets = [
        set(m for m, t in sent if t.startswith("NN")) for sent in kkma_by_sent
    ]  # 명사

    total_tokens = sum(len(sent) for sent in kkma_by_sent)
    total_content_tokens = sum(len(content) for content in content_sets)
    total_noun_tokens = sum(len(noun) for noun in noun_sets)
    total_substantive_tokens = sum(len(substantive) for substantive in substantive_sets)

    num_sentences = len(kkma_by_sent)

    result["C_adjOverlap"] = 0
    result["NN_adjOverlap"] = 0
    result["morph_adjOverlap"] = 0
    result["N_adjOverlap"] = 0

    result["C_adjOverlapToken"] = 0
    result["NN_adjOverlapToken"] = 0
    result["morph_adjOverlapToken"] = 0
    result["N_adjOverlapToken"] = 0

    result["C_adjOverlap3"] = 0
    result["NN_adjOverlap3"] = 0
    result["morph_adjOverlap3"] = 0
    result["N_adjOverlap3"] = 0

    result["C_adjOverlap3Token"] = 0
    result["NN_adjOverlap3Token"] = 0
    result["morphe_adjOverlap3Token"] = 0
    result["N_adjOverlap3Token"] = 0

    if num_sentences > 1:
        for i in range(num_sentences - 1):
            result["C_adjOverlap"] += bool(content_sets[i] & content_sets[i + 1])
            result["NN_adjOverlap"] += bool(noun_sets[i] & noun_sets[i + 1])
            result["morph_adjOverlap"] += bool(sent_sets[i] & sent_sets[i + 1])
            result["N_adjOverlap"] += bool(
                substantive_sets[i] & substantive_sets[i + 1]
            )

            result["C_adjOverlapToken"] += len(content_sets[i] & content_sets[i + 1])
            result["NN_adjOverlapToken"] += len(noun_sets[i] & noun_sets[i + 1])
            result["morph_adjOverlapToken"] += len(sent_sets[i] & sent_sets[i + 1])
            result["N_adjOverlapToken"] += len(
                substantive_sets[i] & substantive_sets[i + 1]
            )

        result["C_adjOverlapRatio"] = result["C_adjOverlap"] / (num_sentences - 1)
        result["NN_adjOverlapRatio"] = result["NN_adjOverlap"] / (num_sentences - 1)
        result["morph_adjOverlapRatio"] = result["morph_adjOverlap"] / (
            num_sentences - 1
        )
        result["N_adjOverlapRatio"] = result["N_adjOverlap"] / (num_sentences - 1)

        result["C_adjOverlapTokenRatio"] = (
            result["C_adjOverlapToken"] / total_content_tokens
            if total_content_tokens > 0
            else 0
        )
        result["NN_adjOverlapTokenRatio"] = (
            result["NN_adjOverlapToken"] / total_noun_tokens
            if total_noun_tokens > 0
            else 0
        )
        result["morph_adjOverlapTokenRatio"] = (
            result["morph_adjOverlapToken"] / total_tokens if total_tokens > 0 else 0
        )
        result["N_adjOverlapTokenRatio"] = (
            result["N_adjOverlapToken"] / total_substantive_tokens
            if total_substantive_tokens > 0
            else 0
        )

    if num_sentences > 2:
        for i in range(num_sentences - 2):
            result["C_adjOverlap3"] += bool(content_sets[i] & content_sets[i + 2])
            result["NN_adjOverlap3"] += bool(noun_sets[i] & noun_sets[i + 2])
            result["morph_adjOverlap3"] += bool(sent_sets[i] & sent_sets[i + 2])
            result["N_adjOverlap3"] += bool(
                substantive_sets[i] & substantive_sets[i + 2]
            )

            result["C_adjOverlap3Token"] += len(content_sets[i] & content_sets[i + 2])
            result["NN_adjOverlap3Token"] += len(noun_sets[i] & noun_sets[i + 2])
            result["morphe_adjOverlap3Token"] += len(sent_sets[i] & sent_sets[i + 2])
            result["N_adjOverlap3Token"] += len(
                substantive_sets[i] & substantive_sets[i + 2]
            )

        result["C_adjOverlap3Ratio"] = result["C_adjOverlap3"] / (num_sentences - 2)
        result["NN_adjOverlap3Ratio"] = result["NN_adjOverlap3"] / (num_sentences - 2)
        result["morph_adjOverlap3Ratio"] = result["morph_adjOverlap3"] / (
            num_sentences - 2
        )
        result["N_adjOverlap3Ratio"] = result["N_adjOverlap3"] / (num_sentences - 2)

        result["C_adjOverlap3TokenRatio"] = (
            result["C_adjOverlap3Token"] / total_content_tokens
            if total_content_tokens > 0
            else 0
        )
        result["NN_adjOverlap3TokenRatio"] = (
            result["NN_adjOverlap3Token"] / total_noun_tokens
            if total_noun_tokens > 0
            else 0
        )
        result["morph_adjOverlap3TokenRatio"] = (
            result["morphe_adjOverlap3Token"] / total_tokens if total_tokens > 0 else 0
        )
        result["N_adjOverlap3TokenRatio"] = (
            result["N_adjOverlap3Token"] / total_substantive_tokens
            if total_substantive_tokens > 0
            else 0
        )
    return result


def processReadability(text, kkma, sentences, grade):
    result = collections.defaultdict()

    # 0.1579*(어려운 단어 수(기초 어휘 목록에 없는) / 전체 형태소 수 * 100)+0.0496*(전체 형태소 수 / 문장 수)
    result["text_dalechall"] = (
        0.1579
        * (
            sum(
                item["cnt"]
                for grade_level, items in grade
                for item in items
                if int(grade_level) > 2
            )
            / len(kkma)
            * 100
        )
        + 0.0496 * (len(kkma) / len(sentences))
        + 3.6365
    )

    result["text_flesch"] = (
        206.835 - 1.015 * (len(kkma) / len(sentences)) - 84.6 * (len(text) / len(kkma))
    )

    result["text_fleschkincaid"] = (
        0.39 * (len(kkma) / len(sentences)) + 11.8 * (len(text) / len(kkma)) - 15.59
    )

    C = [m for m in kkma if m[1] in morphs_C]
    C_CTTR = TTR.cttr(C)
    result["text_oridx"] = ((0.7 * C_CTTR + 0.3 * len(C)) * 500) + 100
    return result


def process(
    text,
    targets=["ttr", "similarity", "basic", "adjacency", "readability", "essay_score"],
):
    result = collections.defaultdict()

    # text preprocessing -----------------------------------------------------------------
    sentences = textpreprocess.splitText(text)
    words = textpreprocess.splitSen(sentences)

    kkma = []
    kkma_by_sent = []
    kkma_simple = []
    kkma_list = []
    voc_grades = []

    result["morpheme"] = morph.tags(sentences).as_json()
    corrections = morph.correction(text)
    result["correction"] = MessageToDict(corrections)

    for idx, sentence in enumerate(sentences):
        inf = morph.pos(sentence)
        inf_simple = morph.pos(sentence)

        kkma_list.append(inf)
        kkma += kkma_list[idx]
        kkma_simple += inf_simple
        kkma_by_sent.append(inf)

        voc_grades += morph2.grade(sentence)

    grade = []
    grade_dict = collections.defaultdict(list)
    for item in voc_grades:
        if not any(d["voc"] == item["voc"] for d in grade_dict[str(item["grade"])]):
            grade_dict[str(item["grade"])].append(
                {
                    "voc": item["voc"],
                    "pos_tagged": item["pos_tagged"],
                    "pos": item["pos"],
                    "type": item["type"],
                    "meaning": item["meaning"],
                    "field": item["field"],
                    "cnt": voc_grades.count(item),
                }
            )

    grade = [(k, v) for k, v in grade_dict.items()]
    result["voc_grades"] = grade

    # processing -------------------------------------------------------------------------
    num_workers = min(2, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        if "ttr" in targets:
            curr_time = time.time()
            futures["ttr"] = executor.submit(processTTR, kkma)

        if "similarity" in targets:
            curr_time = time.time()
            futures["similarity"] = executor.submit(processSimilarity, text)

        if "adjacency" in targets:
            curr_time = time.time()
            if len(kkma_list) < 2:
                result["adjacency"] = []
            else:
                futures["adjacency"] = executor.submit(
                    processCohesion, kkma_list, kkma, kkma_by_sent
                )

        if "basic" in targets:
            curr_time = time.time()
            futures["basic"] = executor.submit(
                counter.counter,
                text,
                sentences,
                words,
                kkma,
                kkma_list,
                kkma_simple,
                kkma_by_sent,
            )

        if "readability" in targets:
            curr_time = time.time()
            futures["readability"] = executor.submit(
                processReadability, text, kkma, sentences, grade
            )

        for key, future in futures.items():
            logging.info(f"Waiting for {key} to complete...")
            result[key] = future.result()
            logging.info(f"{key} completed. {time.time() - curr_time}")

        if "basic" in targets and "basic" in futures:
            temp = result["basic"]
            result.pop("basic")
            result["basic_count"] = temp["basic_count"]
            result["basic_density"] = temp["basic_density"]
            result["basic_level"] = temp["basic_level"]
            result["basic_list"] = temp["basic_list"]
            result["NDW"] = temp["NDW"]
            result["sentenceLvl"] = temp["sentenceLvl"]
            result["sentenceLvlRep"] = temp["sentenceLvlRep"]
            result["sentenceLvlRep_list"] = temp["sentenceLvlRep_list"]

        if "essay_score" in targets:
            curr_time = time.time()
            try:
                essay_score = score_results(result, bert_model, gru_model, tokenizer)
                result["essay_score"] = essay_score
            except Exception as e:
                print(f"Error scoring essay: {e}")
                result["essay_score"] = "error"
            logging.info(f"essay_score completed. {time.time() - curr_time}")

    return result
