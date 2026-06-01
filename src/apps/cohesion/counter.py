import collections
import math
import string
import logging

logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨 이상만 출력
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 어휘와 형태소의 평균 길이/표준편차 계산
# 문장/단어의 복잡성 평가 (짧은 단어 위주인지, 긴 단어를 자주 쓰는지)
def avgLen(words, kkma_simple):
    morph_len = [len(kkma[0]) for kkma in kkma_simple]
    morph_avg = sum(morph_len) / len(morph_len)
    morph_std = math.sqrt(sum([(x - morph_avg) ** 2 for x in morph_len]) / len(morph_len))

    for i in range(len(words)):
        words[i] = words[i].translate(str.maketrans("", "", string.punctuation))

    words_len = [len(word) for word in words]
    words_avg = sum(words_len) / len(words_len)
    words_std = math.sqrt(sum([(x - words_avg) ** 2 for x in words_len]) / len(words_len))

    return morph_avg, morph_std, words_avg, words_std


# 형태소 단위 n-gram 리스트 생성
def ngram(kkma, n):
    result = []
    for i in range(len(kkma) - n + 1):
        ngram = kkma[i : i + n]
        ng = []

        for j in range(n):
            ng.append(ngram[j][0])

        ngg = " ".join(ng)
        result.append(ngg)
    return result


# 인접 문장 간 n-gram 겹침 정도 측정
def ngram_overlap(kkma_by_sent, overlap, n):
    overlap_counts = []
    if len(kkma_by_sent) < overlap:
        return 0
    for i in range(len(kkma_by_sent) - overlap + 1):
        ngram_sets = [set(ngram(kkma_by_sent[j], n)) for j in range(i, i + overlap)]
        overlap_count = len(set.intersection(*ngram_sets))
        overlap_counts.append(overlap_count)
    return sum(overlap_counts)

# 단어 또는 문자 n-gram을 빈도별로 분류
# 이거 실질/형식 형태소로 n-gram 만든 후 빈도별 목록화 할 때 사용하면 될 듯
def ngram_list(words, n):
    result = []
    for i in range(len(words) - n + 1):
        ngram = words[i : i + n]
        ng = " ".join(ngram)
        result.append(ng)

    ngram_counts = collections.Counter(result)
    ngram_key_value_list = collections.defaultdict(list)
    for ngram, count in ngram_counts.items():
        ngram_key_value_list[str(count)].append(ngram)
    return ngram_key_value_list


def sentence_level(sentences, words, kkma, kkma_by_sent, paraCnt):
    result = collections.defaultdict()
    resultRep = collections.defaultdict()
    resultRepList = collections.defaultdict()

    sentCnt = len(sentences)
    M_Cnt = sum(1 for morp in kkma if morp[1].startswith("M"))
    MM_Cnt = sum(1 for morp in kkma if morp[1].startswith("MM"))
    MA_Cnt = sum(1 for morp in kkma if morp[1].startswith("MA"))

    result["M_sentLen"] = M_Cnt / sentCnt
    result["MM_sentLen"] = MM_Cnt / sentCnt
    result["MA_sentLen"] = MA_Cnt / sentCnt

    # Sentence Length avg and std ---------------------------------------------------------
    result["char_sentLenAvg"] = sum(len(sent) for sent in sentences) / sentCnt
    result["morph_sentLenAvg"] = sum(len(sent) for sent in kkma_by_sent) / sentCnt
    result["word_sentLenAvg"] = sum(len(sent.split()) for sent in sentences) / sentCnt
    result["V_sentLenAvg"] = sum(1 for morp in kkma if morp[1].startswith("V")) / sentCnt
    if sentCnt > 1:
        result["char_sentLenStd"] = math.sqrt(
            sum((len(sent) - result["char_sentLenAvg"]) ** 2 for sent in sentences) / (sentCnt - 1)
        )
        result["morph_sentLenStd"] = math.sqrt(
            sum((len(sent) - result["morph_sentLenAvg"]) ** 2 for sent in kkma_by_sent) / (sentCnt - 1)
        )
        result["word_sentLenStd"] = math.sqrt(
            sum((len(sent.split()) - result["word_sentLenAvg"]) ** 2 for sent in sentences) / (sentCnt - 1)
        )
        result["V_sentLenStd"] = math.sqrt(
            sum((1 - result["V_sentLenAvg"]) ** 2 for morp in kkma if morp[1].startswith("V")) / (sentCnt - 1)
        )
    else:
        result["char_sentLenStd"] = 0
        result["morph_sentLenStd"] = 0
        result["word_sentLenStd"] = 0
        result["V_sentLenStd"] = 0

    # paragraph level ---------------------------------------------------------------------
    result["char_paraLenAvg"] = sum(len(sent) for sent in sentences) / paraCnt
    result["morph_paraLenAvg"] = sum(len(sent) for sent in kkma_by_sent) / paraCnt
    result["word_paraLenAvg"] = sum(len(sent.split()) for sent in sentences) / paraCnt
    result["V_paraLenAvg"] = sum(1 for morp in kkma if morp[1].startswith("V")) / paraCnt
    if paraCnt == 1:
        result["char_paraLenStd"] = 0
        result["morph_paraLenStd"] = 0
        result["word_paraLenStd"] = 0
        result["V_paraLenStd"] = 0
    else:
        result["char_paraLenStd"] = math.sqrt(
            sum((len(sent) - result["char_paraLenAvg"]) ** 2 for sent in sentences) / (paraCnt - 1)
        )
        result["morph_paraLenStd"] = math.sqrt(
            sum((len(sent) - result["morph_paraLenAvg"]) ** 2 for sent in kkma_by_sent) / (paraCnt - 1)
        )
        result["word_paraLenStd"] = math.sqrt(
            sum((len(sent.split()) - result["word_paraLenAvg"]) ** 2 for sent in sentences) / (paraCnt - 1)
        )
        result["V_paraLenStd"] = math.sqrt(
            sum((1 - result["V_paraLenAvg"]) ** 2 for morp in kkma if morp[1].startswith("V")) / (paraCnt - 1)
        )

    # NGRAM
    for i in range(2, 9):
        ng = ngram(kkma, i)
        ngramCnt = collections.Counter(ng)

        # sentence Ngram
        resultRep[f"morph_sentNgram2_N{i}"] = sum(1 for n in ngramCnt.values() if n > 1) / sentCnt
        resultRep[f"morph_sentNgram3_N{i}"] = sum(1 for n in ngramCnt.values() if n > 2) / sentCnt
        if sentCnt == 1:
            resultRep[f"morph_sentNgram2Std_N{i}"] = 0
            resultRep[f"morph_sentNgram3Std_N{i}"] = 0
        else:
            resultRep[f"morph_sentNgram2Std_N{i}"] = math.sqrt(
                sum((n - resultRep[f"morph_sentNgram2_N{i}"]) ** 2 for n in ngramCnt.values()) / (sentCnt - 1)
            )
            resultRep[f"morph_sentNgram3Std_N{i}"] = math.sqrt(
                sum((n - resultRep[f"morph_sentNgram3_N{i}"]) ** 2 for n in ngramCnt.values()) / (sentCnt - 1)
            )

        # paragraph Ngram
        resultRep[f"morph_paraNgram2_N{i}"] = sum(1 for n in ngramCnt.values() if n > 1) / paraCnt
        resultRep[f"morph_paraNgram3_N{i}"] = sum(1 for n in ngramCnt.values() if n > 2) / paraCnt
        if paraCnt == 1:
            resultRep[f"morph_paraNgram2Std_N{i}"] = 0
            resultRep[f"morph_paraNgram3Std_N{i}"] = 0
        else:
            resultRep[f"morph_paraNgram2Std_N{i}"] = math.sqrt(
                sum((n - resultRep[f"morph_paraNgram2_N{i}"]) ** 2 for n in ngramCnt.values()) / (paraCnt - 1)
            )
            resultRep[f"morph_paraNgram3Std_N{i}"] = math.sqrt(
                sum((n - resultRep[f"morph_paraNgram3_N{i}"]) ** 2 for n in ngramCnt.values()) / (paraCnt - 1)
            )

        # NGRAM LIST
        ngramList = ngram_list(words, i)
        resultRepList[f"word_NgramList_N{i}"] = ngramList

        letters = [letter for word in words for letter in word]
        ngramList_letters = ngram_list(letters, i)
        resultRepList[f"char_NgramList_N{i}"] = ngramList_letters

        # Adjacent Ngram
        if sentCnt > 1:
            resultRep[f"morph_sentNgramOverlap2_N{i}"] = ngram_overlap(kkma_by_sent, 2, i) / len(kkma_by_sent) - 1
            resultRep[f"morph_sentNgramOverlap2Avg_N{i}"] = ngram_overlap(kkma_by_sent, 2, i) / len(kkma)

        if sentCnt > 2:
            resultRep[f"morph_sentNgramOverlap3_N{i}"] = ngram_overlap(kkma_by_sent, 3, i) / len(kkma_by_sent) - 2
            resultRep[f"morph_sentNgramOverlap3Avg_N{i}"] = ngram_overlap(kkma_by_sent, 3, i) / len(kkma)

    return result, resultRep, resultRepList


def counter(text, sentences, words, kkma, kkma_list, kkma_simple, kkma_by_sent):
    finalresult = collections.defaultdict()
    result = collections.defaultdict()
    resultLst = collections.defaultdict()

    # counters ---------------------------------------------------------------------------
    charCnt = 0
    paraCnt = 0

    # morphs list ------------------------------------------------------------------------
    # (morph, tag, including sentence)
    morphLst_C = []
    morphLst_N = []
    morphLst_NN = []
    morphLst_NNG = []
    morphLst_NNP = []
    morphLst_NNB = []
    morphLst_NP = []
    morphLst_NP_people = []
    morphLst_NP_things = []
    morphLst_NR = []
    morphLst_V = []
    morphLst_VV = []
    morphLst_VA = []
    morphLst_VX = []
    morphLst_VCP = []
    morphLst_VCN = []
    morphLst_M = []
    morphLst_MM = []
    morphLst_MA = []
    morphLst_F = []
    morphLst_J = []
    morphLst_E = []
    morphLst_X = []
    morphLst_IC = []

    # morphs -----------------------------------------------------------------------------
    morphs_C = [ # 내용
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
    morphs_N = ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]  # 체언
    morphs_NN = ["NNG", "NNP", "NNB", "NNBC"]  # 명사
    morphs_NP_people = [
        "나",
        "저",
        "우리",
        "저희",
        "너",
        "너희",
        "당신",
        "그대",
        "그",
        "그들",
        "이들",
        "저들",
        "자기",
        "저",
        "당신",
        "누구",
        "아무",
    ]
    morphs_NP_things = [
        "이",
        "이것",
        "여기",
        "이리",
        "저",
        "저것",
        "저기",
        "저리",
        "그",
        "그것",
        "거기",
        "그곳",
        "그리",
        "무엇",
        "어디",
        "아무것",
        "아무데",
    ]
    morphs_V = ["VV", "VA", "VX", "VCP", "VCN"]
    morphs_M = ["MM", "MMA", "MMD", "MMN", "MAG", "MAJ"]
    morphs_MA = ["MAG", "MAJ"]
    morphs_F = [
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JC",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
    ]
    morphs_J = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JX", "JC"]
    morphs_E = ["EP", "EF", "EC", "ETN", "ETM"]
    morphs_X = ["XPN", "XSN", "XSV", "XSA"]

    # counting ---------------------------------------------------------------------------
    for i in range(len(text)):
        if "\uAC00" <= text[i] <= "\uD7AF":
            charCnt += 1
        if text[i] == "\n" and i != 0:
            if text[i - 1] == "\n":
                continue
            paraCnt += 1
        if i == len(text) - 1:
            paraCnt += 1

    for i, pos_list in enumerate(kkma_list):
        for pos in pos_list:
            morph, tag = pos[0], pos[1]
            if tag in morphs_C:
                morphLst_C.append((len(morphLst_C), morph, tag, sentences[i]))
                if tag in morphs_N:
                    morphLst_N.append((len(morphLst_N), morph, tag, sentences[i]))
                    if tag in morphs_NN:
                        morphLst_NN.append((len(morphLst_NN), morph, tag, sentences[i]))
                        if tag == "NNG":
                            morphLst_NNG.append((len(morphLst_NNG), morph, tag, sentences[i]))
                        elif tag == "NNP":
                            morphLst_NNP.append((len(morphLst_NNP), morph, tag, sentences[i]))
                        elif tag in ["NNB", "NNBC"]:
                            morphLst_NNB.append((len(morphLst_NNB), morph, tag, sentences[i]))
                    elif tag == "NP":
                        morphLst_NP.append((len(morphLst_NP), morph, tag, sentences[i]))
                        if morph in morphs_NP_people:
                            morphLst_NP_people.append((len(morphLst_NP_people), morph, tag, sentences[i]))
                        elif morph in morphs_NP_things:
                            morphLst_NP_things.append((len(morphLst_NP_things), morph, tag, sentences[i]))
                    elif tag == "NR":
                        morphLst_NR.append((len(morphLst_NR), morph, tag, sentences[i]))
                elif tag == "IC":
                    morphLst_IC.append((len(morphLst_IC), morph, tag, sentences[i]))
                elif tag in morphs_V:
                    morphLst_V.append((len(morphLst_V), morph, tag, sentences[i]))
                    if tag == "VV":
                        morphLst_VV.append((len(morphLst_VV), morph, tag, sentences[i]))
                    elif tag == "VA":
                        morphLst_VA.append((len(morphLst_VA), morph, tag, sentences[i]))
                    elif tag == "VX":
                        morphLst_VX.append((len(morphLst_VX), morph, tag, sentences[i]))
                    elif tag == "VCP":
                        morphLst_VCP.append((len(morphLst_VCP), morph, tag, sentences[i]))
                    elif tag == "VCN":
                        morphLst_VCN.append((len(morphLst_VCN), morph, tag, sentences[i]))
                elif tag in morphs_M:
                    morphLst_M.append((len(morphLst_M), morph, tag, sentences[i]))
                    if tag.startswith("MM"):
                        morphLst_MM.append((len(morphLst_MM), morph, tag, sentences[i]))
                    elif tag in morphs_MA:
                        morphLst_MA.append((len(morphLst_MA), morph, tag, sentences[i]))
            elif tag in morphs_F:
                morphLst_F.append((len(morphLst_F), morph, tag, sentences[i]))
                if tag in morphs_J:
                    morphLst_J.append((len(morphLst_J), morph, tag, sentences[i]))
                elif tag in morphs_E:
                    morphLst_E.append((len(morphLst_E), morph, tag, sentences[i]))
                elif tag in morphs_X:
                    morphLst_X.append((len(morphLst_X), morph, tag, sentences[i]))

    # cnt result --------------------------------------------------------------------------
    result["para_Cnt"] = paraCnt
    result["sentence_Cnt"] = len(sentences)
    result["word_Cnt"] = len(words)
    result["morph_Cnt"] = len(kkma)
    result["char_Cnt"] = charCnt

    result["C_Cnt"] = len(morphLst_C)
    result["N_Cnt"] = len(morphLst_N)
    result["NN_Cnt"] = len(morphLst_NN)
    result["NNG_Cnt"] = len(morphLst_NNG)
    result["NNP_Cnt"] = len(morphLst_NNP)
    result["NNB_Cnt"] = len(morphLst_NNB)
    result["NP_Cnt"] = len(morphLst_NP)
    result["NPP_Cnt"] = len(morphLst_NP_people)
    result["NPT_Cnt"] = len(morphLst_NP_things)
    result["NR_Cnt"] = len(morphLst_NR)

    result["V_Cnt"] = len(morphLst_V)
    result["VV_Cnt"] = len(morphLst_VV)
    result["VA_Cnt"] = len(morphLst_VA)
    result["VX_Cnt"] = len(morphLst_VX)
    result["VCP_Cnt"] = len(morphLst_VCP)
    result["VCN_Cnt"] = len(morphLst_VCN)

    result["M_Cnt"] = len(morphLst_M)
    result["MA_Cnt"] = len(morphLst_MA)
    result["MM_Cnt"] = len(morphLst_MM)

    result["F_Cnt"] = len(morphLst_F)
    result["J_Cnt"] = len(morphLst_J)
    result["E_Cnt"] = len(morphLst_E)
    result["X_Cnt"] = len(morphLst_X)
    result["IC_Cnt"] = len(morphLst_IC)

    # Sentence Level -----------------------------------------------------------------------
    resultSentComp, resultSentRep, resultSentRepList = sentence_level(sentences, words, kkma, kkma_by_sent, paraCnt)

    # Number of Different Words ------------------------------------------------------------
    resultNDW = collections.defaultdict()

    # resultNDW["word_NDW"] = len(set(words))
    resultNDW["morph_NDW"] = len(set(kkma))
    
    # Ngram NDW 추가 부분 (25.5.20(수정))
    for n in range(2, 9):
        ngram_list_str = ngram(kkma, n)            
        ngram_list_tuple = [tuple(ng.split()) for ng in ngram_list_str]
        resultNDW[f"{n}-gram_NDW"] = len(set(ngram_list_tuple))
    logging.debug("=====================디버깅용=====================\n%s", resultNDW[f"{n}-gram_NDW"])
    

    resultNDW["N_NDW"] = len(set(morphLst_N))
    resultNDW["NN_NDW"] = len(set(morphLst_NN))
    resultNDW["NNG_NDW"] = len(set(morphLst_NNG))
    resultNDW["NNP_NDW"] = len(set(morphLst_NNP))
    resultNDW["NNB_NDW"] = len(set(morphLst_NNB))
    resultNDW["NP_NDW"] = len(set(morphLst_NP))
    resultNDW["NR_NDW"] = len(set(morphLst_NR))

    resultNDW["V_NDW"] = len(set(morphLst_V))
    resultNDW["VV_NDW"] = len(set(morphLst_VV))
    resultNDW["VA_NDW"] = len(set(morphLst_VA))

    resultNDW["M_NDW"] = len(set(morphLst_M))
    resultNDW["MM_NDW"] = len(set(morphLst_MM))

    resultNDW["IC_NDW"] = len(set(morphLst_IC))
    resultNDW["J_NDW"] = len(set(morphLst_J))
    resultNDW["E_NDW"] = len(set(morphLst_E))
    resultNDW["X_NDW"] = len(set(morphLst_X))
    resultNDW["C_NDW"] = len(set(morphLst_C))
    resultNDW["F_NDW"] = len(set(morphLst_F))

    # density result -----------------------------------------------------------------------
    resultDensity = collections.defaultdict()

    resultDensity["C_Den"] = len(morphLst_C) / len(kkma)
    resultDensity["F_Den"] = len(morphLst_F) / len(kkma)
    resultDensity["N_Den"] = len(morphLst_N) / len(kkma)
    resultDensity["NN_Den"] = len(morphLst_NN) / len(kkma)
    resultDensity["NNG_Den"] = len(morphLst_NNG) / len(kkma)
    resultDensity["NNP_Den"] = len(morphLst_NNP) / len(kkma)
    resultDensity["NNB_Den"] = len(morphLst_NNB) / len(kkma)
    resultDensity["NP_Den"] = len(morphLst_NP) / len(kkma)
    resultDensity["NM_Den"] = len(morphLst_NR) / len(kkma)
    resultDensity["V_Den"] = len(morphLst_V) / len(kkma)
    resultDensity["VV_Den"] = len(morphLst_VV) / len(kkma)
    resultDensity["VA_Den"] = len(morphLst_VA) / len(kkma)
    resultDensity["M_Den"] = len(morphLst_M) / len(kkma)
    resultDensity["MM_Den"] = len(morphLst_MM) / len(kkma)
    resultDensity["MA_Den"] = len(morphLst_MA) / len(kkma)
    resultDensity["J_Den"] = len(morphLst_J) / len(kkma)
    resultDensity["E_Den"] = len(morphLst_E) / len(kkma)
    resultDensity["X_Den"] = len(morphLst_X) / len(kkma)
    resultDensity["IT_Den"] = len(morphLst_IC) / len(kkma)

    if len(morphLst_C) > 0:
        resultDensity["NC_Den"] = len(morphLst_N) / len(morphLst_C)
        resultDensity["NNC_Den"] = len(morphLst_NN) / len(morphLst_C)
        resultDensity["NNGC_Den"] = len(morphLst_NNP) / len(morphLst_C)
        resultDensity["NNBC_Den"] = len(morphLst_NNB) / len(morphLst_C)
        resultDensity["NPC_Den"] = len(morphLst_NP) / len(morphLst_C)
        resultDensity["NNPC_Den"] = len(morphLst_NNP) / len(morphLst_C)
        resultDensity["NMC_Den"] = len(morphLst_NR) / len(morphLst_C)
        resultDensity["VC_Den"] = len(morphLst_V) / len(morphLst_C)
        resultDensity["VVC_Den"] = len(morphLst_VV) / len(morphLst_C)
        resultDensity["VAC_Den"] = len(morphLst_VA) / len(morphLst_C)
        resultDensity["MC_Den"] = len(morphLst_M) / len(morphLst_C)
        resultDensity["MMC_Den"] = len(morphLst_MM) / len(morphLst_C)
        resultDensity["MAC_Den"] = len(morphLst_MA) / len(morphLst_C)
        resultDensity["INC_Den"] = len(morphLst_IC) / len(morphLst_C)

    if len(morphLst_F) > 0:
        resultDensity["JF_Den"] = len(morphLst_J) / len(morphLst_F)
        resultDensity["EF_Den"] = len(morphLst_E) / len(morphLst_F)
        resultDensity["XF_Den"] = len(morphLst_X) / len(morphLst_F)

    # result list ------------------------------------------------------------------------
    resultLst["N_Lst"] = morphLst_N
    resultLst["NN_Lst"] = morphLst_NN
    resultLst["NNG_Lst"] = morphLst_NNG
    resultLst["NNP_Lst"] = morphLst_NNP
    resultLst["NNB_Lst"] = morphLst_NNB
    resultLst["NP_Lst"] = morphLst_NP
    resultLst["NPP_Lst"] = morphLst_NP_people
    resultLst["NPT_Lst"] = morphLst_NP_things
    resultLst["NR_Lst"] = morphLst_NR
    resultLst["V_Lst"] = morphLst_V
    resultLst["VV_Lst"] = morphLst_VV
    resultLst["VA_Lst"] = morphLst_VA
    resultLst["VX_Lst"] = morphLst_VX
    resultLst["VCP_Lst"] = morphLst_VCP
    resultLst["VCN_Lst"] = morphLst_VCN
    resultLst["M_Lst"] = morphLst_M
    resultLst["MA_Lst"] = morphLst_MA
    resultLst["MM_Lst"] = morphLst_MM
    resultLst["J_Lst"] = morphLst_J
    resultLst["E_Lst"] = morphLst_E
    resultLst["X_Lst"] = morphLst_X
    resultLst["IC_Lst"] = morphLst_IC

    resultLvl = collections.defaultdict()

    (
        resultLvl["morph_LenAvg"],
        resultLvl["morph_LenStd"],
        resultLvl["word_LenAvg"],
        resultLvl["word_LenStd"],
    ) = avgLen(words, kkma_simple)

    # final result ------------------------------------------------------------------------
    finalresult["basic_count"] = result
    finalresult["NDW"] = resultNDW
    finalresult["basic_density"] = resultDensity
    finalresult["basic_level"] = resultLvl
    finalresult["basic_list"] = resultLst
    
    finalresult["sentenceLvl"] = resultSentComp
    finalresult["sentenceLvlRep"] = resultSentRep
    finalresult["sentenceLvlRep_list"] = resultSentRepList

    logging.debug("=====================디버깅용=====================\n%s", finalresult)
    return finalresult
