import collections
import math
import logging
from . import counter
from lexical_diversity import lex_div as ld
from lexicalrichness import LexicalRichness as lr
from .counter import ngram


morphs_CL = [
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
	"MAG",
	"MAJ",
	"IC",
	"XR",
]
morphs_NL = ["NNG", "NNP", "NNB", "NNBC", "NP", "NR"]  # 체언
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
morphs_VL = ["VV", "VA", "VX", "VCP", "VCN"]
morphs_ML = ["MM", "MAG", "MAJ"]
morphs_MA = ["MAG", "MAJ"]
morphs_FL = [
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


def safe_divide(numerator, denominator):
	if denominator == 0 or denominator == 0.0:
		index = 0
	else:
		index = numerator / denominator
	return index


def cttr(text):
	ntokens = len(text)
	ntypes = len(set(text))

	return safe_divide(ntypes, math.sqrt(2 * ntokens))


class TTR:
	def __init__(self, kkma):
		self.kkma = [morp for morp in kkma if not morp[1].startswith("S")]
		self.TTR = collections.defaultdict()
		self.TTR["lemma_Cnt"] = len(self.kkma)

		self.lemma_lst = []
		self.C_lst = []
		self.N_lst = []
		self.NN_lst = []
		self.NNG_lst = []
		self.NNP_lst = []
		self.NNB_lst = []
		self.NP_lst = []
		self.NR_lst = []
		self.V_lst = []
		self.VV_lst = []
		self.VA_lst = []
		self.M_lst = []
		self.MM_lst = []
		self.MA_lst = []
		self.MAJ_lst = []
		self.IC_lst = []
		self.F_lst = []
		self.J_lst = []
		self.E_lst = []
		self.X_lst = []

		for morp in self.kkma:
			self.lemma_lst.append(morp[0])
			if morp[1] in morphs_CL:
				self.C_lst.append(morp[0])
			if morp[1] in morphs_NL:
				self.N_lst.append(morp[0])
			if morp[1] in morphs_NN:
				self.NN_lst.append(morp[0])
			if morp[1] == "NNG":
				self.NNG_lst.append(morp[0])
			if morp[1] == "NNP":
				self.NNP_lst.append(morp[0])
			if morp[1] == "NNB":
				self.NNB_lst.append(morp[0])
			if morp[1] == "NP":
				self.NP_lst.append(morp[0])
			if morp[1] == "NR":
				self.NR_lst.append(morp[0])
			if morp[1] in morphs_VL:
				self.V_lst.append(morp[0])
			if morp[1] == "VV":
				self.VV_lst.append(morp[0])
			if morp[1] == "VA":
				self.VA_lst.append(morp[0])
			if morp[1] in morphs_ML:
				self.M_lst.append(morp[0])
			if morp[1] == "MM":
				self.MM_lst.append(morp[0])
			if morp[1] in morphs_MA:
				self.MA_lst.append(morp[0])
			if morp[1] == "MAJ":
				self.MAJ_lst.append(morp[0])
			if morp[1] == "IC":
				self.IC_lst.append(morp[0])
			if morp[1] in morphs_FL:
				self.F_lst.append(morp[0])
			if morp[1] in morphs_J:
				self.J_lst.append(morp[0])
			if morp[1] in morphs_E:
				self.E_lst.append(morp[0])
			if morp[1] in morphs_X:
				self.X_lst.append(morp[0])

		self.cal_TTR()
		self.cal_RTTR()
		self.cal_CTTR()
		self.cal_MSTTR()
		self.cal_MATTR()
		self.cal_MTLD()
		self.cal_VOCDD()
		self.cal_HDD()

		self.cal_ngram_TTRs()


	def cal_ngram_TTRs(self):
			total_ngram_token_count = sum(len(ngram(self.kkma, n)) for n in range(2, 9))
			for n in range(2, 9):
				ngram_list = ngram(self.kkma, n)
				ngram_type_count = len(set(ngram_list))
				ngram_token_count = len(ngram_list)
				self.TTR[f"{n}-gram_TTR"] = ngram_type_count / total_ngram_token_count if total_ngram_token_count > 0 else 0
				self.TTR[f"{n}-gram_RTTR"] = ngram_type_count / (total_ngram_token_count ** 0.5) if total_ngram_token_count > 0 else 0
				self.TTR[f"{n}-gram_CTTR"] = ngram_type_count / ((2 * total_ngram_token_count) ** 0.5) if total_ngram_token_count > 0 else 0


	def cal_TTR(self):
		self.TTR["lemma_TTR"] = ld.ttr(self.lemma_lst)
		self.TTR["C_TTR"] = ld.ttr(self.C_lst)
		self.TTR["N_TTR"] = ld.ttr(self.N_lst)
		self.TTR["NN_TTR"] = ld.ttr(self.NN_lst)
		self.TTR["NNG_TTR"] = ld.ttr(self.NNG_lst)
		self.TTR["NNP_TTR"] = ld.ttr(self.NNP_lst)
		self.TTR["NNB_TTR"] = ld.ttr(self.NNB_lst)
		self.TTR["NP_TTR"] = ld.ttr(self.NP_lst)
		self.TTR["NR_TTR"] = ld.ttr(self.NR_lst)
		self.TTR["V_TTR"] = ld.ttr(self.V_lst)
		self.TTR["VV_TTR"] = ld.ttr(self.VV_lst)
		self.TTR["VA_TTR"] = ld.ttr(self.VA_lst)
		self.TTR["M_TTR"] = ld.ttr(self.M_lst)
		self.TTR["MM_TTR"] = ld.ttr(self.MM_lst)
		self.TTR["MA_TTR"] = ld.ttr(self.MA_lst)
		self.TTR["MAJ_TTR"] = ld.ttr(self.MAJ_lst)
		self.TTR["IC_TTR"] = ld.ttr(self.IC_lst)
		self.TTR["F_TTR"] = ld.ttr(self.F_lst)
		self.TTR["J_TTR"] = ld.ttr(self.J_lst)
		self.TTR["E_TTR"] = ld.ttr(self.E_lst)
		self.TTR["X_TTR"] = ld.ttr(self.X_lst)

	def cal_RTTR(self):
		self.TTR["lemma_RTTR"] = ld.root_ttr(self.lemma_lst)
		self.TTR["C_RTTR"] = ld.root_ttr(self.C_lst)
		self.TTR["N_RTTR"] = ld.root_ttr(self.N_lst)
		self.TTR["NN_RTTR"] = ld.root_ttr(self.NN_lst)
		self.TTR["NNG_RTTR"] = ld.root_ttr(self.NNG_lst)
		self.TTR["NNP_RTTR"] = ld.root_ttr(self.NNP_lst)
		self.TTR["NNB_RTTR"] = ld.root_ttr(self.NNB_lst)
		self.TTR["NP_RTTR"] = ld.root_ttr(self.NP_lst)
		self.TTR["NR_RTTR"] = ld.root_ttr(self.NR_lst)
		self.TTR["V_RTTR"] = ld.root_ttr(self.V_lst)
		self.TTR["VV_RTTR"] = ld.root_ttr(self.VV_lst)
		self.TTR["VA_RTTR"] = ld.root_ttr(self.VA_lst)
		self.TTR["M_RTTR"] = ld.root_ttr(self.M_lst)
		self.TTR["MM_RTTR"] = ld.root_ttr(self.MM_lst)
		self.TTR["MA_RTTR"] = ld.root_ttr(self.MA_lst)
		self.TTR["MAJ_RTTR"] = ld.root_ttr(self.MAJ_lst)
		self.TTR["IC_RTTR"] = ld.root_ttr(self.IC_lst)
		self.TTR["F_RTTR"] = ld.root_ttr(self.F_lst)
		self.TTR["J_RTTR"] = ld.root_ttr(self.J_lst)
		self.TTR["E_RTTR"] = ld.root_ttr(self.E_lst)
		self.TTR["X_RTTR"] = ld.root_ttr(self.X_lst)

	def cal_CTTR(self):
		self.TTR["lemma_CTTR"] = cttr(self.lemma_lst)
		self.TTR["C_CTTR"] = cttr(self.C_lst)
		self.TTR["N_CTTR"] = cttr(self.N_lst)
		self.TTR["NN_CTTR"] = cttr(self.NN_lst)
		self.TTR["NNG_CTTR"] = cttr(self.NNG_lst)
		self.TTR["NNP_CTTR"] = cttr(self.NNP_lst)
		self.TTR["NNB_CTTR"] = cttr(self.NNB_lst)
		self.TTR["NP_CTTR"] = cttr(self.NP_lst)
		self.TTR["NR_CTTR"] = cttr(self.NR_lst)
		self.TTR["V_CTTR"] = cttr(self.V_lst)
		self.TTR["VV_CTTR"] = cttr(self.VV_lst)
		self.TTR["VA_CTTR"] = cttr(self.VA_lst)
		self.TTR["M_CTTR"] = cttr(self.M_lst)
		self.TTR["MM_CTTR"] = cttr(self.MM_lst)
		self.TTR["MA_CTTR"] = cttr(self.MA_lst)
		self.TTR["MAJ_CTTR"] = cttr(self.MAJ_lst)
		self.TTR["IC_CTTR"] = cttr(self.IC_lst)
		self.TTR["F_CTTR"] = cttr(self.F_lst)
		self.TTR["J_CTTR"] = cttr(self.J_lst)
		self.TTR["E_CTTR"] = cttr(self.E_lst)
		self.TTR["X_CTTR"] = cttr(self.X_lst)

	def cal_MSTTR(self):
		self.TTR["lemma_MSTTR"] = ld.msttr(self.lemma_lst)
		self.TTR["C_MSTTR"] = ld.msttr(self.C_lst)
		self.TTR["N_MSTTR"] = ld.msttr(self.N_lst)
		self.TTR["NN_MSTTR"] = ld.msttr(self.NN_lst)
		self.TTR["NNG_MSTTR"] = ld.msttr(self.NNG_lst)
		self.TTR["NNP_MSTTR"] = ld.msttr(self.NNP_lst)
		self.TTR["NNB_MSTTR"] = ld.msttr(self.NNB_lst)
		self.TTR["NP_MSTTR"] = ld.msttr(self.NP_lst)
		self.TTR["NR_MSTTR"] = ld.msttr(self.NR_lst)
		self.TTR["V_MSTTR"] = ld.msttr(self.V_lst)
		self.TTR["VV_MSTTR"] = ld.msttr(self.VV_lst)
		self.TTR["VA_MSTTR"] = ld.msttr(self.VA_lst)
		self.TTR["M_MSTTR"] = ld.msttr(self.M_lst)
		self.TTR["MM_MSTTR"] = ld.msttr(self.MM_lst)
		self.TTR["MA_MSTTR"] = ld.msttr(self.MA_lst)
		self.TTR["MAJ_MSTTR"] = ld.msttr(self.MAJ_lst)
		self.TTR["IC_MSTTR"] = ld.msttr(self.IC_lst)
		self.TTR["F_MSTTR"] = ld.msttr(self.F_lst)
		self.TTR["J_MSTTR"] = ld.msttr(self.J_lst)
		self.TTR["E_MSTTR"] = ld.msttr(self.E_lst)
		self.TTR["X_MSTTR"] = ld.msttr(self.X_lst)

	def cal_MATTR(self):
		self.TTR["lemma_MATTR"] = ld.mattr(self.lemma_lst)
		self.TTR["C_MATTR"] = ld.mattr(self.C_lst)
		self.TTR["N_MATTR"] = ld.mattr(self.N_lst)
		self.TTR["NN_MATTR"] = ld.mattr(self.NN_lst)
		self.TTR["NNG_MATTR"] = ld.mattr(self.NNG_lst)
		self.TTR["NNP_MATTR"] = ld.mattr(self.NNP_lst)
		self.TTR["NNB_MATTR"] = ld.mattr(self.NNB_lst)
		self.TTR["NP_MATTR"] = ld.mattr(self.NP_lst)
		self.TTR["NR_MATTR"] = ld.mattr(self.NR_lst)
		self.TTR["V_MATTR"] = ld.mattr(self.V_lst)
		self.TTR["VV_MATTR"] = ld.mattr(self.VV_lst)
		self.TTR["VA_MATTR"] = ld.mattr(self.VA_lst)
		self.TTR["M_MATTR"] = ld.mattr(self.M_lst)
		self.TTR["MM_MATTR"] = ld.mattr(self.MM_lst)
		self.TTR["MA_MATTR"] = ld.mattr(self.MA_lst)
		self.TTR["MAJ_MATTR"] = ld.mattr(self.MAJ_lst)
		self.TTR["IC_MATTR"] = ld.mattr(self.IC_lst)
		self.TTR["F_MATTR"] = ld.mattr(self.F_lst)
		self.TTR["J_MATTR"] = ld.mattr(self.J_lst)
		self.TTR["E_MATTR"] = ld.mattr(self.E_lst)
		self.TTR["X_MATTR"] = ld.mattr(self.X_lst)

	def cal_MTLD(self):
		self.TTR["lemma_MTLD"] = ld.mtld(self.lemma_lst)
		self.TTR["C_MTLD"] = ld.mtld(self.C_lst)
		self.TTR["N_MTLD"] = ld.mtld(self.N_lst)
		self.TTR["NN_MTLD"] = ld.mtld(self.NN_lst)
		self.TTR["NNG_MTLD"] = ld.mtld(self.NNG_lst)
		self.TTR["NNP_MTLD"] = ld.mtld(self.NNP_lst)
		self.TTR["NNB_MTLD"] = ld.mtld(self.NNB_lst)
		self.TTR["NP_MTLD"] = ld.mtld(self.NP_lst)
		self.TTR["NR_MTLD"] = ld.mtld(self.NR_lst)
		self.TTR["V_MTLD"] = ld.mtld(self.V_lst)
		self.TTR["VV_MTLD"] = ld.mtld(self.VV_lst)
		self.TTR["VA_MTLD"] = ld.mtld(self.VA_lst)
		self.TTR["M_MTLD"] = ld.mtld(self.M_lst)
		self.TTR["MM_MTLD"] = ld.mtld(self.MM_lst)
		self.TTR["MA_MTLD"] = ld.mtld(self.MA_lst)
		self.TTR["MAJ_MTLD"] = ld.mtld(self.MAJ_lst)
		self.TTR["IC_MTLD"] = ld.mtld(self.IC_lst)
		self.TTR["F_MTLD"] = ld.mtld(self.F_lst)
		self.TTR["J_MTLD"] = ld.mtld(self.J_lst)
		self.TTR["E_MTLD"] = ld.mtld(self.E_lst)
		self.TTR["X_MTLD"] = ld.mtld(self.X_lst)

	def cal_VOCDD(self):
		lists = [
			("lemma_lst", "lemma_VOCDD"),
			("C_lst", "C_VOCDD"),
			("N_lst", "N_VOCDD"),
			("NN_lst", "NN_VOCDD"),
			("NNG_lst", "NNG_VOCDD"),
			("NNP_lst", "NNP_VOCDD"),
			("NNB_lst", "NNB_VOCDD"),
			("NP_lst", "NP_VOCDD"),
			("NR_lst", "NR_VOCDD"),
			("V_lst", "V_VOCDD"),
			("VV_lst", "VV_VOCDD"),
			("VA_lst", "VA_VOCDD"),
			("M_lst", "M_VOCDD"),
			("MM_lst", "MM_VOCDD"),
			("MA_lst", "MA_VOCDD"),
			("MAJ_lst", "MAJ_VOCDD"),
			("IC_lst", "IC_VOCDD"),
			("F_lst", "F_VOCDD"),
			("J_lst", "J_VOCDD"),
			("E_lst", "E_VOCDD"),
			("X_lst", "X_VOCDD"),
		]

		for lst_name, vocdd_name in lists:
			lex = lr(getattr(self, lst_name), preprocessor=None, tokenizer=None)
			try:
				self.TTR[vocdd_name] = lex.vocd()
			except:
				self.TTR[vocdd_name] = 0

	def cal_HDD(self):
		self.TTR["lemma_HDD"] = ld.hdd(self.lemma_lst)
		self.TTR["C_HDD"] = ld.hdd(self.C_lst)
		self.TTR["N_HDD"] = ld.hdd(self.N_lst)
		self.TTR["NN_HDD"] = ld.hdd(self.NN_lst)
		self.TTR["NNG_HDD"] = ld.hdd(self.NNG_lst)
		self.TTR["NNP_HDD"] = ld.hdd(self.NNP_lst)
		self.TTR["NNB_HDD"] = ld.hdd(self.NNB_lst)
		self.TTR["NP_HDD"] = ld.hdd(self.NP_lst)
		self.TTR["NR_HDD"] = ld.hdd(self.NR_lst)
		self.TTR["V_HDD"] = ld.hdd(self.V_lst)
		self.TTR["VV_HDD"] = ld.hdd(self.VV_lst)
		self.TTR["VA_HDD"] = ld.hdd(self.VA_lst)
		self.TTR["M_HDD"] = ld.hdd(self.M_lst)
		self.TTR["MM_HDD"] = ld.hdd(self.MM_lst)
		self.TTR["MA_HDD"] = ld.hdd(self.MA_lst)
		self.TTR["MAJ_HDD"] = ld.hdd(self.MAJ_lst)
		self.TTR["IC_HDD"] = ld.hdd(self.IC_lst)
		self.TTR["F_HDD"] = ld.hdd(self.F_lst)
		self.TTR["J_HDD"] = ld.hdd(self.J_lst)
		self.TTR["E_HDD"] = ld.hdd(self.E_lst)
		self.TTR["X_HDD"] = ld.hdd(self.X_lst)

	def get_TTR(self):
		return self.TTR
