from mecab import MeCab

mecab_kr = MeCab()


class mecab:
	def __init__(self):
		pass

	def morphs(self, text):
		return mecab_kr.morphs(text)

	def nouns(self, text):
		return mecab_kr.nouns(text)

	def pos(self, text, simple=False):
		parse = mecab_kr.parse(text)
		pos = []

		if simple:
			for r in parse:
				pos.append((r[1], r[2][0]))
			return pos

		for r in parse:
			if r[2][7] != None:
				temp = r[2][7].split('+')
				for t in temp:
					pos.append((t.split('/')[0], t.split('/')[1]))
			else:
				pos.append((r[1], r[2][0]))
		return pos

	def parse(self, text):
		return mecab_kr.parse(text)
