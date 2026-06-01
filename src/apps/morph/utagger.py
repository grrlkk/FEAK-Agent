import apps.morph.bareun as br
import pandas as pd

# import pyutagger.downloader as ud
import pyutagger.utagger as ut
from pathlib import Path

# ud.install_utagger("utagger3")  # 유태거 3
# ud.install_utagger("utagger4")  # 유태거 4
# ud.install_utagger("utagger4hj")  # 유태거 4 훈민정음(옛한글 전용)


class utagger:
    def __init__(self):
        self.utg4 = ut.utagger_loader("utagger4")
        self.utg4.load()
        if not self.utg4:
            print("로드 실패")
            print("failed to load")

        WORD_GRADES = Path(__file__).parent / "word_grades.csv"
        self.word_grades = pd.read_csv(WORD_GRADES)
        # ['grade', 'vocab', 'homonym_num', 'pos', 'type', 'original', 'meaning', 'field']

    def morphs(self, text):
        return self.utg4.morphs(text)

    def nouns(self, text):
        return self.utg4.nouns(text)

    def pos(self, text):
        try:
            pos = self.utg4.pos(text)
        except:
            br_inf = br.bareun()
            pos = br_inf.pos(text)
        return pos

    def grade(self, text=None, pos=None):
        if pos is None:
            pos = self.pos(text)
        grades = []
        for word in pos:
            split_word = word[0].split("__")
            voc = split_word[0]
            hyn = split_word[1] if len(split_word) > 1 else 0

            if word[1].startswith("S") or word[1].startswith("E"):
                continue

            row = self.word_grades[
                (self.word_grades["vocab"] == voc)
                & (self.word_grades["homonym_num"] == int(hyn))
            ]
            if row.empty:
                row = self.word_grades[
                    (self.word_grades["vocab"] == voc + "다")
                    & (self.word_grades["homonym_num"] == int(hyn))
                ]

            if row.empty:
                grades.append(
                    {
                        "voc": word[0],
                        "pos_tagged": word[1],
                        "pos": "",
                        "type": "",
                        "meaning": "Not Found in Dictionary",
                        "field": "",
                        "grade": -1,
                    }
                )
                continue

            if str(row["pos_tag"].values[0]) not in word[1]:
                grades.append(
                    {
                        "voc": word[0],
                        "pos_tagged": word[1],
                        "pos": "",
                        "type": "",
                        "meaning": "Morpheme POS MisMatch",
                        "field": "",
                        "grade": -1,
                    }
                )
                continue

            grade = row["grade"].values[0]
            grade = int(grade.split("등급")[0])
            grades.append(
                {
                    "voc": word[0],
                    "grade": grade,
                    "pos_tagged": word[1],
                    "pos": row["pos"].values[0],
                    "type": row["type"].values[0],
                    "meaning": row["meaning"].values[0],
                    "field": row["field"].values[0],
                }
            )
        return grades
