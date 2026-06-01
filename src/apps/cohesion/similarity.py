import re
import torch
from sentence_transformers import SentenceTransformer, models, util
from keybert import KeyBERT
from transformers import BertModel


def model():
    embedding_model = models.Transformer(
        model_name_or_path="KDHyun08/TAACO_STS", max_seq_length=256, do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[embedding_model, pooling_model])

    return model


def similar(text, simil_model, kw_model, device="cuda"):
    torch.cuda.empty_cache()

    docs = re.split("\. |\? |\!", text)
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=10
    )

    document_embeddings = simil_model.encode(docs, convert_to_tensor=True)

    query = keywords[0]
    query_embedding = simil_model.encode(query, convert_to_tensor=True)

    top_k = len(docs)

    # 입력 문장 - 문장 후보군 간 코사인 유사도 계산 후,
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # 코사인 유사도 순으로 `top_k` 개 문장 추출
    top_results = torch.topk(cos_scores, k=top_k)

    # print(f"입력 문장: {query}")
    # print(f"\n<입력 문장과 유사한 {top_k} 개의 문장>\n")
    average = 0.0
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        average += score
    average /= len(docs)
    return average.item(), query[1]
