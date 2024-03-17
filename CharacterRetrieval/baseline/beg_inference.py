from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
import pickle
import random
from data_align import data_align, Node
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 下载模型并初始化SentenceTransformer
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
model = SentenceTransformer(model_dir)

def encode_sentences(sentences, normalize=True):
    embeddings = model.encode(sentences, normalize_embeddings=normalize)
    return embeddings


def generate_inference_dict(data_dict, num_negatives=32):
    # 将所有材料（包括正例和可能的负例）收集到一个列表中以供随机抽样
    all_materials = list(data_dict.values())
    
    # 初始化新的数据结构以存储每个查询及其对应的正例和负例
    aligned_data = {}
    
    for query, positive_material in data_dict.items():
        # 为了确保负例不包括正例材料，从列表中移除当前的正例材料
        potential_negatives = [mat for mat in all_materials if mat != positive_material]
        
        # 随机抽取负例
        if len(potential_negatives) > num_negatives:
            negative_samples = random.sample(potential_negatives, num_negatives)
        else:
            # 如果可用的负例不足指定数量，就使用所有可能的负例
            negative_samples = potential_negatives
        
        # 将查询、正例和负例存储在新的数据结构中
        aligned_data[query] = {"positive": positive_material, "negatives": negative_samples}
    
    return aligned_data


def evaluate_model(data_dict):
    ranked_lists = []
    for query, materials in tqdm(data_dict.items()):
        positive_embedding = encode_sentences([materials["positive"]])
        negative_embeddings = encode_sentences(materials["negatives"])
        query_embedding = encode_sentences([query])
        
        # 计算相似度
        positive_score = cosine_similarity(query_embedding, positive_embedding)
        negative_scores = cosine_similarity(query_embedding, negative_embeddings)
        
        # 生成排名列表
        scores = np.append(positive_score, negative_scores)
        ranked_indices = np.argsort(-scores)  # 降序排列
        ranked_list = ["positive" if idx == 0 else f"negative_{idx-1}" for idx in ranked_indices]
        ranked_lists.append(ranked_list)
    return ranked_lists

def calculate_hit_at_k(ranked_lists, k=1):
    hit_count = 0
    for ranked_list in ranked_lists:
        if ranked_list.index("positive") < k:
            hit_count += 1
    return hit_count / len(ranked_lists)

def calculate_mrr(ranked_lists):
    reciprocal_ranks = []
    for ranked_list in ranked_lists:
        rank = ranked_list.index("positive") + 1
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


if __name__ == "__main__":
    plot_pkl_path = "../data/reflection_result/plot_0.pkl"
    question_pkl_path = "../data/reflection_result/plot_0_question.pkl"
    data_dict = generate_inference_dict(data_align(plot_pkl_path, question_pkl_path))

    ranked_lists = evaluate_model(data_dict)
    hit_at_k = calculate_hit_at_k(ranked_lists, k=1)
    mrr = calculate_mrr(ranked_lists)

    print(f"Hit@1: {hit_at_k:.4f}, MRR: {mrr:.4f}")
