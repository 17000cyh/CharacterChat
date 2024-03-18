from transformers import RobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaTokenizer
from tqdm import tqdm
from typing import Dict, List
import numpy as np

# 检查是否可以使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContrastiveLearningModel(nn.Module):
    def __init__(self, pretrained_model_name='retrieval/roberta'):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用池化后的输出作为特征表示
        return pooled_output


# 数据和模型初始化
tokenizer = RobertaTokenizer.from_pretrained('retrieval/roberta')

model = ContrastiveLearningModel().to(device)

model.load_state_dict(
    torch.load("retrieval/best_model.pth")
)
model.eval()

def load_embeddings(material_list, model=model, tokenizer=tokenizer):
    # 加载所有材料的嵌入
    material_embeddings = {}
    for material in tqdm(material_list):
        input_ids = tokenizer(material, return_tensors="pt")['input_ids'][:,:500].to(device) # 进行截断处理
        attention_mask = tokenizer(material, return_tensors="pt")['attention_mask'][:,:500].to(device)
        with torch.no_grad():
            # print(f"material: {material}")
            # print(f"shape of input_ids: {input_ids.shape}")
            # print(f"shape of attention_mask: {attention_mask.shape}")
            material_embeddings[material] = model(input_ids, attention_mask=attention_mask).cpu().squeeze().numpy()
    return material_embeddings

def find_topk_similar_materials(query_embedding: np.ndarray, material_embeddings: Dict[str, np.ndarray], topk: int) -> List[str]:
    """
    根据query_embedding，在material_embeddings中找到最相似的topk个材料的名称。

    参数:
    - query_embedding: 查询的嵌入向量，numpy array。
    - material_embeddings: 材料的嵌入向量字典，{材料名称: 嵌入向量}。
    - topk: 返回相似度最高的topk个材料的数量。

    返回:
    - topk_similar_materials: 一个列表，包含与查询最相似的topk个材料的名称。
    """
    
    # 计算query_embedding与所有material_embeddings之间的相似度
    similarities = {material: np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding)) for material, embedding in material_embeddings.items()}
    
    # 根据相似度排序，取topk个
    topk_similar_materials = sorted(similarities, key=similarities.get, reverse=True)[:topk]
    
    return topk_similar_materials

def search_material(query, material_embeddings, topk=5):
    input_ids = tokenizer(query, return_tensors="pt")['input_ids'].to(device)
    attention_mask = tokenizer(query, return_tensors="pt")['attention_mask'].to(device)
    with torch.no_grad():
        query_embedding = model(input_ids, attention_mask=attention_mask).cpu().squeeze().numpy()
    topk_similar_materials = find_topk_similar_materials(query_embedding, material_embeddings, topk)
    return topk_similar_materials




if __name__ == "__main__":
    material_list = [
        "The cat is on the mat",
        "The dog is in the fog",
    ]
    material_embeddings = load_embeddings(material_list, model, tokenizer)

    print(
        search_material("The cat is in", material_embeddings, topk=1)
    )