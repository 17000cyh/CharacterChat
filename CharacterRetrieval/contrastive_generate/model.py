from transformers import RobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLearningModel(nn.Module):
    def __init__(self, pretrained_model_name='roberta-base'):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用池化后的输出作为特征表示
        return pooled_output

def info_nce_loss(features_query, features_positive, features_negatives, temperature=0.1):
    """
    InfoNCE Loss计算。
    features_query: 查询的特征表示。
    features_positive: 正例的特征表示。
    features_negatives: 负例的特征表示列表。
    temperature: 温度参数。
    """
    # 计算查询与正例的相似度
    positives_similarity = F.cosine_similarity(features_query, features_positive)
    positives_similarity = positives_similarity.unsqueeze(1)  # 调整形状以便后续操作

    # 计算查询与所有负例的相似度
    negatives_similarity = torch.stack([F.cosine_similarity(features_query, neg) for neg in features_negatives], dim=1)

    # 组合正例和负例相似度
    all_similarities = torch.cat([positives_similarity, negatives_similarity], dim=1)
    
    # 应用softmax并计算loss
    labels = torch.zeros(all_similarities.size(0), dtype=torch.long).to(features_query.device)
    loss = F.cross_entropy(all_similarities / temperature, labels)
    
    return loss
