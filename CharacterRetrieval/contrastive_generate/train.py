import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer
from tqdm import tqdm
import numpy as np
import os
from data_align import data_align, Node
from dataloader import QueryMaterialDataset, contrastive_collate_fn_with_multiple_negatives
from model import ContrastiveLearningModel, info_nce_loss
import torch.nn.functional as F



# 检查是否可以使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据和模型初始化
tokenizer = RobertaTokenizer.from_pretrained('roberta')

question_id_list = [2,3,4,5,8,9,10,11]

test_question_id_list = [0,1]

data_dict = {}
for question_id in question_id_list:
    plot_pkl_path = f"../data/reflection_result/plot_{question_id}.pkl"
    question_pkl_path = f"../data/reflection_result/plot_{question_id}_question.pkl"
    
    temp_dict = data_align(plot_pkl_path, question_pkl_path)
    data_dict.update(temp_dict)

dataset = QueryMaterialDataset(data_dict, tokenizer)

# 分割数据集为训练集和测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])

# 构建测试数据集
data_dict = {}
for question_id in test_question_id_list:
    plot_pkl_path = f"../data/reflection_result/plot_{question_id}.pkl"
    question_pkl_path = f"../data/reflection_result/plot_{question_id}_question.pkl"
    
    temp_dict = data_align(plot_pkl_path, question_pkl_path)
    data_dict.update(temp_dict)
test_dataset = QueryMaterialDataset(data_dict, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=contrastive_collate_fn_with_multiple_negatives)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=contrastive_collate_fn_with_multiple_negatives,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=contrastive_collate_fn_with_multiple_negatives, drop_last=True)



def generate_ranked_lists(model, test_loader, device):
    ranked_lists = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            queries, positives, negatives_list = batch
            batch_size = queries['input_ids'].size(0)
            
            # 处理查询
            query_input_ids = queries['input_ids'].squeeze(1).to(device)
            query_attention_mask = queries['attention_mask'].squeeze(1).to(device)
            query_features = model(query_input_ids, attention_mask=query_attention_mask)
            
            # 初始化相似度矩阵
            similarities = []
            
            # 处理正例
            positive_input_ids = positives['input_ids'].squeeze(1).to(device)
            positive_attention_mask = positives['attention_mask'].squeeze(1).to(device)
            positive_features = model(positive_input_ids, attention_mask=positive_attention_mask)
            positive_similarities = F.cosine_similarity(query_features, positive_features)
            similarities.append(positive_similarities.unsqueeze(0))
            
            # 处理负例
            for negatives in negatives_list:
                negative_input_ids = negatives['input_ids'].to(device)
                negative_attention_mask = negatives['attention_mask'].to(device)
                negative_features = model(negative_input_ids.squeeze(1), attention_mask=negative_attention_mask.squeeze(1))
                negative_similarities = F.cosine_similarity(query_features, negative_features)
                similarities.append(negative_similarities.unsqueeze(0))
            
            # 计算最终相似度分数并排序
            similarities = torch.cat(similarities, dim=0)
            _, sorted_indices = torch.sort(similarities, descending=True, dim=0)
            
            # 生成排名列表
            for i in range(batch_size):
                ranked_list = sorted_indices[:, i].tolist()
                # 将排名转换为"positive"/"negative"标签
                ranked_list = ['positive' if idx == 0 else f'negative_{idx}' for idx in ranked_list]
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


print("Data and model initialized")
# 打印数据集长度信息
print(f"Train Dataset: {len(train_dataset)}")
print(f"Valid Dataset: {len(valid_dataset)}")
print(f"Test Dataset: {len(test_dataset)}")

model = ContrastiveLearningModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练过程
num_epochs = 100
best_loss = np.inf
patience, trials = 10, 0

print("begin to train ...")

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            queries, positives, negatives = batch
            query_input_ids, query_attention_mask = queries['input_ids'].squeeze(1).to(device), queries['attention_mask'].squeeze(1).to(device)
            positive_input_ids, positive_attention_mask = positives['input_ids'].squeeze(1).to(device), positives['attention_mask'].squeeze(1).to(device)
            
            query_features = model(query_input_ids, query_attention_mask)
            positive_features = model(positive_input_ids, positive_attention_mask)
            
            negative_features_list = []
            for negative in negatives:
                neg_input_ids, neg_attention_mask = negative['input_ids'].to(device), negative['attention_mask'].to(device)
                neg_features = model(neg_input_ids.squeeze(1), neg_attention_mask.squeeze(1))
                negative_features_list.append(neg_features)
            
            loss = info_nce_loss(query_features, positive_features, negative_features_list)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        queries, positives, negatives_batch = batch
        # 处理queries和positives
        queries_input_ids, queries_attention_mask = queries['input_ids'].squeeze(1).to(device), queries['attention_mask'].squeeze(1).to(device)
        positives_input_ids, positives_attention_mask = positives['input_ids'].squeeze(1).to(device), positives['attention_mask'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        query_features = model(queries_input_ids, queries_attention_mask)
        positive_features = model(positives_input_ids, positives_attention_mask)
        
        # 处理负例
        negative_features = []
        for negatives in negatives_batch:
            neg_input_ids, neg_attention_mask = negatives['input_ids'].squeeze(1).to(device), negatives['attention_mask'].squeeze(1).to(device)
            neg_features = model(neg_input_ids, neg_attention_mask)
            negative_features.append(neg_features)
        
        loss = info_nce_loss(query_features, positive_features, negative_features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    # 开始在验证集上进行测试
    val_loss = evaluate(model, valid_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # 开始进行早停机制
    if val_loss < best_loss:
        trials = 0
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping")
            break

    print("begin to test on test data ")
    print("=" * 60)
    ranked_lists = generate_ranked_lists(model, test_loader, device)
    hit_at_1 = calculate_hit_at_k(ranked_lists, k=1)
    mrr = calculate_mrr(ranked_lists)

    print(f"Hit@1: {hit_at_1:.4f}, MRR: {mrr:.4f}")




# 加载最佳模型并在测试集上评估
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
total_loss = 0

# 假设我们有一个函数evaluate_model返回每个查询的ranked_lists
ranked_lists = generate_ranked_lists(model, test_loader, device)
hit_at_1 = calculate_hit_at_k(ranked_lists, k=1)
mrr = calculate_mrr(ranked_lists)

print(f"Hit@1: {hit_at_1:.4f}, MRR: {mrr:.4f}")
