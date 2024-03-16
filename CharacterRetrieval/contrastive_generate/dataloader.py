from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data.dataloader import default_collate
from data_align import data_align, Node


class QueryMaterialDataset(Dataset):
    def __init__(self, data_dict, tokenizer):
        self.data_items = list(data_dict.items())
        self.tokenizer = tokenizer  # 接收tokenizer作为参数
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        query, positive_material = self.data_items[idx]
        # 使用tokenizer处理查询和正例材料
        tokenized_query = self.tokenizer(query, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        tokenized_positive = self.tokenizer(positive_material, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        return tokenized_query, tokenized_positive



def contrastive_collate_fn_with_multiple_negatives(batch, num_negatives=2):
    queries, materials = zip(*batch)
    batch_size = len(batch)
    all_materials = list(materials)
    
    negatives = []
    for _ in range(batch_size):
        selected_negatives = random.sample(all_materials, num_negatives)
        negatives.append(selected_negatives)

    
    batch_with_negatives = [(queries[i], materials[i], negatives[i]) for i in range(batch_size)]
    
    return default_collate(batch_with_negatives)


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    
    plot_pkl_path = "../data/reflection_result/plot_0.pkl"
    question_pkl_path = "../data/reflection_result/plot_0_question.pkl"
    data_dict = data_align(plot_pkl_path, question_pkl_path)

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = QueryMaterialDataset(data_dict, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=contrastive_collate_fn_with_multiple_negatives)
    for batch in dataloader:
        print(batch)
        break

