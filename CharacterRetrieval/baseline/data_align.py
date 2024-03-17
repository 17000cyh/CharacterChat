import pickle 
import re


class Node:
    def __init__(self, node_type, node_id, level, content, children=None, parent=None):
        self.node_type = node_type  # 'event' or 'insight'
        self.node_id = node_id
        self.level = level
        self.content = content
        self.children = children if children is not None else []
        self.parent = parent

    def __str__(self):
        return f"Node(ID: {self.node_id}, Type: {self.node_type}, Level: {self.level}, Content: {self.content})"

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


def data_align(plot_pkl_path, questoin_pkl_path):
    plot = pickle.load(open(plot_pkl_path,"rb"))
    question_dict = pickle.load(open(questoin_pkl_path,"rb"))

    query_material_pair_dict = {}

    node_id_context_dict = {}


    def find_next_node(node_sets):
        next_node_set = []
        for node in node_sets:
            next_node_set += node.children
            node_id = node.node_id
            node_content = node.content
            if node_content.strip("\n") == "":
                continue
            node_id_context_dict[node_id] = node_content
            
        if len(next_node_set) == 0:
            return
        find_next_node(next_node_set)
    
    find_next_node(set([plot]))
    
    
    for key in question_dict.keys():
        questions = question_dict[key]
        for question in questions:
            try:
                if len(node_id_context_dict[key]) < 30:
                    continue
            except:
                continue
            question = re.sub(r"^\d+\.\s?", "", question)
            query_material_pair_dict[question] = node_id_context_dict[key]

    print(f"5 Examples of query_material_pair_dict: {list(query_material_pair_dict.items())[:5]}")
    return query_material_pair_dict
