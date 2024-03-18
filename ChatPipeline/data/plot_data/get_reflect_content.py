import pickle 
import re
import json

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

def get_content(pkl_path,save_path):
    pkl_dict = pickle.load(open(pkl_path,"rb"))
    content_set = set()

    def find_next_node(node_sets):
        next_node_set = []
        for node in node_sets:
            next_node_set += node.children
            node_id = node.node_id
            node_content = node.content
            if node_content.strip("\n") == "":
                continue
            content_set.add(node_content)
        if len(next_node_set) == 0:
            return
        find_next_node(next_node_set)
    
    find_next_node(set([pkl_dict]))

    json.dump(list(content_set),open(save_path,"w"),ensure_ascii=False,indent=4)

if __name__ == "__main__":
    get_content("plot_0.pkl","plot_0.json")
    print('yes')
    get_content("plot_1.pkl","plot_1.json")
    
    
