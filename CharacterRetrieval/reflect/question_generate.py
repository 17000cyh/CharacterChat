from utils import question_generate
from reflect import Node


def build_questions_dict_from_tree(file_path="tree.pkl"):
    with open(file_path, "rb") as file:
        root = pickle.load(file)
    
    node_questions_dict = {}
    queue = [root]  # 使用队列实现广度优先遍历

    while queue:
        current_node = queue.pop(0)  # 取出队列的第一个元素
        questions = question_generate([current_node])  # 为当前节点生成问题
        node_questions_dict[current_node.node_id] = questions  # 将问题添加到字典中
        print("=" * 60)
        print(f"node")
        print(current_node)
        print(f"questions")
        print(questions)
        print("=" * 60)
        print("\n\n")
        queue.extend(current_node.children)  # 将当前节点的子节点添加到队列中

    return node_questions_dict

if __name__ == "__main__":
    import pickle
    from tqdm import tqdm

    node_questions_dict = build_questions_dict_from_tree("../data/reflection_result/plot_0.pkl")
    
    with open("../data/reflection_result/plot_0_question.pkl","wb") as file:
        pickle.dump(node_questions_dict, file)
    
