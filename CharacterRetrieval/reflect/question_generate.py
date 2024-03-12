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

def serialize_questions_dict(plot_number):
    # 生成树文件路径和问题文件路径
    tree_file_path = f"../data/reflection_result/plot_{plot_number}.pkl"
    question_file_path = f"../data/reflection_result/plot_{plot_number}_question.pkl"

    # 从树文件构建问题字典
    node_questions_dict = build_questions_dict_from_tree(tree_file_path)
    
    # 序列化问题字典到文件
    with open(question_file_path, "wb") as file:
        pickle.dump(node_questions_dict, file)
    print(f"Questions dictionary has been serialized and saved to '{question_file_path}'.")


if __name__ == "__main__":
    import pickle
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser(description="Serialize questions dictionary from a story plot tree.")
    parser.add_argument("plot_number", type=int, help="The number of the plot to process")
    
    args = parser.parse_args()
    
    serialize_questions_dict(args.plot_number)
