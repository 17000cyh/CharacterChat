from utils import send_request, question_generate, insight_generate
from tqdm import tqdm

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


def generate_insights(nodes, window_size, max_depth, current_level=1):
    if current_level > max_depth or not nodes:
        root = Node('root', 'root', 0, 'Root Node')
        for node in nodes:
            root.add_child(node)
        return root
    
    output_list = []

    # Iterate through the nodes in windows to generate insights
    for i in tqdm(range(0, len(nodes), window_size)):
        window_nodes = nodes[i:i+window_size]
        questions = question_generate(window_nodes)
        insights = insight_generate(questions, window_nodes)    
        print("=" * 60)
        print("questions for this winodw size is")
        for i, question in enumerate(questions):
            print(f"{i}. {question}")
        print('-' * 60)
        print("insights for this window size is")
        for i, insight in enumerate(insights):
            print(f"{i}. {insight}")
        print("=" * 60)
        print("\n\n")

        for insight_id, insight in enumerate(insights, start=1):
            new_node = Node('insight', f"{current_level}-{insight_id}", current_level, insight)
            for node in window_nodes:
                new_node.add_child(node)
            output_list.append(new_node)
    
    # Recursively process the next level of insights
    return generate_insights(output_list, window_size, max_depth, current_level + 1)

def print_and_serialize_tree(node, level=0, file_path="tree.pkl"):
    print("  " * level + str(node))
    for child in node.children:
        print_and_serialize_tree(child, level + 1)
    
    # 只在根级别序列化以避免多次序列化
    if level == 0:
        with open(file_path, "wb") as file:
            pickle.dump(node, file)
        print(f"Tree has been serialized and saved to '{file_path}'.")


def process_story(plot_number):
    file_path = f"../data/original_story/plot_{plot_number}.txt"
    original_story = open(file_path).read()

    plots = original_story.split("-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - ")

    events = [Node('event', f"event{i}", 1, plot) for i, plot in enumerate(plots)]

    print("initial events are:")
    for event in events:
        print(event)
    
    root = generate_insights(events, 3, 2)  # window_size=3, max_depth=2

    print_and_serialize_tree(root, file_path=f"../data/reflection_result/plot_{plot_number}.pkl")


if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description="Process a story plot.")
    parser.add_argument("--plot_number", type=int, help="The number of the plot to process")
    
    args = parser.parse_args()
    
    process_story(args.plot_number)


