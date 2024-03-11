from utils import send_request, question_generate, insight_generate

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
    for i in range(0, len(nodes), window_size):
        window_nodes = nodes[i:i+window_size]
        questions = question_generate(window_nodes)
        insights = insight_generate(questions, materials)
        
        for insight_id, insight in enumerate(insights, start=1):
            new_node = Node('insight', f"{current_level}-{insight_id}", current_level, insight)
            for node in window_nodes:
                new_node.add_child(node)
            output_list.append(new_node)
    
    # Recursively process the next level of insights
    return generate_insights(output_list, window_size, max_depth, current_level + 1)

if __name__ == "__main__":
    

    # Example usage with revised parameters
    events = [Node('event', f"event{i}", 1, f"Event {i} content") for i in range(1, 7)]
    root = generate_insights(events, 3, 3)  # window_size=3, max_depth=3

    # Function to print the tree for visualization
    def print_tree(node, level=0):
        print("  " * level + str(node))
        for child in node.children:
            print_tree(child, level + 1)

    print_tree(root)
