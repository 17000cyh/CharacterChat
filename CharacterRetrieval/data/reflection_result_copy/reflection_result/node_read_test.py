import pickle

class Node:
    def __init__(self, node_type, node_id, level, content, children=None, parent=None):
        self.node_type = node_type  # 'event' or 'insight'
        self.node_id = node_id
        self.level = level
        self.content = content
        self.children = children if children is not None else []
        self.parent = parent

    def __str__(self):
        child_str = ""
        for item in self.children:
            child_str += str(item.node_id)
        return f"Node(ID: {self.node_id}, Type: {self.node_type}, Level: {self.level}, Content: {self.content}), Child: {child_str}"

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

# Redefine the print_tree function to ensure it's available after reset
content_set = set()

def print_tree(node, level=0):
    # print("  " * level + str(node))
    global content_set
    print("  " * level + str(level) + str(node)[45:60]+"...")
    content_set.add(
        node.content
    )
    for child in node.children:
        print_tree(child, level + 1)

# Function to deserialize the tree from a pickle file and print its structure

def deserialize_and_print_tree(file_path="tree.pkl"):
    with open(file_path, "rb") as file:
        root = pickle.load(file)
    
    # Use the previously defined print_tree function to print the deserialized tree
    print_tree(root)

# Assuming the tree was serialized and saved to 'tree.pkl'
# Since this environment is reset, ensure the Node class and any necessary classes or functions are defined before deserialization.
# deserialize_and_print_tree()

deserialize_and_print_tree("plot_0.pkl")
print(f"len of content is {len(content_set)}")
