import time
from qwen_utils import send_request
import requests
from openai import OpenAI

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


# client = OpenAI(api_key="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")

# client = OpenAI(api_key="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")


# def send_request(query, model="gpt-4"):
#   response = ""

#   for i in range(0,10):
#     # try:
#       completion = client.chat.completions.create(
#         model=model,
#         messages=[
#           {"role": "user", "content": query}
#         ],
#         temperature=0.5,
#       )

#       # print(completion.choices[0].message)

#       return completion.choices[0].message.content
#     # except:
#     #   print("sth wrong, skip")
      
#       continue
#   return response



def characters_dict_to_str(characters_dict):
    characters_str = ""
    for name, description in characters_dict.items():
        characters_str += f"{name}：{description}\n"
    return characters_str.strip()

if __name__ == "__main__":
    print(send_request('hi'))