from retrieval import search_material, load_embeddings
from utils import send_request
import re
import copy 
from tqdm import tqdm

planner_prompt_template = open("prompts/planner_prompt.txt").read()
searchqa_prompt_template = open("prompts/searchqa_prompt.txt").read()
style_transfer_template = open("prompts/style_transfer_prompt.txt").read()

def template_construct(prompt_template, template_dict):
    query_template = copy.deepcopy(prompt_template)
    for key in template_dict.keys():

        query_template = query_template.replace("{{"+key+"}}", template_dict[key])
    
    return query_template

def parse_text(input_str):
    # 匹配Action后面的方法和方括号内的内容
    pattern = r"Action: (\w+)\[(.*?)\]"
    match = re.search(pattern, input_str)
    
    if match:
        action_method = match.group(1)
        content = match.group(2)
        return action_method, content
    else:
        return "Response", input_str

    
def chat_pipeline(knoledge_embeddings, query, prompt_template_dict, topk=5):
    planner_prompt = template_construct(planner_prompt_template, prompt_template_dict["planner template"])
    style_transfer_prompt = template_construct(style_transfer_template, prompt_template_dict["style transfer template"])

    planner_query = planner_prompt.replace("{{query}}", query)
    print(f"planner query is {planner_query}")
    planner_response = send_request(planner_query)
    action_method, content = parse_text(planner_response)

    print("planner_response: ", planner_response)
    print("action_method: ", action_method)
    print("content: ", content)

    searchqa_response = ""
    materials_str = ""

    if "searchqa" in action_method.lower():
        materials = search_material(query, knoledge_embeddings, topk)
        for i, material in enumerate(materials):
            materials_str += f"{i+1}. {material}\n"
        print("materials: ", materials)
        searchqa_prompt = searchqa_prompt_template.replace("{{query}}", content).replace("{{documents}}", materials_str)
        print("searchqa_prompt: ", searchqa_prompt)
        searchqa_response = send_request(searchqa_prompt)
        content = searchqa_response
        print("searchqa_response: ", content)

    
    style_transfer_query = style_transfer_prompt.replace("{{query}}", content)
    print("style_transfer_query: ", style_transfer_query)
    stylized_content = send_request(style_transfer_query)

    print("stylized_content: ", stylized_content)
    
    return {
        "query": query,
        "planner_response": planner_response,
        "searchqa_response": searchqa_response,
        "stylized_content": stylized_content,
        "materials": materials_str,
    }


def deal_one_conversation_list(knoledge_embeddings, conversation_list, prompt_template_dict, jsonl_file):
    conversation_result = []
    for query in tqdm(conversation_list):
        try:
            result = chat_pipeline(knoledge_embeddings, query, prompt_template_dict)
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        except:
            print(f"error in query: {query}")
            continue



if __name__ == "__main__":
    import json
    import os

    knoledge_embeddings = load_embeddings(
        json.load(open("data/plot_data/plot_1.json")),
    )
    query_list = open("data/test_data/plot_1.txt").readlines()
    prompt_template_dict =  json.load(open("data/prompt_template_variable/plot_1_dict_template.json"))
    
    os.makedirs("results_chat", exist_ok=True)
    jsonl_file = open("results_chat/plot_qwen72b_1.jsonl", "w")
    deal_one_conversation_list(knoledge_embeddings, query_list, prompt_template_dict, jsonl_file)




    