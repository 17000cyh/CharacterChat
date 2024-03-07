from utils import send_request, characters_dict_to_str
import re
from parse import Parser
from tqdm import tqdm
import json


parser = Parser()

def framework_insert_edit(
        original_framework: dict,
        last_plot: dict,
        next_plot: dict,
        insert_select_example=open("Prompts/场景内插选择Prompt example.txt").read(),
        edit_example=open("Prompts/场景内插批评后修改Prompt example.txt").read(),
        critique_example=open("Prompts/场景内插批评Prompt example.txt").read(),
):
    print("开始进行情节修正")
    for i in tqdm(range(5)):
        print("开始对当前的情节进行分析")
        critique_prompt = open("Prompts/场景内插批评Prompt.txt").read()
        critique_prompt = critique_prompt.replace("{{plot}}", original_framework["plot"])
        critique_prompt = critique_prompt.replace("{{plot_last}}", last_plot["plot"])
        critique_prompt = critique_prompt.replace("{{plot_next}}", next_plot["plot"])
        critique_prompt = critique_prompt.replace("{{examples}}", critique_example)
        critique_str = send_request(critique_prompt)

        print(f"-------------- critique str is ---------------")
        print("critique prompt is")
        print(critique_prompt)
        print(critique_str)
        print(f"-------------- critique str over ---------------")

        # 根据批评的结果进行修改
        print("开始进行情节修改")
        edit_prompt = open("Prompts/场景内插批评后修改Prompt.txt").read()
        edit_prompt = edit_prompt.replace("{{plot}}", original_framework["plot"])
        edit_prompt = edit_prompt.replace("{{plot_last}}", last_plot["plot"])
        edit_prompt = edit_prompt.replace("{{plot_next}}", next_plot["plot"])

        edit_prompt = edit_prompt.replace("{{setting}}", original_framework["setting"])
        edit_prompt = edit_prompt.replace("{{setting_last}}", last_plot["setting"])
        edit_prompt = edit_prompt.replace("{{setting_next}}", next_plot["setting"])

        edit_prompt = edit_prompt.replace("{{character}}", characters_dict_to_str(original_framework["characters_dict"]))
        edit_prompt = edit_prompt.replace("{{character_last}}", characters_dict_to_str(last_plot["characters_dict"]))
        edit_prompt = edit_prompt.replace("{{character_next}}", characters_dict_to_str(next_plot["characters_dict"]))

        edit_prompt = edit_prompt.replace("{{examples}}", edit_example)

        edit_prompt = edit_prompt.replace("{{critique}}", critique_str)
        edit_str = send_request(edit_prompt)
        print(f"-------------- edit str is ---------------")
        print("edit prompt is")
        print(edit_prompt)
        print(edit_str)
        print(f"-------------- edit str over ---------------")

        edit_result = parser.parse_single_plot(edit_str)

        # 判断修改之后的结果是否更好。如果确实更好，直接return。否则继续修改。
        print("开始进行情节选择")
        select_prompt = open("Prompts/场景内插选择Prompt.txt").read()
        select_prompt = select_prompt.replace("{{plot1}}", original_framework["plot"])
        select_prompt = select_prompt.replace("{{plot2}}", edit_result["plot"])
        select_prompt = select_prompt.replace("{{plot_last}}", last_plot["plot"])
        select_prompt = select_prompt.replace("{{plot_next}}", next_plot["plot"])

        select_prompt = select_prompt.replace("{{examples}}", insert_select_example)
        select_str = send_request(select_prompt)

        print(f"-------------- select str is ---------------")
        print("select prompt is")
        print(select_prompt)
        print(select_str)
        print(f"-------------- select str over ---------------")

        select_json = json.loads(select_str)
        if select_json["selection"] == "1":
            # 说明原始的结果更好，直接返回
            print("当前的情节已经是最优情节，不继续进行修改")
            return original_framework
        else:
            # 说明修改之后的结果更好，继续修改
            print("当前可能还存在最优情节，继续修改")
            original_framework = edit_result

    return edit_result


def framework_insert(last_plot:dict, next_plot:dict, example=open("Prompts/场景内插Prompt example.txt").read(), multi_generate_time=5):
    setting_last = last_plot["setting"]
    plot_last = last_plot["plot"]
    character_last = characters_dict_to_str(last_plot["characters_dict"])
    setting_next = next_plot["setting"]
    plot_next = next_plot["plot"]
    character_next = characters_dict_to_str(next_plot["characters_dict"])

    insert_prompt = open("Prompts/场景内插Prompt.txt").read()
    insert_prompt = insert_prompt.replace("{{setting_last}}", setting_last)
    insert_prompt = insert_prompt.replace("{{plot_last}}", plot_last)
    insert_prompt = insert_prompt.replace("{{character_last}}", character_last)
    insert_prompt = insert_prompt.replace("{{setting_next}}", setting_next)
    insert_prompt = insert_prompt.replace("{{plot_next}}", plot_next)
    insert_prompt = insert_prompt.replace("{{character_next}}", character_next)
    insert_prompt = insert_prompt.replace("{{examples}}", example)

    # 多生成几次，取最好的
    orignal_insert = send_request(insert_prompt)
    print(f"-------------- insert str is ---------------")
    print("insert prompt is")
    print(insert_prompt)
    print(orignal_insert)
    print(f"-------------- insert str over ---------------")

    original_insert = parser.parse_single_plot(orignal_insert)

    return framework_insert_edit(
        original_insert,
        last_plot,
        next_plot,
    )

    # return parser.parse_single_plot(insert_str)

def detail_plot(now_plot:dict, example=open("Prompts/场景细化Prompt  example.txt").read()):
    setting = now_plot["setting"]
    plot = now_plot["plot"]
    character = characters_dict_to_str(now_plot["characters_dict"])
    detail_prompt = open("Prompts/场景细化Prompt.txt").read()

    detail_prompt = detail_prompt.replace("{{setting}}", setting)
    detail_prompt = detail_prompt.replace("{{plot}}", plot)
    detail_prompt = detail_prompt.replace("{{character}}", character)
    detail_prompt = detail_prompt.replace("{{examples}}", example)
    detail_str = send_request(detail_prompt)

    print(f"-------------- detail str is ---------------")
    print("detail prompt is")
    print(detail_prompt)
    print(detail_str)
    print(f"-------------- detail str over ---------------")
    print("\n")

    return parser.parse_multi_plot(detail_str)

def create_scene(now_plot:dict, last_content="", example=open("Prompts/场景描写Prompt example.txt").read()):
    setting = now_plot["setting"]
    plot = now_plot["plot"]
    character = characters_dict_to_str(now_plot["characters_dict"])
    create_prompt = open("Prompts/场景描写Prompt.txt").read()
    create_prompt = create_prompt.replace("{{setting}}", setting)
    create_prompt = create_prompt.replace("{{plot}}", plot)
    create_prompt = create_prompt.replace("{{character}}", character)
    create_prompt = create_prompt.replace("{{examples}}", example)
    create_prompt = create_prompt.replace("{{last_content}}", last_content)
    create_str = send_request(create_prompt)

    print(f"-------------- create str is ---------------")
    print(f"prompt is ")
    print(create_prompt)
    print(create_str)
    print(f"-------------- create str over ---------------")
    print("\n")

    return create_str


def generate_story_framework(
    seed_framework: [dict],
    horizontal_expansion:int=1,
    vertical_expansion:int=1,
    result_json_file = open("plot_json.json","w"),
    result_txt_file = open("plot.txt","w")
):
    framework_list = seed_framework

    for _ in range(horizontal_expansion):
        expand_framework_list = []

        for i in tqdm(range(0, len(framework_list) - 1)):
            expand_framework_list.append(framework_list[i])

            last_plot = framework_list[i]
            next_plot = framework_list[i + 1]
            insert_plot = framework_insert(last_plot, next_plot)
            expand_framework_list.append(insert_plot)
        expand_framework_list.append(seed_framework[-1])

        framework_list = expand_framework_list

        print(f"now frame work list is ")
        for item in framework_list:
            print(item)

    vertical_framework_list = [
        framework_list
    ]

    for _ in range(vertical_expansion):
        expand_framework_list_child_layer = []

        for i in tqdm(range(0, len(vertical_framework_list[-1]))):
            plot_item = framework_list[i]
            detail_plot_list = detail_plot(plot_item)

            expand_framework_list_child_layer += detail_plot_list

        vertical_framework_list.append(expand_framework_list_child_layer)

    json.dump(vertical_framework_list, result_json_file
               , indent=4, ensure_ascii=False)

    create_str = ""
    for i in range(len(vertical_framework_list[-1])):
        create_str = create_scene(vertical_framework_list[-1][i], last_content=create_str)
        result_txt_file.write(create_str)
        result_txt_file.write(" - " * 20)
        result_txt_file.write("\n")


if __name__ == "__main__":
    import os
    from tqdm import tqdm

    os.makedirs("novel_result", exist_ok=True)

    for i, line in enumerate(open("seed_result.jsonl").readlines()):

        # try:
            seed_dict = json.loads(line)
            beginning = seed_dict['begin']
            ending = seed_dict['end']
            
            seed_framework = [beginning, ending]
            generate_story_framework(
                seed_framework,
                result_json_file = open(f"novel_result/plot_{i}_json.json","w"),
                result_txt_file = open(f"novel_result/plot_{i}.txt","w")
                )
        # except:
        #     print("sth wrong")