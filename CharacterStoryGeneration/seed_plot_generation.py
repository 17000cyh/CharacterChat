from utils import send_request


generation_prompt = """
你是一个才华横溢的专业小说家，你的写作风格你做的事情是根据一个基本的情节设定来构建小说的开头和结尾。

你需要注意：
1. 尽可能多的设计一些小说的故事背景，并且尽可能多样。这个背景是一些很宏大的背景，包含了发生的时间和发生的空间。
2. 故事要以某一个人为主角，反应他在很长一段时间当中的经历。注意，你起的名字不能太过于普通。
3. 明确故事的开头和结尾。
4. 使用中文。

你的输出是一个json的dict，内容如下:
{
    "begin" {
        "setting": 故事发生的设定背景,
        "plot": 故事的开头情节,
        "characters_dict": {
            "人物名称1": 人物介绍,
            "人物名称2": 人物介绍,
            .... 可能还存在其他人物
        }
    }
    "end" {
        "setting": 故事发生的设定背景,
        "plot": 故事的结尾情节,
        "characters_dict": {
            "人物名称1": 人物介绍,
            "人物名称2": 人物介绍,
            .... 可能还存在其他人物
        }
    }
}

注意，上述的characters_dict可以有更多个，但是至少要有一个。一般来说有5-10个人物的参与。
基本的情节如下:

{{plot}}

你的输出如下:

"""

import json
from tqdm import tqdm

seed_result = open("seed_result.json", "a")

for i, plot in tqdm(enumerate(open("seed_plot.txt").readlines())):
    if i < 55:
        continue

    if plot.strip() == "":
        continue
    response = send_request(generation_prompt.replace("{{plot}}", plot))
    print("=" * 60)
    print(f"query is\n {generation_prompt.replace('{{plot}}', plot)}")
    print(f"response is\n {response}")
    print("\n\n" + "=" * 60)
    try:
        response_json = json.loads(response)
    except:
        print("parse wrong")
        continue


    seed_result.write(json.dumps(response_json, ensure_ascii=False))
    seed_result.write("\n")

