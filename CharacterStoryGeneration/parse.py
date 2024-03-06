import re

class Parser:
    def __init__(self):
        pass

    def parse_multi_plot(self, script):
        # 使用正则表达式匹配任意数字的幕
        acts = re.split(r'\[第.*?幕\]', script)
        acts = [act.strip() for act in acts if act.strip()]

        parsed_data = []

        for act in acts:
            # 找出每个部分的内容
            setting = re.search(r'\[setting\]:\s*(.*?)\s*\[plot\]:', act, re.DOTALL)
            plot = re.search(r'\[plot\]:\s*(.*?)\s*\[characters\]:', act, re.DOTALL)
            characters = re.search(r'\[characters\]:\s*(.*)', act, re.DOTALL)

            # 清理和解析文本
            setting_text = setting.group(1).strip() if setting else ""
            plot_text = plot.group(1).strip() if plot else ""
            characters_text = characters.group(1).strip() if characters else ""

            # 解析人物
            characters_dict = {}
            for line in characters_text.split('\n'):
                if ':' in line:
                    name, description = line.split(':', 1)
                    characters_dict[name.strip()] = description.strip()
                elif '：' in line:
                    name, description = line.split('：', 1)
                    characters_dict[name.strip()] = description.strip()


            # 组合信息
            act_data = {
                "setting": setting_text,
                "plot": plot_text,
                "characters_dict": characters_dict
            }

            parsed_data.append(act_data)

        return parsed_data

    def parse_single_plot(self, script):
        # 解析场景、情节和人物
        setting = re.search(r'\[setting\]:\s*(.*?)\s*\[plot\]:', script, re.DOTALL)
        plot = re.search(r'\[plot\]:\s*(.*?)\s*\[characters\]:', script, re.DOTALL)
        characters = re.search(r'\[characters\]:\s*(.*)', script, re.DOTALL)

        # 清理和解析文本
        setting_text = setting.group(1).strip() if setting else ""
        plot_text = plot.group(1).strip() if plot else ""
        characters_text = characters.group(1).strip() if characters else ""

        # 解析人物
        characters_dict = {}
        for line in characters_text.split('\n'):
            if ':' in line:
                name, description = line.split(':', 1)
                characters_dict[name.strip()] = description.strip()
            elif '：' in line:
                name, description = line.split('：', 1)
                characters_dict[name.strip()] = description.strip()

        # 组合信息
        scene_data = {
            "setting": setting_text,
            "plot": plot_text,
            "characters_dict": characters_dict
        }

        return scene_data


if __name__ == '__main__':


    parser = Parser()

    test1 = \
    """
[setting]:
宇宙飞船穿越星际的途中，周围是浩渺无垠的宇宙空间，星星点点，宛如一片璀璨的宝石海洋。飞船内部，各种仪器发出微弱的嗡嗡声，显示屏上不断闪烁着各种数据和图像。
[plot]:
在前往遥远星系的途中，艾伦和萨拉轮流驾驶飞船，时刻保持警惕。他们不仅要密切关注飞船的各项指标，还要应对可能出现的突发状况。途中，他们遇到了一次猛烈的太阳风暴。风暴中的高能粒子和电磁辐射对飞船的通信系统造成了严重干扰，甚至威胁到飞船的结构安全。艾伦凭借丰富的知识和经验，迅速调整了飞船的航向和防御系统，成功避开了风暴的主力。然而，风暴过后，他们发现飞船的能源系统受到了严重损伤，无法继续维持高速航行。
[characters]:
艾伦: 在应对太阳风暴的过程中展现出卓越的判断力和驾驶技术，成功避免了飞船受损的更严重后果。
萨拉: 在风暴过后，迅速投入到能源系统的维修工作中，她的技术能力和冷静应对为飞船的恢复争取了宝贵的时间。
    """
    print(parser.parse_single_plot(test1))