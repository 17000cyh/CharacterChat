import time
# from qwen_utils import send_request
import requests
from openai import OpenAI

# client = OpenAI(api_key="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")

client = OpenAI(api_key="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")


def send_request(query, model="gpt-3.5-turbo"):
  response = ""

  for i in range(0,10):
    # try:
      completion = client.chat.completions.create(
        model=model,
        messages=[
          {"role": "user", "content": query}
        ],
        temperature=0.5,
      )

      # print(completion.choices[0].message)

      return completion.choices[0].message.content
    # except:
    #   print("sth wrong, skip")
      
      continue
  return response



def characters_dict_to_str(characters_dict):
    characters_str = ""
    for name, description in characters_dict.items():
        characters_str += f"{name}：{description}\n"
    return characters_str.strip()


def question_generate(nodes):
    # 准备大模型的输入
    materials = [node.content for node in nodes]  # 从nodes提取材料内容
    prompt = open("prompts/reflection_question_generation.txt").read() 
    prompt = prompt.replace("{{materials_text}}", "\n".join(materials))
    # 调用send_request与大模型交互
    response = send_request(prompt)
    
    # 解析大模型的输出并返回
    # 假设response的格式是"1. Question1 2. Question2"，我们需要解析这个格式
    questions = response.split('\n')[1:]  # 移除首行，然后每个问题是一行
    print(f"questions is {questions}")
    return questions

def insight_generate(questions):
    # Updated to use questions for insight generation
    insights = [f"Insight based on question: {question}" for question in questions]
    return insights

if __name__ == "__main__":
    print(send_request('hi'))
