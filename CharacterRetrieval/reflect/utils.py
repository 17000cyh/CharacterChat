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

if __name__ == "__main__":
    print(send_request('hi'))
