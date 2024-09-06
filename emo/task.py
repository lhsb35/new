import json
from lmdeploy import pipeline

# 初始化模型
'''
30_8_2e-4_adamW ✔
10_8_2e-4_adamW ✔
20_8_2e-4_adamW ✔
30_4_2e-4_adamW ✔
30_1_2e-4_adamW ✔
30_8_1e-3_adamW ✔
30_8_3e-5_adamW ✔
30_8_2e-4_adam ✔
30_8_2e-4_adamax ✔
'''
result_name = "cn_30_8_2e-4_adamax.txt"

pipe = pipeline("/root/autodl-tmp/Alexander/internlm2_5-7b-chat2")
json_name = "data.json"

# 定义情感分析的函数
def analyze_sentiment(text):
    prompt = f"请判断以下句子的情感：'{text}'。愉快、中性和愤怒分别标记为0, 1, 2。只返回数字。"
    response = pipe([prompt])
    response_text = response[0].text  # 提取text属性
    # 提取数字并返回
    return int(response_text.strip())

# 读取JSON文件
with open(json_name, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 分析情感并存储结果
results = []
for item in data:
    output_text = item['conversation'][0]['output']
    sentiment = analyze_sentiment(output_text)
    results.append(sentiment)

# 将结果写入文本文件
with open(result_name, 'w', encoding='utf-8') as file:
    file.write(str(results))

print(f"情感分析结果已保存到 {result_name} 中，内容为：{results}")
