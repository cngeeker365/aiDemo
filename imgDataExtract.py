import openai
import requests
from requests.structures import CaseInsensitiveDict
import json

openai.api_key = "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"

model = "image-alpha-001"
img_url = "https://x0.ifengimg.com/ucms/2021_12/18AB799AA136B5129D7D73A4C4E0ED8BE12A8091_size107_w1080_h810.jpg"
prompt = "这是一张医疗检验报告单，帮我提取报告中每一行数据的项目名称、结果、参考区间"

response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    inputs={
        "image": [img_url],
    },
    output_format="json"
)

output_url = response["choices"][0]["text"].strip()
response = requests.get(output_url)

