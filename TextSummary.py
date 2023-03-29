#-*- coding:utf-8 -*-
import openai
import re

openai.api_key = "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"

# 定义总结文本的函数
def summarize_text(text, max_tokens=120):
    prompt = "Please summarize the following text:\n\n" + text + "\n\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens
    )
    summary = response.choices[0].text
    summary = re.sub('^[0-9]*[.、）]', '', summary)
    return summary

# 测试函数
text = "OpenAI is an artificial intelligence company headquartered in San Francisco, founded in 2015. The company's mission is to create a more intelligent future for humanity."
summary = summarize_text(text)
print(summary)