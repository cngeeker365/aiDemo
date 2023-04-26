import openai

openai.api_key = "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"

prompt = "今天天气如何？"
model = "text-davinci-002"
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=50
)
print(response["choices"][0]["text"])

