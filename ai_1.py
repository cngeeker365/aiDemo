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

