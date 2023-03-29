import openai

openai.api_key = "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"

audio_file = open("test.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="text", language="zh")

print(transcript)

