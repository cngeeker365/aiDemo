import subprocess

input_file = '/Users/apple/test.webm'
output_file = 'test.mp3'

# 使用FFmpeg将WebM文件转换为MP3
subprocess.call(["ffmpeg", "-i", input_file, "-vn", "-ar", "44100", "-ac", "2", "-ab", "192k", "-f", "mp3", output_file])

print('转换完成！')