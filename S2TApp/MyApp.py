import sys
import time

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton
import sounddevice as sd
import numpy as np
import wavio
import threading
import openai

openai.api_key = "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"

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Add the conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        layout.addWidget(self.conversation_display)

        # Add the record button
        self.record_button = QPushButton("Record")
        layout.addWidget(self.record_button)

        # Connect the record button to a function
        self.record_button.clicked.connect(self.toggle_recording)

        # Set the layout for the app
        self.setLayout(layout)

    # Add the toggle_recording function
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
            self.record_button.setText("Start Recording")
        else:
            self.start_recording()
            self.record_button.setText("Stop Recording")

    # Add the start_recording function
    def start_recording(self):
        self.is_recording = True
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def record(self):
        fs = 44100  # 采样频率
        duration = 5  # 录音时长，以秒为单位

        # 录音
        print("开始录音...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        print("录音结束")

        # 保存录音文件
        fileName = time.time() + ".wav"
        wavio.write(fileName, recording, fs, sampwidth=2)

    # Add the stop_recording function
    def stop_recording(self):
        self.is_recording = False
        self.record_thread.join()  # 等待录音线程结束

    # Add the play_recording function
    def play_recording(self):
        # Code to play recording
        pass

    # Add the speech to text function
    def speech_to_text(self):
        # Code to convert speech to text
        audio_file = open("test.mp3", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="text", language="zh")

        print(transcript)

    # Add the text to speech function
    def text_to_speech(self):
        # Code to convert text to speech
        pass

    # Add conversation display function
    def display_conversation(self):
        # Code to display the
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec_())
