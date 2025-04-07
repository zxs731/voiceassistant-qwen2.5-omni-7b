import pyaudio
import wave
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import logging
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化配置
load_dotenv("./qwen.env")
client = OpenAI(
    api_key=os.environ["key"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class AudioConfig:
    """音频配置常量类"""
    # 录音参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000           # 16kHz更适合语音识别
    CHUNK = 480            # 30ms的帧大小
    SILENCE_TIMEOUT = 0.4  # 静音超时(秒)
    THRESHOLD = 10000      # 音量阈值
    RECORD_MAXSECONDS = 20  # 最大录音时长
    # 播放参数
    PLAY_RATE = 24000      # TTS通常使用24kHz
    PLAY_CHUNK = 1024      # 播放块大小

class ConversationManager:
    """管理对话历史和系统消息"""
    def __init__(self):
        self.sys_msg = {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        }
        self.messages: List[Dict[str, Any]] = []
        self.max_history = 5  # 保留最近的5条对话
    
    def add_user_message(self, content: Dict[str, Any]) -> None:
        """添加用户消息"""
        self.messages.append({"role": "user", "content": [content]})
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手回复"""
        if content.strip():
            self.messages.append({"role": "assistant", "content": content.strip()})
            self._trim_history()
    
    def get_recent_messages(self) -> List[Dict[str, Any]]:
        """获取最近的对话历史"""
        return [self.sys_msg] + self.messages[-self.max_history:]
    
    def _trim_history(self) -> None:
        """修剪对话历史"""
        if len(self.messages) > self.max_history * 2:  # 用户和助手各5条
            self.messages = self.messages[-self.max_history * 2:]

class AudioPlayer:
    """专用于音频播放的类"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def __del__(self):
        self.close()
    
    def close(self):
        """释放资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
    
    def play_audio_stream(self, audio_data: bytes) -> None:
        """播放音频流，带有缓冲处理"""
        if not self.stream:
            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.PLAY_RATE,
                output=True,
                frames_per_buffer=AudioConfig.PLAY_CHUNK,
                start=False
            )
            self.stream.start_stream()
        
        try:
            self.stream.write(audio_data)
        except Exception as e:
            logger.error(f"播放音频时出错: {e}")
            self.close()
            raise
            
class AudioProcessor:
    """处理音频录制和播放"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
    
    def __del__(self):
        self.audio.terminate()
    
    @staticmethod
    def get_volume(data: bytes) -> float:
        """计算音频数据的音量"""
        return np.linalg.norm(np.frombuffer(data, dtype=np.int16))
    
    def record_audio(self, output_path: str = "output.wav") -> bool:
        """
        录制音频直到检测到静音或达到最大时长
        返回: 是否成功录制
        """
        stream = self.audio.open(
            format=AudioConfig.FORMAT,
            channels=AudioConfig.CHANNELS,
            rate=AudioConfig.RATE,
            input=True,
            frames_per_buffer=AudioConfig.CHUNK
        )
        logger.info("请开始说话...（静音自动结束）")
        
        frames = []
        recording = False
        silence_start_time = None
        start_time = time.time()
        
        try:
            while True:
                data = stream.read(AudioConfig.CHUNK)
                volume = self.get_volume(data)
                
                # 检测语音开始
                if volume > AudioConfig.THRESHOLD and not recording:
                    logger.info("检测到语音，开始录音...")
                    recording = True
                    start_time = time.time()
                    silence_start_time = None
                
                if recording:
                    frames.append(data)
                    
                    # 检查最大录音时长
                    if time.time() - start_time > AudioConfig.RECORD_MAXSECONDS:
                        logger.info("达到最大录音时间，停止录音")
                        break
                    
                    # 前2秒不检测静音
                    if time.time() - start_time < 2:
                        continue
                    
                    # 检查静音
                    if volume <= AudioConfig.THRESHOLD:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > AudioConfig.SILENCE_TIMEOUT:
                            logger.info("检测到静音，停止录音")
                            break
                    else:
                        silence_start_time = None
        finally:
            stream.stop_stream()
            stream.close()
        
        if frames:
            self._save_wav(output_path, frames)
            return True
        return False
    
    def _save_wav(self, path: str, frames: List[bytes]) -> None:
        """保存WAV文件"""
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(AudioConfig.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(AudioConfig.FORMAT))
            wf.setframerate(AudioConfig.RATE)
            wf.writeframes(b''.join(frames))
    
    @staticmethod
    def encode_audio(file_path: str) -> str:
        """将WAV文件编码为base64"""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    
    def play_audio_stream(self, audio_data: bytes) -> None:
        """流式播放音频"""
        stream = self.audio.open(
            format=AudioConfig.FORMAT,
            channels=AudioConfig.CHANNELS,
            rate=AudioConfig.PLAY_RATE,
            output=True,
            frames_per_buffer=AudioConfig.PLAY_CHUNK
        )
        try:
            stream.write(audio_data)
        finally:
            stream.stop_stream()
            stream.close()

class VoiceAssistant:
    """语音助手主类"""
    def __init__(self):
        self.conversation = ConversationManager()
        self.audio_processor = AudioProcessor()
        self.audio_player = AudioPlayer()  
    
    def run(self) -> None:
        """运行语音助手主循环"""
        logger.info("语音助手已启动，按Ctrl+C退出")
        try:
            while True:
                self.process_conversation()
        except KeyboardInterrupt:
            logger.info("\n程序退出")
    
    def process_conversation(self) -> None:
        """处理一次完整的对话交互"""
        # 1. 录音
        if not self.audio_processor.record_audio():
            logger.warning("未检测到有效语音")
            return
        
        # 2. 准备请求
        base64_audio = self.audio_processor.encode_audio("output.wav")
        self.conversation.add_user_message({
            "type": "input_audio",
            "input_audio": {
                "data": f"data:;base64,{base64_audio}",
                "format": "wav",
            }
        })
        
        # 3. 流式请求和播放
        full_response = ""
        
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-omni-7b",
                messages=self.conversation.get_recent_messages(),
                modalities=["text", "audio"],
                audio={"voice": "Chelsie", "format": "wav"},
                stream=True,
                stream_options={"include_usage": True},
            )
            
            print("\nAI: ")
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'audio'):
                        # 处理文本转录
                        transcript = delta.audio.get("transcript", "")
                        if transcript:
                            print(transcript, end="", flush=True)
                            full_response += transcript
                        
                        # 处理音频数据
                        audio_data = delta.audio.get("data", "")
                        if audio_data:
                            wav_bytes = base64.b64decode(audio_data)
                            self.audio_player.play_audio_stream(wav_bytes)
        except Exception as e:
            logger.error(f"请求过程中发生错误: {e}")
            return
        
        
        # 4. 保存对话
        self.conversation.add_assistant_message(full_response)
        if full_response.strip():
            print()  # 换行
        else:
            logger.warning("未收到有效回复")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()