from hume.client import HumeClient
import cv2
import pyaudio
import wave
import numpy as np
import threading
import queue
import tempfile
import os
import asyncio
from datetime import datetime

class LiveHumeAnalyzer:
    def __init__(self, api_key):
        self.client = HumeClient(api_key=api_key)
        self.emotion_buffer = queue.Queue()
        self.is_recording = False
        
        # Audio recording settings
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio_queue = queue.Queue()
        
        # Video capture settings
        self.frame_queue = queue.Queue()
        self.fps = 30
        
    async def start_recording(self):
        """Start recording both audio and video"""
        self.is_recording = True
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        # Start video recording thread
        self.video_thread = threading.Thread(target=self._record_video)
        self.video_thread.start()
        
        # Start analysis loop
        await self._analysis_loop()
        
    async def stop_recording(self):
        """Stop recording and analysis"""
        self.is_recording = False
        self.audio_thread.join()
        self.video_thread.join()
        
    def _record_audio(self):
        """Record audio from microphone"""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        while self.is_recording:
            try:
                data = stream.read(self.chunk)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Audio recording error: {e}")
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
    def _record_video(self):
        """Record video from webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        while self.is_recording:
            ret, frame = cap.read()
            if ret:
                self.frame_queue.put(frame)
            else:
                print("Failed to capture video frame")
                
        cap.release()
        
    async def _analysis_loop(self):
        """Analyze the audio and video streams"""
        temp_dir = tempfile.mkdtemp()
        
        while self.is_recording:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                audio_path = os.path.join(temp_dir, f"audio_{timestamp}.wav")
                self._save_audio_segment(audio_path)
                
                video_path = os.path.join(temp_dir, f"video_{timestamp}.mp4")
                self._save_video_segment(video_path)
                
                # Use the async client for analysis
                job = await self.client.expression_measurement.batch.submit_jobs(
                    files=[video_path, audio_path],
                    models=["face", "prosody"]
                )
                
                results = await job.get_predictions()
                
                # Process results
                emotions = self._process_results(results)
                self.emotion_buffer.put(emotions)
                
                # Cleanup temporary files
                os.remove(audio_path)
                os.remove(video_path)
                
                await asyncio.sleep(1)  # Prevent overloading
                
            except Exception as e:
                print(f"Analysis error: {e}")
                await asyncio.sleep(1)
                
        os.rmdir(temp_dir)
        
    def _save_audio_segment(self, filepath, duration=3):
        """Save a short audio segment to file"""
        frames = []
        for _ in range(int(self.rate * duration / self.chunk)):
            if not self.audio_queue.empty():
                frames.append(self.audio_queue.get())
                
        if frames:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(pyaudio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                
    def _save_video_segment(self, filepath, duration=3):
        """Save a short video segment to file"""
        frames = []
        for _ in range(int(self.fps * duration)):
            if not self.frame_queue.empty():
                frames.append(self.frame_queue.get())
                
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
    def _process_results(self, results):
        """Process the analysis results"""
        emotions = {
            'face': {},
            'prosody': {},
            'transcript': ''
        }
        
        for result in results:
            if 'face' in result:
                for emotion in result['face']['predictions'][0]['emotions']:
                    emotions['face'][emotion['name']] = emotion['score']
                    
            if 'prosody' in result:
                for emotion in result['prosody']['predictions'][0]['emotions']:
                    emotions['prosody'][emotion['name']] = emotion['score']
                if 'text' in result['prosody']['predictions'][0]:
                    emotions['transcript'] += result['prosody']['predictions'][0]['text'] + ' '
                    
        return emotions
        
    def get_latest_emotions(self):
        """Get the most recent emotion analysis"""
        if not self.emotion_buffer.empty():
            return self.emotion_buffer.get()
        return None