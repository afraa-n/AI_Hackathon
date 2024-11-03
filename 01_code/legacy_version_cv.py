import cv2
import numpy as np
from deepface import DeepFace
import time
from threading import Thread, Event
import speech_recognition as sr
from picamera import PiCamera
from io import BytesIO
from PIL import Image


class SimpleMoodDetector:
    def __init__(self):
        print("[DEBUG] Initializing SimpleMoodDetector...")
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize Raspberry Pi camera
        try:
            self.camera = PiCamera()
            self.camera.resolution = (640, 480)
            self.camera.framerate = 24
            # Warm up camera
            time.sleep(2)
            print("[DEBUG] Raspberry Pi camera initialized successfully")
        except Exception as e:
            print(f"[DEBUG] Error initializing camera: {e}")
            raise

        # Set up speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("[DEBUG] Speech recognition initialized")

        # Threading events
        self.stop_camera = Event()
        self.speaking = Event()

        # We'll only keep these emotions
        self.target_emotions = ["happy", "sad", "angry", "neutral"]
        self.detected_mood = None

    def capture_frame(self):
        """Capture a frame from PiCamera and convert to OpenCV format"""
        stream = BytesIO()
        self.camera.capture(stream, format="jpeg")
        stream.seek(0)
        image = Image.open(stream)
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame

    def detect_mood(self, frame):
        """Detect mood using DeepFace"""
        try:
            print("[DEBUG] Starting mood detection with DeepFace...")
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )

            emotion = result[0]["dominant_emotion"]
            print(f"[DEBUG] DeepFace detected emotion: {emotion}")
            print(f"[DEBUG] All emotions detected: {result[0]['emotion']}")

            if emotion in self.target_emotions:
                print(f"[DEBUG] Returning target emotion: {emotion}")
                return emotion
            print("[DEBUG] Emotion not in target list, defaulting to neutral")
            return "neutral"

        except Exception as e:
            print(f"[DEBUG] Error in mood detection: {e}")
            return None

    def camera_loop(self):
        """Main camera loop for face detection"""
        print("[DEBUG] Starting camera loop")
        while not self.stop_camera.is_set():
            if self.speaking.is_set():
                try:
                    # Capture frame from Raspberry Pi camera
                    frame = self.capture_frame()
                    print("[DEBUG] Frame captured successfully")

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces using OpenCV
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    # If face detected, process mood
                    if len(faces) > 0:
                        print(f"[DEBUG] Detected {len(faces)} faces")
                        self.detected_mood = self.detect_mood(frame)

                        if self.detected_mood:
                            print("[DEBUG] Mood detected, stopping camera")
                            self.stop_camera.set()
                            self.speaking.clear()
                            break
                    else:
                        print("[DEBUG] No faces detected in this frame")

                except Exception as e:
                    print(f"[DEBUG] Error capturing frame: {e}")
                    continue

    def listen_for_speech(self):
        """Listen for speech and trigger face detection"""
        print("[DEBUG] Starting speech recognition")
        with self.microphone as source:
            print("[DEBUG] Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("[DEBUG] Ambient noise adjustment complete")

            while not self.stop_camera.is_set():
                try:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=5)
                    print("[DEBUG] Speech detected!")

                    # When speech detected, trigger camera
                    self.speaking.set()
                    print("[DEBUG] Camera triggered")
                    time.sleep(0.5)  # Give camera time to initialize

                except sr.WaitTimeoutError:
                    print("[DEBUG] Speech timeout, continuing to listen")
                    continue
                except Exception as e:
                    print(f"[DEBUG] Error in speech recognition: {e}")
                    continue

    def get_mood(self):
        """Main function to run the mood detection pipeline"""
        try:
            print("[DEBUG] Starting mood detection pipeline")
            # Start speech recognition thread
            speech_thread = Thread(target=self.listen_for_speech)
            speech_thread.daemon = True
            speech_thread.start()
            print("[DEBUG] Speech thread started")

            # Start camera thread
            camera_thread = Thread(target=self.camera_loop)
            camera_thread.daemon = True
            camera_thread.start()
            print("[DEBUG] Camera thread started")

            # Wait for mood detection
            while not self.stop_camera.is_set():
                time.sleep(0.1)

        finally:
            # Clean up
            print("[DEBUG] Cleaning up resources")
            self.camera.close()
            cv2.destroyAllWindows()

        return self.detected_mood


def main():
    print("[DEBUG] Starting main program")
    # Initialize the detector
    detector = SimpleMoodDetector()

    # Get the mood
    detected_mood = detector.get_mood()

    # Pass the mood to your next model
    if detected_mood:
        print(f"[DEBUG] Final detected mood: {detected_mood}")
    else:
        print("[DEBUG] No mood was detected")


if __name__ == "__main__":
    main()
