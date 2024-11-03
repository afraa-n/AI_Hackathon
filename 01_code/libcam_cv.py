import cv2
import numpy as np
from deepface import DeepFace
import time
from threading import Thread, Event
import speech_recognition as sr
import subprocess
import os


class SimpleMoodDetector:
    def __init__(self):
        print("[DEBUG] Initializing SimpleMoodDetector...")
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Set up speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("[DEBUG] Speech recognition initialized")

        # Threading events
        self.stop_process = Event()
        self.speaking = Event()

        # We'll only keep these emotions
        self.target_emotions = ["happy", "sad", "angry", "neutral"]
        self.detected_mood = None

        # Image paths
        self.image_path = "captured_image.jpg"
        self.resized_image_path = "resized_image.jpg"

    def capture_image(self):
        """Capture image using libcamera-still with reduced resolution"""
        try:
            print("[DEBUG] Capturing image with libcamera-still...")
            # Use libcamera-still to capture a lower resolution image
            subprocess.run(
                [
                    "libcamera-still",
                    "-n",  # No preview
                    "-t",
                    "1000",  # Timeout 1 second
                    "--immediate",  # Take picture immediately
                    "--width",
                    "640",  # Reduced width
                    "--height",
                    "480",  # Reduced height
                    "--rotation",
                    "0",  # Add rotation if needed
                    "--output",
                    self.image_path,
                ],
                check=True,
            )

            print(f"[DEBUG] Image captured successfully: {self.image_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[DEBUG] Error capturing image: {e}")
            return False

    def resize_image(self, max_size=640):
        """Resize image if it's too large"""
        try:
            print("[DEBUG] Resizing image...")
            # Read image
            img = cv2.imread(self.image_path)
            height, width = img.shape[:2]

            # Calculate new dimensions while maintaining aspect ratio
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                # Resize image
                resized_img = cv2.resize(img, (new_width, new_height))
                cv2.imwrite(self.resized_image_path, resized_img)
                print(f"[DEBUG] Image resized to {new_width}x{new_height}")
                return self.resized_image_path

            return self.image_path
        except Exception as e:
            print(f"[DEBUG] Error resizing image: {e}")
            return self.image_path

    def detect_mood(self, image_path):
        """Detect mood using DeepFace with memory optimization"""
        try:
            print("[DEBUG] Starting mood detection with DeepFace...")
            # DeepFace analyze returns emotions with confidence scores
            result = DeepFace.analyze(
                image_path,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )

            # Get the emotion with highest confidence score
            emotion = result[0]["dominant_emotion"]
            print(f"[DEBUG] DeepFace detected emotion: {emotion}")
            print(f"[DEBUG] All emotions detected: {result[0]['emotion']}")

            # Map to our target emotions, defaulting to neutral
            if emotion in self.target_emotions:
                print(f"[DEBUG] Returning target emotion: {emotion}")
                return emotion
            print("[DEBUG] Emotion not in target list, defaulting to neutral")
            return "neutral"

        except Exception as e:
            print(f"[DEBUG] Error in mood detection: {e}")
            return None
        finally:
            # Clear memory
            import gc

            gc.collect()

    def process_image(self):
        """Process the captured image"""
        try:
            # Read the captured image
            frame = cv2.imread(self.image_path)
            if frame is None:
                print("[DEBUG] Failed to read captured image")
                return False

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using OpenCV
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # If face detected, process mood
            if len(faces) > 0:
                print(f"[DEBUG] Detected {len(faces)} faces")
                # Process resized image for mood detection
                image_to_process = self.resize_image()
                self.detected_mood = self.detect_mood(image_to_process)
                return True
            else:
                print("[DEBUG] No faces detected in the image")
                return False

        except Exception as e:
            print(f"[DEBUG] Error processing image: {e}")
            return False

    def listen_for_speech(self):
        """Listen for speech and trigger image capture"""
        print("[DEBUG] Starting speech recognition")
        with self.microphone as source:
            print("[DEBUG] Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("[DEBUG] Ambient noise adjustment complete")

            while not self.stop_process.is_set():
                try:
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=5)
                    print("[DEBUG] Speech detected!")

                    # Capture and process image
                    if self.capture_image():
                        if self.process_image():
                            self.stop_process.set()
                            break

                    time.sleep(0.5)

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

            # Wait for mood detection
            while not self.stop_process.is_set():
                time.sleep(0.1)

        finally:
            # Clean up
            print("[DEBUG] Cleaning up resources")
            # Remove temporary image files
            for path in [self.image_path, self.resized_image_path]:
                if os.path.exists(path):
                    os.remove(path)

        return self.detected_mood


def main():
    print("[DEBUG] Starting main program")
    # Initialize the detector
    detector = SimpleMoodDetector()

    # Get the mood
    detected_mood = detector.get_mood()

    # Process the detected mood
    if detected_mood:
        print(f"[DEBUG] Final detected mood: {detected_mood}")
    else:
        print("[DEBUG] No mood was detected")


if __name__ == "__main__":
    main()
