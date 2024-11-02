#!/usr/bin/env python3

import cv2
from deepface import DeepFace
import numpy as np
import time


class EmotionDetector:
    def __init__(self):
        """Initialize the EmotionDetector class."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def initialize_camera(self):
        """Initialize the camera and return the video capture object."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open camera")
        # Set camera properties for better stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def detect_face(self, frame):
        """Detect if there's a face in the frame using OpenCV's face detector."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0

    def analyze_emotion(self, frame):
        """Analyze the emotion in a frame using DeepFace."""
        try:
            result = DeepFace.analyze(
                img_path=frame, actions=["emotion"], enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            return result["dominant_emotion"]
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None

    def run(self):
        """Main loop for emotion detection."""
        try:
            while True:  # Outer loop for camera reconnection
                try:
                    # Initialize camera
                    cap = self.initialize_camera()
                    print("Starting live emotion detection. Press 'q' to quit.")

                    while True:  # Inner loop for frame processing
                        # Capture frame
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to grab frame, attempting to reconnect...")
                            break  # Break inner loop to reinitialize camera

                        # Show live feed
                        cv2.imshow("Live Feed", frame)

                        # Detect face in frame
                        if self.detect_face(frame):
                            print("\nFace detected! Analyzing emotion...")

                            # Analyze emotion on current frame
                            emotion = self.analyze_emotion(frame)
                            if emotion:
                                print(f"Detected emotion: {emotion}")

                                # Display the emotion on the frame
                                cv2.putText(
                                    frame,
                                    f"Emotion: {emotion}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2,
                                )
                                cv2.imshow("Analyzed Frame", frame)

                        # Check for 'q' key to quit
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("\nQuitting...")
                            return

                        # Small delay to prevent CPU overload
                        time.sleep(0.1)

                except Exception as e:
                    print(f"Camera error: {str(e)}")
                    print("Attempting to reconnect in 2 seconds...")
                    time.sleep(2)
                finally:
                    # Clean up current camera instance
                    if "cap" in locals():
                        cap.release()

        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            # Final cleanup
            cv2.destroyAllWindows()
            if "cap" in locals():
                cap.release()


def main():
    """Main entry point of the application."""
    detector = EmotionDetector()
    detector.run()


if __name__ == "__main__":
    main()
