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
        return cap

    def detect_face(self, frame):
        """
        Detect if there's a face in the frame using OpenCV's face detector.

        Args:
            frame: Image frame from the video capture

        Returns:
            bool: True if face is detected, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0

    def analyze_emotion(self, frame):
        """
        Analyze the emotion in a frame using DeepFace.

        Args:
            frame: Image frame to analyze

        Returns:
            str: The dominant emotion detected, or None if analysis fails
        """
        try:
            result = DeepFace.analyze(
                img_path=frame, actions=["emotion"], enforce_detection=False
            )

            # Handle both list and dictionary results
            if isinstance(result, list):
                result = result[0]

            return result["dominant_emotion"]
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return None

    def run(self):
        """Main loop for emotion detection."""
        try:
            # Initialize camera
            cap = self.initialize_camera()
            print("Starting live emotion detection. Press 'q' to quit.")

            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Show live feed
                cv2.imshow("Live Feed", frame)

                # Detect face in frame
                if self.detect_face(frame):
                    print("Face detected! Analyzing emotion...")

                    # Pause for a moment to ensure a clear frame
                    time.sleep(0.5)

                    # Capture one more frame for analysis
                    ret, analysis_frame = cap.read()
                    if ret:
                        # Analyze emotion
                        emotion = self.analyze_emotion(analysis_frame)
                        if emotion:
                            print(f"Detected emotion: {emotion}")

                            # Display the emotion on the frame
                            cv2.putText(
                                analysis_frame,
                                f"Emotion: {emotion}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )
                            cv2.imshow("Analyzed Frame", analysis_frame)

                            # Wait for 2 seconds to show the result
                            time.sleep(2)

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Clean up
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise


def main():
    """Main entry point of the application."""
    detector = EmotionDetector()
    detector.run()


if __name__ == "__main__":
    main()
