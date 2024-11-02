import cv2
from fer import FER
import numpy as np


class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector"""
        self.detector = FER(mtcnn=True)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def run_detection(self):
        """Run the live emotion detection"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Could not open camera")

        print("Starting emotion detection... Press 'q' to quit")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Detect emotions in the frame
            emotions = self.detector.detect_emotions(frame)

            # Process detected emotions
            if emotions:
                for emotion in emotions:
                    # Get the bounding box
                    x, y, w, h = [int(coord) for coord in emotion["box"]]

                    # Get the dominant emotion
                    emotions_dict = emotion["emotions"]
                    dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])[0]

                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Add text with emotion
                    cv2.putText(
                        frame,
                        dominant_emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            # Display the frame
            cv2.imshow("Emotion Detection", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        detector = EmotionDetector()
        detector.run_detection()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
