import cv2


def test_camera():
    print("Starting camera test...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    print("Camera opened successfully")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed")


if __name__ == "__main__":
    print("start")
    test_camera()
