import speech_recognition as sr
import os


def get_user_text():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Please speak now...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust within the with block
            audio_data = recognizer.listen(source)  # Capture within the with block
            text = recognizer.recognize_google(audio_data)
            # print("Transcription:", text)
    except AssertionError as e:
        print(f"Error adjusting ambient noise: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return text


def main():
    """Main function to run speech recognition"""
    try:
        print("Starting speech recognition...")
        print("Speak something (or say 'quit' to exit)")

        while True:
            # Get speech input
            user_text = get_user_text()

            # Print what was recognized
            print(f"You said: {user_text}")

            # Check if user wants to quit
            if user_text.lower() in ["quit", "exit", "stop", "goodbye"]:
                print("Ending speech recognition...")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
    finally:
        print("Speech recognition ended")


if __name__ == "__main__":
    main()
