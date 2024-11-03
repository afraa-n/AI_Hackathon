import os
from pathlib import Path
import pygame
import time
from dotenv import load_dotenv
from openai import OpenAI


class NovaReader:
    def __init__(self):
        """Initialize OpenAI TTS system with Nova voice"""
        # Load environment variables
        load_dotenv()

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.output_file = "speech_output.mp3"
        pygame.mixer.init()

    def read_text(self, text):
        """
        Convert text to speech using OpenAI's Nova voice

        Args:
            text (str): Text to be read
        """
        try:
            print("\nConverting text to speech using Nova voice...")

            # Create speech file using OpenAI
            response = self.client.audio.speech.create(
                model="tts-1-hd",  # Using high-quality model
                voice="nova",  # Using Nova voice
                input=text,
            )

            # Save the audio file
            if os.path.exists(self.output_file):
                os.remove(self.output_file)

            response.stream_to_file(self.output_file)

            # Play the audio
            print("Playing audio...")
            pygame.mixer.music.load(self.output_file)
            pygame.mixer.music.play()

            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Cleanup
            if os.path.exists(self.output_file):
                os.remove(self.output_file)

            print("Finished reading.")

        except Exception as e:
            print(f"Error in text-to-speech conversion: {str(e)}")


def main():
    """Test the Nova voice reader"""
    try:
        reader = NovaReader()

        sample_text = (
            "I'm really sorry to hear that you're feeling this way today. It's completely normal to miss your family, "
            "especially when you're physically distant from them. It's okay to have those moments where the distance feels overwhelming, "
            "and it's important to acknowledge those emotions and give yourself the space to feel them. Remember that it's a sign of deep connection "
            "and love that you miss them so much.\n\n"
            "In times like these, it can be helpful to reach out to your family, even if it's just a quick call or message to let them know you're "
            "thinking of them. Sometimes sharing your feelings with them can provide comfort and support, even from afar. And don't forget to take "
            "care of yourself too—do something that brings you joy or relaxation, whether it's listening to music, going for a walk, or simply taking "
            "a moment to breathe.\n\n"
            "You're not alone in feeling this way, and it's okay to have days that are tough. Remember that brighter days are ahead, and your family's "
            "love is always with you, no matter the distance. Take care of yourself, and know that you are loved and valued.\n\n"
            "I have written a song for you, here it is."
        )

        reader.read_text(sample_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Make sure your OpenAI API key is correctly set in the .env file")


if __name__ == "__main__":
    main()
