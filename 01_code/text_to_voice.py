from gtts import gTTS
import os
import time
import pygame
from pygame import mixer


class ParagraphReader:
    def __init__(self):
        """Initialize the text-to-speech system"""
        self.output_file = "speech_output.mp3"
        pygame.mixer.init()

    def read_paragraph(self, text, language="en", speed="normal"):
        """
        Read a paragraph of text

        Args:
            text (str): The text to be read
            language (str): Language code (default: 'en' for English)
            speed (str): 'slow' or 'normal' speed of speech
        """
        try:
            print(f"\nReading text: \n'{text}'\n")

            # Set speech speed
            slow_speech = True if speed == "slow" else False

            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=slow_speech)

            # Remove existing audio file if it exists
            if os.path.exists(self.output_file):
                os.remove(self.output_file)

            # Save the audio file
            print("Converting text to speech...")
            tts.save(self.output_file)

            # Play the audio file
            print("Playing audio...")
            pygame.mixer.music.load(self.output_file)
            pygame.mixer.music.play()

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Clean up
            if os.path.exists(self.output_file):
                os.remove(self.output_file)

            print("Finished reading.")

        except Exception as e:
            print(f"Error in text-to-speech conversion: {str(e)}")

    def read_with_emotion(self, text):
        """
        Read text with appropriate pauses and pacing based on content

        Args:
            text (str): The text to be read with emotion
        """
        # Add small pauses after punctuation for more natural speech
        text = text.replace(".", "... ")
        text = text.replace("!", "... ")
        text = text.replace("?", "... ")

        self.read_paragraph(text)


def main():
    """Test the paragraph reader with sample text"""
    reader = ParagraphReader()

    # Test with your example text
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

    reader.read_with_emotion(sample_text)

    # Uncomment to test more examples
    # for text in more_examples:
    #     time.sleep(1)  # Pause between paragraphs
    #     reader.read_with_emotion(text)


if __name__ == "__main__":
    main()
