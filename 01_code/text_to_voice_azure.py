import azure.cognitiveservices.speech as speechsdk
import time


class ParagraphReader:
    def __init__(self, subscription_key, region):
        """
        Initialize the Azure Text-to-Speech system

        Args:
            subscription_key (str): Your Azure Speech Service subscription key
            region (str): Your Azure region (e.g., 'eastus')
        """
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        # Set a more natural voice
        # You can choose different voices:
        # 'en-US-JennyMultilingualNeural' - Natural female voice
        # 'en-US-GuyNeural' - Natural male voice
        # 'en-US-AriaNeural' - Natural female voice
        self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    def read_paragraph(self, text, style="empathetic"):
        """
        Read a paragraph of text using Azure's natural voices

        Args:
            text (str): The text to be read
            style (str): Speaking style ('empathetic', 'chat', 'cheerful', etc.)
        """
        try:
            print(f"\nReading text: \n'{text}'\n")

            # Add SSML for more natural speech with emotion
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
                   xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <mstts:express-as style="{style}">
                        {text}
                    </mstts:express-as>
                </voice>
            </speak>
            """

            # Create speech synthesizer
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config
            )

            # Synthesize and play the speech
            print("Converting text to speech...")
            result = speech_synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesis completed successfully.")
            else:
                print(f"Speech synthesis failed: {result.reason}")

        except Exception as e:
            print(f"Error in text-to-speech conversion: {str(e)}")

    def read_with_emotion(self, text):
        """
        Read text with appropriate emotional style

        Args:
            text (str): The text to be read with emotion
        """
        # Detect emotional content and choose appropriate style
        if any(word in text.lower() for word in ["sorry", "sad", "miss", "tough"]):
            style = "empathetic"
        elif any(word in text.lower() for word in ["happy", "great", "wonderful"]):
            style = "cheerful"
        else:
            style = "chat"

        self.read_paragraph(text, style=style)


def main():
    """Test the paragraph reader with sample text"""
    # Replace these with your Azure credentials
    SUBSCRIPTION_KEY = "your_subscription_key"
    REGION = "your_region"

    try:
        reader = ParagraphReader(SUBSCRIPTION_KEY, REGION)

        sample_text = (
            "I'm really sorry to hear that you're feeling this way today. "
            "It's completely normal to miss your family, especially when "
            "you're physically distant from them. It's okay to have those "
            "moments where the distance feels overwhelming, and it's important "
            "to acknowledge those emotions and give yourself the space to feel them. "
            "Remember that it's a sign of deep connection and love that you miss them so much."
        )

        reader.read_with_emotion(sample_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
