from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
from pydub import AudioSegment
from pydub.playback import play


# Function to generate and play audio from text
def generate_and_play_audio(text_prompt):
    # Download and load all models
    preload_models()

    # Generate audio from text
    audio_array = generate_audio(text_prompt)

    # Convert the numpy array to a format compatible with pydub
    audio_segment = AudioSegment(
        data=audio_array.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=audio_array.dtype.itemsize,
        channels=1,  # Assuming mono audio
    )

    # Play the generated audio
    play(audio_segment)

    print("Audio has been generated and played.")


# Main function to test the text-to-speech
if __name__ == "__main__":
    # Example text prompt
    text_prompt = """
        Hello, my name is Suno. And, uh — and I like pizza. [laughs]
        But I also have other interests such as playing tic tac toe.
    """

    # Generate and play the audio file directly
    generate_and_play_audio(text_prompt)
