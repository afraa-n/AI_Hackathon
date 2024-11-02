from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav


# Function to generate and save audio from text
def generate_and_save_audio(text_prompt, output_filename="bark_generation.wav"):
    # Download and load all models
    preload_models()

    # Generate audio from text
    audio_array = generate_audio(text_prompt)

    # Save the audio to disk
    write_wav(output_filename, SAMPLE_RATE, audio_array)

    print(f"Audio has been generated and saved as {output_filename}")


# Main function to test the text-to-speech
if __name__ == "__main__":
    # Example text prompt
    text_prompt = """
        Hello, my name is Suno. And, uh — and I like pizza. [laughs]
        But I also have other interests such as playing tic tac toe.
    """

    # Generate and save the audio file
    generate_and_save_audio(text_prompt)
