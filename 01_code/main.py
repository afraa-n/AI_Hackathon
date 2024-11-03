# main.py

from prompt_engineering import generate_music_details
from song_generator import generate_song_request, poll_song_status
from config import OPENAI_API_KEY, UDIO_KEY
import openai

# Set up the API key for OpenAI
openai.api_key = OPENAI_API_KEY
udio_api_key = UDIO_KEY


def main():
    # Set your API token for both OpenAI and song generation services
    openai_api_token = openai.api_key
    udio_token = udio_api_key
    # Define context for generating music details
    context = (
        "I had such a fantastic day today! Everything seemed to go right, and I couldn’t stop smiling. "
        "I finally finished a project I’ve been working on for weeks, and my boss praised my efforts. "
        "To celebrate, we had a little party at work, complete with cake and laughter while playing my favorite singer Coldplay!. "
        "I feel so full of energy and happiness right now, like I could dance all night! "
    )

    # Generate the prompt, singer name, and music genre based on the context
    generated_prompt, singer_name, music_genre = generate_music_details(
        context, openai_api_token
    )

    # Display generated details
    if generated_prompt and singer_name and music_genre:
        print("Generated Prompt: ", generated_prompt)
        print("Singer Name:", singer_name)
        print("Music Genre:", music_genre)

        # Combine details for song generation
        gpt_description_prompt = f"The song should feature the singing voice closest to {singer_name} in the style of {music_genre}. {generated_prompt}. Limit the song generated to be as shortest in length as possible."

        # Step 1: Start the song generation
        workId = generate_song_request(
            udio_token, generated_prompt, gpt_description_prompt
        )

        if workId:
            # Step 2: Poll the status until the song is ready
            audio_url = poll_song_status(udio_token, workId)
            if audio_url:
                print("Generated Song URL:", audio_url)
    else:
        print("Failed to generate music details.")


if __name__ == "__main__":
    main()
