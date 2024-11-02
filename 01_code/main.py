# main.py
from hume_analyzer import HumeAnalyzer, format_context_from_hume
from prompt_engineering import generate_music_details
from song_generator import generate_song_request, poll_song_status
from config import OPENAI_API_KEY, UDIO_KEY, HUME_API_KEY
import asyncio
import openai

async def process_interaction(video_path=None, audio_path=None):
    """Process a single interaction and generate a song"""
    # Initialize API keys
    openai.api_key = OPENAI_API_KEY
    udio_token = UDIO_KEY
    
    try:
        # Initialize HumeAI analyzer
        hume_analyzer = HumeAnalyzer(HUME_API_KEY)
        
        # Analyze user interaction
        hume_result = await hume_analyzer.analyze_interaction(
            video_path=video_path,
            audio_path=audio_path
        )
        
        # Format the context based on HumeAI analysis
        context = format_context_from_hume(hume_result)
        
        # Generate music details
        generated_prompt, singer_name, music_genre = generate_music_details(context, OPENAI_API_KEY)
        
        if generated_prompt and singer_name and music_genre:
            print("Analysis Results:")
            print("Emotions detected:", hume_result['emotions'])
            print("Overall sentiment:", hume_result['overall_sentiment'])
            print("Transcript:", hume_result['transcript'])
            print("\nGenerated Music Details:")
            print("Prompt:", generated_prompt)
            print("Singer:", singer_name)
            print("Genre:", music_genre)
            
            # Create GPT description prompt
            gpt_description_prompt = (
                f"The song should feature the singing voice closest to {singer_name} "
                f"in the style of {music_genre}. {generated_prompt}. "
                "Limit the song generated to be as shortest in length as possible."
            )
            
            # Generate the song
            workId = generate_song_request(udio_token, generated_prompt, gpt_description_prompt)
            if workId:
                audio_url = poll_song_status(udio_token, workId)
                if audio_url:
                    print("Generated Song URL:", audio_url)
                    return audio_url, hume_result
        
        print("Failed to generate music details.")
        return None, hume_result
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

async def main():
    # For testing, you can replace these paths with your actual video/audio files
    test_video = None  # "path_to_video.mp4"
    test_audio = "path_to_audio.wav"
    
    audio_url, analysis = await process_interaction(test_video, test_audio)
    return audio_url, analysis

if __name__ == "__main__":
    audio_url, analysis = asyncio.run(main())