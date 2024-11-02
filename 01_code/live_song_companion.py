from live_hume_analyzer import LiveHumeAnalyzer
from prompt_engineering import generate_music_details
from song_generator import generate_song_request, poll_song_status
from config import OPENAI_API_KEY, UDIO_KEY, HUME_API_KEY
import asyncio
import time

class LiveSongCompanion:
    def __init__(self):
        self.hume_analyzer = LiveHumeAnalyzer(HUME_API_KEY)
        self.is_running = False
        self.emotion_threshold = 0.7  # Threshold for significant emotion change
        self.last_analysis = None
        self.last_song_time = 0
        self.min_song_interval = 60  # Minimum seconds between songs
        
    async def start(self):
        """Start the live song companion"""
        self.is_running = True
        await self.hume_analyzer.start_recording()
        await self._analysis_loop()  # Start the analysis loop
        
    async def stop(self):
        """Stop the live song companion"""
        self.is_running = False
        await self.hume_analyzer.stop_recording()
        
    async def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            current_emotions = self.hume_analyzer.get_latest_emotions()
            
            if current_emotions and await self._should_generate_song(current_emotions):
                # Generate context from emotions
                context = self._create_context(current_emotions)
                
                # Generate and play song
                await self._generate_song(context)
                
                self.last_analysis = current_emotions
                self.last_song_time = time.time()
                
            await asyncio.sleep(1)  # Check every second
            
    async def _should_generate_song(self, current_emotions):
        """Determine if we should generate a new song based on emotion changes"""
        if not self.last_analysis:
            return True
            
        # Check if enough time has passed since last song
        if time.time() - self.last_song_time < self.min_song_interval:
            return False
            
        # Check for significant changes in emotions
        for mode in ['face', 'prosody']:
            if mode in current_emotions and mode in self.last_analysis:
                for emotion, score in current_emotions[mode].items():
                    last_score = self.last_analysis[mode].get(emotion, 0)
                    if abs(score - last_score) > self.emotion_threshold:
                        return True
                        
        return False
        
    def _create_context(self, emotions):
        """Create context for song generation from emotions"""
        face_emotions = sorted(
            emotions['face'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        prosody_emotions = sorted(
            emotions['prosody'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        context = (
            f"Based on facial expressions, the person is feeling primarily "
            f"{', '.join(f'{emotion} ({score:.2f})' for emotion, score in face_emotions)}. "
            f"Their voice indicates emotions of "
            f"{', '.join(f'{emotion} ({score:.2f})' for emotion, score in prosody_emotions)}. "
            f"They said: {emotions['transcript'].strip()}"
        )
        
        return context
        
    async def _generate_song(self, context):
        """Generate a song based on the emotional context"""
        try:
            # Generate music details
            generated_prompt, singer_name, music_genre = generate_music_details(context, OPENAI_API_KEY)
            
            if generated_prompt and singer_name and music_genre:
                print("\nGenerating song based on current emotions...")
                print("Context:", context)
                print("Singer:", singer_name)
                print("Genre:", music_genre)
                
                # Create GPT description prompt
                gpt_description_prompt = (
                    f"The song should feature the singing voice closest to {singer_name} "
                    f"in the style of {music_genre}. {generated_prompt}. "
                    "Limit the song generated to be as shortest in length as possible."
                )
                
                # Generate the song
                workId = await generate_song_request(UDIO_KEY, generated_prompt, gpt_description_prompt)
                if workId:
                    audio_url = await poll_song_status(UDIO_KEY, workId)
                    if audio_url:
                        print("New song generated:", audio_url)
                        # Here you can add code to play the song through your preferred audio player
                        
        except Exception as e:
            print(f"Error generating song: {e}")

async def main():
    companion = LiveSongCompanion()
    
    try:
        print("Starting Live Song Companion...")
        print("Press Ctrl+C to stop")
        await companion.start()
        
    except KeyboardInterrupt:
        print("\nStopping Live Song Companion...")
        await companion.stop()

if __name__ == "__main__":
    asyncio.run(main())