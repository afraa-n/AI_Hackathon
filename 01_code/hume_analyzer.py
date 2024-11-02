# hume_analyzer.py
from hume.api import HumeApi
import numpy as np
import asyncio

class HumeAnalyzer:
    def __init__(self, api_key):
        self.client = HumeApi(api_key)
    
    async def analyze_interaction(self, video_path=None, audio_path=None):
        """Analyze both facial expressions and voice prosody from video/audio input"""
        result = {
            'emotions': {},
            'transcript': '',
            'overall_sentiment': 0.0
        }
        
        try:
            # Process video if available
            if video_path:
                face_predictions = await self.client.run_face_model(
                    video_path,
                    notify_every_n=None
                )
                
                if face_predictions:
                    for prediction in face_predictions:
                        if 'emotions' in prediction:
                            for emotion in prediction['emotions']:
                                if emotion['name'] not in result['emotions']:
                                    result['emotions'][emotion['name']] = []
                                result['emotions'][emotion['name']].append(emotion['score'])
            
            # Process audio if available
            if audio_path:
                prosody_predictions = await self.client.run_prosody_model(
                    audio_path,
                    notify_every_n=None
                )
                
                if prosody_predictions:
                    for prediction in prosody_predictions:
                        if 'emotions' in prediction:
                            for emotion in prediction['emotions']:
                                if emotion['name'] not in result['emotions']:
                                    result['emotions'][emotion['name']] = []
                                result['emotions'][emotion['name']].append(emotion['score'])
                        
                        if 'text' in prediction:
                            result['transcript'] += prediction['text'] + ' '
            
            # Average the emotion scores
            for emotion in result['emotions']:
                result['emotions'][emotion] = float(np.mean(result['emotions'][emotion]))
            
            # Calculate overall sentiment
            positive_emotions = ['joy', 'excitement', 'satisfaction']
            negative_emotions = ['sadness', 'anger', 'fear']
            
            sentiment_score = 0.0
            for emotion in positive_emotions:
                if emotion in result['emotions']:
                    sentiment_score += result['emotions'][emotion]
            for emotion in negative_emotions:
                if emotion in result['emotions']:
                    sentiment_score -= result['emotions'][emotion]
                    
            result['overall_sentiment'] = sentiment_score
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise
            
        return result

def format_context_from_hume(hume_result):
    """Convert HumeAI analysis results into context for song generation"""
    emotions_desc = []
    for emotion, score in hume_result['emotions'].items():
        if score > 0.5:  # Only include significant emotions
            emotions_desc.append(f"{emotion} ({score:.2f})")
    
    context = (
        f"Based on the analysis of voice and facial expressions, the person is feeling: {', '.join(emotions_desc)}. "
        f"Their overall emotional state appears to be {('positive' if hume_result['overall_sentiment'] > 0 else 'negative')} "
        f"with a sentiment score of {hume_result['overall_sentiment']:.2f}. "
        f"They expressed the following: {hume_result['transcript'].strip()}"
    )
    
    return context