# Bobcat! - Your Singing AI Companion 🎵🤖
Bobcat is not just a gadget—it's your personal, empathetic AI friend that listens, responds, and creates songs tailored to your emotions and personal story. Conceived during Duke AI Hackathon 2024, this innovative project merges art, technology, and human connection.

🏆 **2nd Grand Prize Winner**  
🏆 **Best Art, Media & Design Prize**  
Awarded at the **Duke AI Hackathon 2024**

**Sponsors**: OpenAI, Microsoft, decorX, 2ndF, Fiscus, Inquisite, ticketdude, Quotient AI, Red Bull

## Presentation & WriteUp
You can find our project presentation here: [Introducing Bobcat: The Singing AI Companion Friend](AI_Hackathon/Introducing%20Bobcat%20The%20Singing%20AI%20Companion%20Friend.pdf)

You can find our project Writeup here: [Bobcat: Your AI Companion That Sings Your Story](Bobcat_%20Your%20AI%20Companion%20That%20Sings%20Your%20Story.pdf)

You can find our project Demo Video here: [Devpost: 2nd Grand Prize](https://devpost.com/software/bobcat-your-singing-ai-companion?ref_content=my-projects-tab&ref_feature=my_projects)

![bobcat logo (2)](https://github.com/user-attachments/assets/57a96817-8771-4aee-a4e5-2e4d1502c63d)

## Overview

Bobcat bridges the gap between practical tech and human warmth, delivering real-time, personalized emotional support through song and conversation. Packaged in a charming 3D-printed shell, it represents the intersection of creative AI, empathy, and user-centric design.

## Key Features

- **Multimodal Sentiment Analysis**: Combines **DeepFace** for facial recognition and **speech-based sentiment analysis** for a comprehensive understanding of your mood.
- **Voice Transcription**: Translates spoken input into text using **Speech Recognition**, enabling contextual awareness.
- **Empathetic AI Interaction**: Chat responses powered by **OpenAI GPT-3.5 Turbo** and **Text-to-Speech (TTS-1-HD)** mimic natural, human-like conversation.
- **Dynamic Song Generation**: Utilizes **UDIO** to compose and sing personalized songs that align with your emotional state, creating a unique, mood-reflective experience.
- **Portable & Engaging Design**: Encased in a 3D-printed body with a friendly aesthetic that invites interaction.
  
## Technical Architecture

### Hardware
- **Raspberry Pi 4B**: The processing hub that powers Bobcat.
- **Arducam IMX708 Camera**: Captures facial expressions for mood detection.
- **Audio Core HAT WM8060**: Delivers high-quality sound for an immersive experience.
- **Integrated Microphone and Speakers**: Enable voice interaction and audio feedback.
- **Lithium Battery Module**: Ensures portability and long-lasting operation.

### Software Stack
- **DeepFace**: Facial recognition for real-time mood analysis.
- **Speech Recognition Library**: Transcribes voice input to text.
- **OpenAI GPT-3.5 Turbo**: Empathetic, AI-driven conversation.
- **Text-to-Speech (TTS-1-HD)**: Natural, expressive vocal synthesis.
- **UDIO**: Generates and performs songs tailored to the detected emotional state.

### Project Pipeline
Bobcat’s real-time operations rely on seamless integration:
1. **Facial and Voice Input** ➡️ Mood Detection (DeepFace + Sentiment Analysis)
2. **Transcription & Interpretation** ➡️ Chat Response (Speech Recognition + GPT-3.5 Turbo + TTS)
3. **Song Composition** ➡️ Performance Output (UDIO)

![CI_CD pipeline llm_RAG (1)](https://github.com/user-attachments/assets/28ea490d-214c-4824-9dc9-b766fe02bc87)

## Novelty & Value Proposition

Bobcat stands out as a truly innovative blend of empathy, technology, and creativity:
- **Novelty**: Real-time, multimodal sentiment analysis paired with song generation offers a uniquely interactive, personalized experience.
- **Value**: In an era where digital interactions often lack warmth, Bobcat embodies a shift toward emotionally intelligent tech designed to resonate with users and foster genuine emotional connection.

## Use Cases
- **Children**: An interactive, musical friend for play and comfort.
- **Adults**: A stress-relieving, empathetic companion.
- **Seniors**: A comforting partner that brings joy through personalized songs.

## Future Developments
- **Cloud Integration (AWS Lex & Polly)**: Expand Bobcat’s reach to digital platforms for seamless interaction.
- **Social Media Sharing**: Enable users to share their songs and experiences.
- **Deeper Personalization**: Enhance responses for richer, more tailored user engagement.

## Development Highlights
Creating Bobcat involved overcoming significant technical challenges:
- **48-Hour Hackathon Constraints**: Integrated hardware-software synergy and real-time functionality within limited time.
- **Precision Engineering**: Ensured smooth, on-the-go operation through meticulous design and testing.

## Contributors
- **Afraa Noureen**
- **Charles Xie**
- **Tina Yi**
- **Bob Zhang**

## Acknowledgments
Special thanks to Duke University for the hackathon platform and to our sponsors for their support. Bobcat is a testament to innovation, teamwork, and the power of creative AI.
