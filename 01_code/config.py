from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UDIO_KEY = os.getenv("UDIO_KEY")
HUME_API_KEY = os.getenv("HUME__API_KEY")

# You can raise an error if the key isn't found for security
if OPENAI_API_KEY is None and UDIO_KEY is None:
    raise ValueError("OpenAI & UDIO API key not found. Please set it in the .env file.")
elif UDIO_KEY is None:
    raise ValueError("UDIO API key not found. Please set it in the .env file.")
elif OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")
