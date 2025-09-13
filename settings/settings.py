from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    LLM_MODEL: str = os.getenv("LLM_MODEL")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")