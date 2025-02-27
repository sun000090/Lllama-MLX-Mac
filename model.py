from langchain_groq import ChatGroq
import httpx
from logs.logger import Logger
import os

logss = Logger().get_logger('Logger')

class LLMModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name, api_key, temperature):
        if not hasattr(self, 'initialized'): 
            self.model_name = model_name
            self.api_key = api_key
            self.temperature = temperature
            self.initialized = True

    def load_model(self):
        try:
            llm = ChatGroq(
                model_name=self.model_name,
                groq_api_key=self.api_key,
                temperature=self.temperature,
                timeout=httpx.Timeout(60.0, read=10.0, write=30.0, connect=20.0)
            )
            logss.info("Model successfully loaded")
            return llm
        except Exception as e:
            logss.error(f"Model loading issue: {e}")
            return None
