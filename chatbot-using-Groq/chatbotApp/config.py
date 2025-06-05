import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "default_model": "mixtral-8x7b-32768",
        "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", "2000")),
        "top_p": float(os.getenv("MODEL_TOP_P", "0.9")),
        "api_key": os.getenv("GROQ_API_KEY")
    }
}

# Default model to use
DEFAULT_MODEL = "groq"

# Vector store settings
VECTOR_STORE_CONFIG = {
    "similarity_threshold": 0.75,
    "max_chunks": 3,
    "search_k": 5,
    "fetch_k": 10
}

# Session settings
SESSION_CONFIG = {
    "default_session": "default",
    "session_timeout": 3600  # 1 hour in seconds
}

# File storage settings
STORAGE_CONFIG = {
    "upload_dir": "media/uploads",
    "vector_dir": "chroma_db",
    "allowed_extensions": [".pdf"]
}

# API settings
API_CONFIG = {
    "stream_timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1  # seconds
}

def get_model_config(model_name: str = None) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS["groq"]

def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration"""
    return VECTOR_STORE_CONFIG

def get_session_config() -> Dict[str, Any]:
    """Get session configuration"""
    return SESSION_CONFIG

def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration"""
    return STORAGE_CONFIG

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return API_CONFIG 