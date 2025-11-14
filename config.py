"""
Configuration management for job-resume-matcher.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for accessing environment variables and settings."""

    # API Keys
    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '')
    INDEED_API_KEY = os.getenv('INDEED_API_KEY', '')
    ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID', '')
    ADZUNA_API_KEY = os.getenv('ADZUNA_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

    # Application Settings
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', '7'))

    # Data Paths
    BASE_DIR = Path(__file__).parent
    RESUME_DATASET = BASE_DIR / 'UpdatedResumeDataSet.csv'
    JOB_DATASET = BASE_DIR / 'data_Data_Analyst_in_USA_2023-05-13.csv'

    # API Endpoints
    JSEARCH_API_URL = "https://jsearch.p.rapidapi.com/search"
    JSEARCH_API_HOST = "jsearch.p.rapidapi.com"

    # DOL H1B Data
    DOL_API_URL = "https://api.dol.gov/v1/lca"
    DOL_PERFORMANCE_DATA_URL = "https://www.dol.gov/agencies/eta/foreign-labor/performance"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.RAPIDAPI_KEY:
            print("⚠️  WARNING: RAPIDAPI_KEY not set in .env file")
            print("   Copy .env.example to .env and add your API key")
            return False
        return True

    @classmethod
    def print_config(cls):
        """Print current configuration (masking sensitive data)."""
        print("=" * 60)
        print("Configuration Status")
        print("=" * 60)
        print(f"RapidAPI Key: {'✓ Set' if cls.RAPIDAPI_KEY else '✗ Not set'}")
        print(f"Indeed API Key: {'✓ Set' if cls.INDEED_API_KEY else '✗ Not set (optional)'}")
        print(f"Adzuna API: {'✓ Set' if cls.ADZUNA_API_KEY else '✗ Not set (optional)'}")
        print(f"OpenAI API Key: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not set (optional)'}")
        print(f"Sample Size: {cls.SAMPLE_SIZE}")
        print(f"Resume Dataset: {cls.RESUME_DATASET.exists() and '✓ Found' or '✗ Missing'}")
        print(f"Job Dataset: {cls.JOB_DATASET.exists() and '✓ Found' or '✗ Missing'}")
        print("=" * 60)

if __name__ == "__main__":
    # Test configuration
    Config.print_config()
