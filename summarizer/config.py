import os
from openai import AsyncOpenAI

if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
    raise Exception('Missing OPENAI_API_KEY environment variable')

if 'OPENAI_API_BASE' not in os.environ or not os.environ['OPENAI_API_BASE']:
    raise Exception('Missing OPENAI_API_BASE environment variable')

OPENAPI_TOKEN = os.environ['OPENAI_API_KEY']

ai_client = AsyncOpenAI(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=OPENAPI_TOKEN
)

SUMMARIZER_MODEL = os.environ.get('SUMMARIZER_MODEL', 'google/gemini-2.5-flash-lite-preview-09-2025')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
