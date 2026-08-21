import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def get_backend_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url:
        raise EnvironmentError("SUPABASE_URL not found in environment variables.")
    if not key:
        raise EnvironmentError("SUPABASE_SERVICE_KEY not found in environment variables.")

    return create_client(url, key)


def get_frontend_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url:
        raise EnvironmentError("SUPABASE_URL not found in environment variables.")
    if not key:
        raise EnvironmentError("SUPABASE_ANON_KEY not found in environment variables.")

    return create_client(url, key)