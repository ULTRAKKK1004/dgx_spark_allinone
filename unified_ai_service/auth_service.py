import secrets
import json
import os

TOKEN_FILE = "/home/yanus/unified_ai_service/api_tokens.json"

def get_tokens():
    if not os.path.exists(TOKEN_FILE):
        return {}
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)

def save_tokens(tokens):
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)

def generate_new_token(label: str):
    tokens = get_tokens()
    new_token = f"dgx_{secrets.token_urlsafe(32)}"
    tokens[new_token] = label
    save_tokens(tokens)
    return new_token

def verify_token(token: str):
    tokens = get_tokens()
    return token in tokens
