import os
# from flask import request, jsonify

VALID_TOKEN = os.getenv("API_AUTH_TOKEN", "fallback-token")

def require_auth():
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {VALID_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401
    return None

# Print confirmation message
print("\n✅ authenticate.py successfully executed")
