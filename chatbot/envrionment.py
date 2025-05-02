import os

DEPLOY_MODE = os.getenv("DEPLOY_MODE", "local")
PORT = int(os.getenv("PORT", 5000))
DEBUG = DEPLOY_MODE == "local"

# Print confirmation message
print(f"\n✅ environment.py loaded with DEPLOY_MODE = {DEPLOY_MODE}")
