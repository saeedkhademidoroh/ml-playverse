from flask import Flask, request, jsonify
from chatbot import ask_openai
from authenticate import require_auth
from environment import DEBUG, PORT

app = Flask(__name__)

try:
    from retrieve import retrieve_context
    from augments import augment_prompt
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False

@app.route("/chat", methods=["POST"])
def chat():
    auth_error = require_auth()
    if auth_error:
        return auth_error

    data = request.json
    prompt = data.get("message", "")
    if not prompt:
        return jsonify({"error": "No message provided"}), 400

    if RAG_ENABLED:
        context = retrieve_context(prompt)
        prompt = augment_prompt(prompt, context)

    reply = ask_openai(prompt)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT)

# Print confirmation message
print("\n✅ main.py successfully executed")
