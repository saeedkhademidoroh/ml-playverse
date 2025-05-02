def augment_prompt(user_input: str, context: str) -> str:
    return f"Context: {context}\n\nUser: {user_input}"

# Print confirmation message
print("\n✅ augments.py successfully executed")
