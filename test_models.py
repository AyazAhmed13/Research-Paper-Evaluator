import openai

# Configure the OpenAI client to point to OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # <-- Replace with your key
)

# List of models you want to test
models_to_test = [
    "nousresearch/hermes-3-405b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please respond with just the word 'OK'.",
                }
            ],
            # Optional but recommended headers to identify your app to OpenRouter
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",  # Your site URL
                "X-Title": "My Model Test Script",  # Your app name
            },
            max_tokens=10,
        )
        print(f"✅ Success: {model}")
        print(f"   Response: {completion.choices[0].message.content}\n")
    except Exception as e:
        print(f"❌ Error with {model}: {e}\n")