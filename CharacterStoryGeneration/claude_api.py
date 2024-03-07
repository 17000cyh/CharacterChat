import anthropic

anthropic.Anthropic().messages.create(
    api_key="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",
    model="claude-2.1",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)