import openai

# Set up OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"


def generate_prompt_for_gpt(user_input):
    # Craft a prompt to instruct GPT-3.5 Turbo to respond appropriately
    prompt = (
        f'User Input: "{user_input}"\n\n'
        "Respond as an empathetic and friendly conversational partner. Your response should be engaging, supportive, "
        "and continuous for approximately one minute, speaking as a comforting friend who listens, reassures, and "
        "validates the user's emotions or experience. Make sure to conclude the response with: "
        "'I have written a song for you, here it is.'"
    )
    return prompt


def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an empathetic and friendly AI that responds thoughtfully to user inputs.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,  # Adjust token limit to control the response length for approximately one minute of speaking
        temperature=0.7,  # Adjust temperature for creative and engaging responses
    )

    # Extract the response text
    generated_response = response["choices"][0]["message"]["content"].strip()

    return generated_response


def main():
    user_input = "Today just wasn’t my day. I woke up feeling down, missing my family, and that feeling of being so far from everyone I care about really hit hard."
    prompt = generate_prompt_for_gpt(user_input)
    response = get_gpt_response(prompt)

    print("GPT Response:")
    print(response)


if __name__ == "__main__":
    main()
