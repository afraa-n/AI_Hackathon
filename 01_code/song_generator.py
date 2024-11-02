# song_generator.py

import requests
import json
import time


# Function to start the song generation process
def generate_song_request(
    api_token,
    prompt,
    gpt_description_prompt,
    model="chirp-v3.0",
    custom_mode=False,
    make_instrumental=False,
):
    url = "https://udioapi.pro/api/generate"
    payload = {
        "prompt": prompt,
        "gpt_description_prompt": gpt_description_prompt,
        "custom_mode": custom_mode,
        "make_instrumental": make_instrumental,
        "model": model,
        "callback_url": "",
        "disable_callback": True,
        "token": api_token,
    }
    headers = {"Content-Type": "application/json"}

    # Send the POST request to start generation
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("Song generation request sent successfully!")
        return response.json().get("workId")
    else:
        print(
            "Failed to send song generation request. Status code:", response.status_code
        )
        print("Error:", response.text)
        return None


# Function to poll the status of the song generation
def poll_song_status(api_token, workId, interval=10):
    url = "https://udioapi.pro/api/feed"
    headers = {"Authorization": f"Bearer {api_token}"}
    start_time = time.time()  # Record the start time
    print("Song generation is in progress...")
    while True:
        # Send a GET request to check the status
        response = requests.get(f"{url}?workId={workId}", headers=headers)

        if response.status_code == 200:
            data = response.json()

            if data.get("type") == "complete":
                audio_url = data["response_data"][0]["audio_url"]
                print("Song generation complete!")
                print("Audio URL:", audio_url)
                end_time = time.time()  # Record the end time
                total_time = end_time - start_time  # Calculate total time taken
                print(
                    "Total time to generate the song:", round(total_time, 2), "seconds"
                )
                return audio_url
        else:
            print(
                "Failed to retrieve the song status. Status code:", response.status_code
            )
            print("Error:", response.text)
            break

        # Wait before polling again
        time.sleep(interval)
