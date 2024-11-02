import requests
import json
import time
import asyncio

async def generate_song_request(api_token, prompt, gpt_description_prompt, model="chirp-v3.0", custom_mode=False, make_instrumental=False):
    url = "https://udioapi.pro/api/generate"
    payload = {
        "prompt": prompt,
        "gpt_description_prompt": gpt_description_prompt,
        "custom_mode": custom_mode,
        "make_instrumental": make_instrumental,
        "model": model,
        "callback_url": "",
        "disable_callback": True,
        "token": api_token
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send the POST request to start generation
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, data=json.dumps(payload))
        )
        
        if response.status_code == 200:
            print("Song generation request sent successfully!")
            return response.json().get("workId")
        else:
            print("Failed to send song generation request. Status code:", response.status_code)
            print("Error:", response.text)
            return None
    except Exception as e:
        print(f"Error in generate_song_request: {e}")
        return None

async def poll_song_status(api_token, workId, interval=10):
    url = "https://udioapi.pro/api/feed"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    start_time = time.time()
    print("Song generation is in progress...")
    
    while True:
        try:
            # Send a GET request to check the status
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(f"{url}?workId={workId}", headers=headers)
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("type") == "complete":
                    audio_url = data["response_data"][0]["audio_url"]
                    print("Song generation complete!")
                    print("Audio URL:", audio_url)
                    end_time = time.time()
                    total_time = end_time - start_time
                    print("Total time to generate the song:", round(total_time, 2), "seconds")
                    return audio_url
            else:
                print("Failed to retrieve the song status. Status code:", response.status_code)
                print("Error:", response.text)
                return None
                
            # Wait before polling again
            await asyncio.sleep(interval)
            
        except Exception as e:
            print(f"Error in poll_song_status: {e}")
            return None