import json
import requests


def generate_image_url(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "num_images": 1,
        "size": "1024x1024",
        "response_format": "url"
    }

    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)

    if response.status_code == 200:
        result = json.loads(response.text)
        return result["data"][0]["url"]
    else:
        print(f"Error generating image: {response.status_code}")
        return None
