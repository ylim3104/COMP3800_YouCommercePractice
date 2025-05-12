import os
from dotenv import load_dotenv
import base64
from openai import OpenAI

load_dotenv()

XAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"
IMAGES_DIR = "images"

def base64_encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

GROK_VISION_MODEL = "grok-2-vision-latest"
GROK_MODEL = "grok-2-latest"
grok_client = OpenAI(base_url=BASE_URL, api_key=XAI_API_KEY)

def analyze_image(image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64, {base64_encode_image(image_path)}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    completion = grok_client.chat.completions.create(
        model = GROK_VISION_MODEL, messages=messages
    )

    return completion.choices[0].message.content

