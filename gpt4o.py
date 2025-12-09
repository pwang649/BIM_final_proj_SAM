import base64
import os
import requests
import sys
import io
from PIL import Image


def read_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image_format = image.format if image.format else "PNG"
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_openai_chat_completion(api_key: str, model: str, user_prompt: str, base64_image: str, max_tokens: int = 300) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    url = "https://api.openai.com/v1/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request to OpenAI API failed: {e}", file=sys.stderr)
        raise

    data = response.json()
    if "choices" not in data or not data["choices"]:
        raise ValueError("No completion choices received from API.")

    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    image_path = "rgb.png"
    api_key = os.environ.get("OPENAI_API_KEY")
    model = "gpt-4o"
    user_prompt = (
        "Describe the objects in the image as prompts (one per line) that's useful for an image segmentation model like SAM. "
        "Don't include the prompts background and table. Don't include any extra symbols in the response. "
        "Return the objects in an order that is best for sequencing remove to clear the table.")
    max_tokens = 300

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api_key or set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    # Read and encode the image
    base64_image = read_image_as_base64(image_path)

    # Call the OpenAI API
    completion_text = call_openai_chat_completion(
        api_key=api_key,
        model=model,
        user_prompt=user_prompt,
        base64_image=base64_image,
        max_tokens=max_tokens
    )
    print(completion_text)

    # Split the response into lines
    prompts_list = [line.strip() for line in completion_text.split("\n") if line.strip()]
    print(prompts_list)
