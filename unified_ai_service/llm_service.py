import os
import json
from openai import AsyncOpenAI

# vLLM setup - We assume vLLM will run locally on port 8080
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-4-26B-A4B-it")

client = AsyncOpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)

async def generate_text(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return f"Error connecting to LLM server ({VLLM_URL}): {e}"

async def generate_ppt_structure(topic: str) -> list:
    sys_prompt = """You are an expert presentation creator. Generate a JSON array representing slides. 
Each object should have a 'title' string and a 'points' array of strings. 
Example: [{"title": "Intro", "points": ["Welcome", "Overview"]}]
Output ONLY valid JSON without markdown formatting."""
    
    prompt = f"Create a detailed 5-slide presentation about: {topic}"
    
    content = await generate_text(prompt, sys_prompt)
    
    # Clean up response in case it includes markdown code blocks
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        return json.loads(content)
    except Exception as e:
        print(f"JSON parsing error: {e}. Raw content: {content}")
        return [{"title": "Error", "points": ["Failed to parse LLM response into slides.", str(e)]}]

async def analyze_image(image_url_or_base64: str, prompt: str) -> str:
    """
    Function to use VLM capabilities.
    Requires an OpenAI-compatible server running a VLM (like Qwen2-VL or LLaVA).
    """
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url_or_base64}
                        }
                    ]
                }
            ],
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"VLM Analysis Error: {e}")
        return f"Error analyzing image: {e}"
