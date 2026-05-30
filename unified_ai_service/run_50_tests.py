import asyncio
import aiohttp
import json
import uuid

API_URL = "http://localhost:8081/api/multimodal/execute"
HEADERS = {"X-Email": "yeonwoo.kim03@gmail.com"}

TEST_CASES = [
    # 1-5: Complex Video (Lecture Pro, Drama, Animation, Ad)
    ("Create a 30-minute long lecture video about Quantum Physics with PPT sync and BGM", "standard", "true"),
    ("A dramatic movie scene where a detective finds the clue in the rain", "standard", "true"),
    ("An energetic commercial ad for a refreshing orange soda", "standard", "true"),
    ("A 3D animation of a flying robot saving a cat", "standard", "true"),
    ("Make a professional lecture video using this script and image", "standard", "true"),
    
    # 6-15: Image Generation & Editing (Flux)
    ("Draw a cinematic 8k portrait of a futuristic samurai in neon Tokyo", "high", "true"),
    ("A cute red cat reading a book", "draft", "true"),
    ("A photorealistic rendering of a cyberpunk city at night", "standard", "true"),
    ("Edit this image to make the sky look like sunset", "standard", "true"),
    ("Remove the person in the background using mask", "standard", "true"),
    ("A minimalist logo for a tech startup", "draft", "true"),
    ("A highly detailed sketch of an old wizard", "high", "true"),
    ("Pop art style illustration of a coffee cup", "standard", "true"),
    ("Macro photography of a dew drop on a leaf", "high", "true"),
    ("Abstract geometry with vibrant colors", "draft", "true"),
    
    # 16-25: Music & TTS
    ("Create a 15-second high-energy synthwave track", "standard", "true"),
    ("A calm Lo-Fi hip hop track with soft piano", "standard", "true"),
    ("Epic orchestral soundtrack for a fantasy movie", "standard", "true"),
    ("Upbeat electronic dance music for a commercial", "standard", "true"),
    ("Say 'Welcome to our service, we are glad to have you.' in a professional voice", "standard", "true"),
    ("Read this text energetically", "standard", "true"),
    ("Narrate a dramatic opening for a movie trailer", "standard", "true"),
    ("Explain the concept of gravity as a calm teacher", "standard", "true"),
    ("Say 'Error 404' in a robotic voice", "standard", "true"),
    ("Tell a short joke about a programmer", "standard", "true"),
    
    # 26-35: Presentation & Document Parsing
    ("Create a 3-slide presentation about Space Mining profitability", "standard", "true"),
    ("Presentation about the history of artificial intelligence", "standard", "true"),
    ("A business pitch deck for a new eco-friendly water bottle", "standard", "true"),
    ("Educational slides explaining quantum physics to beginners", "standard", "true"),
    ("A marketing strategy for a new fashion brand", "standard", "true"),
    ("Analyze this image of a graph and explain the trend", "standard", "true"),
    ("Extract the main points from this document table", "standard", "true"),
    ("Transcribe this audio file into text", "standard", "true"),
    ("Create subtitles (srt) for this video audio", "standard", "true"),
    ("Summarize the key events of World War II", "standard", "true"),
    
    # 36-50: VLM Analysis & General Chat
    ("What are the key trends in modern web development?", "standard", "true"),
    ("Explain the theory of relativity simply.", "standard", "true"),
    ("Write a Python script to reverse a string.", "standard", "true"),
    ("How do I make a chocolate cake?", "standard", "true"),
    ("What is the capital of France?", "standard", "true"),
    ("Analyze the lighting in this cinematic shot", "standard", "true"),
    ("Is there any text visible in this image?", "standard", "true"),
    ("Describe the emotions of the people in the video", "standard", "true"),
    ("What instrument is playing in the background?", "standard", "true"),
    ("Create a short catchy jingle for a podcast intro", "standard", "true"),
    ("A time-lapse video of a flower blooming", "standard", "true"),
    ("Generate a 16:9 wallpaper of a neon city", "high", "true"),
    ("Create an idle loop video of this portrait", "standard", "true"),
    ("Edit this video to add a vintage film grain effect", "standard", "true"),
    ("Analyze this spreadsheet image and list the columns", "standard", "true"),
]

async def run_test(session, i, instruction, quality, dry_run):
    form = aiohttp.FormData()
    form.add_field('instruction', instruction)
    form.add_field('quality', quality)
    form.add_field('dry_run', dry_run)

    
    try:
        async with session.post(API_URL, data=form, headers=HEADERS) as resp:
            data = await resp.json()
            if resp.status == 200 and data.get("dry_run"):
                plan = data.get("plan", {})
                goal = plan.get("goal", "unknown")
                print(f"[{i:02d}] ✅ SUCCESS | Goal: {goal:15} | Inst: {instruction[:40]}...")
                return {"id": i, "status": "success", "goal": goal, "instruction": instruction}
            else:
                print(f"[{i:02d}] ❌ FAILED  | Status: {resp.status} | Data: {data}")
                return {"id": i, "status": "failed", "error": str(data)}
    except Exception as e:
        print(f"[{i:02d}] ❌ ERROR   | {e}")
        return {"id": i, "status": "error", "error": str(e)}

async def main():
    print(f"Running {len(TEST_CASES)} Multimodal Planning Tests...")
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (inst, qual, dr) in enumerate(TEST_CASES, 1):
            tasks.append(run_test(session, i, inst, qual, dr))
            # Limit concurrency to avoid overloading the FastAPI server immediately
            if len(tasks) >= 5:
                res = await asyncio.gather(*tasks)
                results.extend(res)
                tasks = []
                await asyncio.sleep(1)
        if tasks:
            res = await asyncio.gather(*tasks)
            results.extend(res)
            
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n--- TEST REPORT ---")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Successful Plans: {success_count}")
    print(f"Failed Plans: {len(TEST_CASES) - success_count}")
    
    with open("/home/yanus/test_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Report saved to /home/yanus/test_report.json")

if __name__ == "__main__":
    asyncio.run(main())
