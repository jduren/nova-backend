import os
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import requests
from openai import OpenAI

# Initialize OpenAI client (expects OPENAI_API_KEY env var)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow mobile app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory memory store (can move to DB later)
user_memory: Dict[str, List[Dict[str, str]]] = {}

# ====== Pydantic models ======

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[Message]


# ====== Whisper transcription endpoint ======

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using OpenAI Whisper."""
    audio_bytes = await file.read()

    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(file.filename, audio_bytes, file.content_type),
    )

    return {"text": result.text}


# ====== Weather tool via Open-Meteo (no API key) ======

def get_weather_for_city(city: str) -> str:
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geocode_url, timeout=10).json()

        if "results" not in geo or not geo["results"]:
            return f"I couldn't find weather information for '{city}'."

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
            f"&longitude={lon}&current_weather=true"
        )
        weather = requests.get(weather_url, timeout=10).json()

        if "current_weather" not in weather:
            return "Weather information is not available right now."

        cw = weather["current_weather"]
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        return f"The current temperature in {city} is {temp}°C with wind speeds of {wind} km/h."
    except Exception as e:
        print("Weather error:", e)
        return "Sorry, I couldn't retrieve the weather right now."


def maybe_handle_tools(user_id: str, message: str) -> str | None:
    """Very simple 'tool' hook. You can extend this later."""
    lower = message.lower()

    # Weather: "weather in Bentonville"
    if "weather in " in lower:
        # crude parse: take everything after "weather in "
        idx = lower.find("weather in ") + len("weather in ")
        city = message[idx:].strip().rstrip("? .,!") or "Bentonville"
        return get_weather_for_city(city)

    # Add simple note tool example: "note: buy hay"
    if lower.startswith("note:") or lower.startswith("note "):
        content = message.split(":", 1)[-1].strip() if ":" in message else message[5:].strip()
        if not content:
            return "Your note is empty. Try 'Note: buy hay for the horses.'"
        user_memory.setdefault(user_id, [])
        user_memory[user_id].append({"role": "system", "content": f"[NOTE] {content}"})
        return f"Got it, I saved a note: {content}"

    return None


# ====== Chat endpoint ======

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint:
    - Uses simple tools (weather, notes).
    - Otherwise calls OpenAI for a reply.
    - Maintains per-user memory in user_memory.
    """
    user_id = req.user_id
    user_message = req.message.strip()

    # 1) Tool check
    tool_reply = maybe_handle_tools(user_id, user_message)
    if tool_reply is not None:
        # Update memory
        user_memory.setdefault(user_id, [])
        user_memory[user_id].append({"role": "user", "content": user_message})
        user_memory[user_id].append({"role": "assistant", "content": tool_reply})

        return JSONResponse({"reply": tool_reply})

    # 2) Build conversation
    # Load any previous memory for that user
    history = user_memory.get(user_id, [])

    # Convert req.history (from app) into messages
    incoming_history = [{"role": m.role, "content": m.content} for m in req.history]

    system_prompt = (
        "You are Nova, a friendly, concise personal AI assistant. "
        "You remember context from the conversation. "
        "If the user asks about weather, notes, or reminders, you can reference what I've already done, "
        "but avoid making up actions I didn't perform."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])        # last 10 memory messages
    messages.extend(incoming_history[-10:])
    messages.append({"role": "user", "content": user_message})

    # 3) Call OpenAI for reply
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        reply = "Sorry, I had an issue talking to my brain (OpenAI). Try again in a moment."

    # 4) Update memory
    user_memory.setdefault(user_id, [])
    user_memory[user_id].append({"role": "user", "content": user_message})
    user_memory[user_id].append({"role": "assistant", "content": reply})

    return JSONResponse({"reply": reply})


# ====== Health check ======

@app.get("/")
def root():
    return {"message": "Nova backend is running!"}
