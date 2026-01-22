import os
import uuid
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq  # Updated to Groq

import chromadb
from chromadb.utils import embedding_functions

from database import db
from model import AppUsageBatch
from typing import Optional, Literal


from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, date
from pymongo import UpdateMany
from pytz import timezone

scheduler = BackgroundScheduler()

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(title="Aivis Backend", version="1.0")

# --------------------------------------------------
# DATABASE & CHROMA SETUP
# --------------------------------------------------
if db is None:
    raise RuntimeError("Database connection failed")

app_usage_collection = db["app_usage"]
chat_history_collection = db["chat_history"]
habits_collection = db["habits"]
habit_logs_collection = db["habit_logs"]
daily_summaries_collection = db["daily_summaries"]

# ChromaDB (RAG MEMORY)
chroma_client = chromadb.PersistentClient(path="./user_data")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

memory_collection = chroma_client.get_or_create_collection(
    name="user_profiles_memory",
    embedding_function=embedding_fn
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class HabitRequest(BaseModel):
    user_id: str
    name: str
    icon: str
    goal_type: Literal["yesNo", "count", "time"]
    goal_value: Optional[int] = None


class HabitProgressRequest(BaseModel):
    habit_id: str
    progress: Optional[int] = 0
    completed: Optional[bool] = False



# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_user_context(user_id: str, query: str) -> str:
    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=3,
            where={"user_id": user_id}
        )
        if results.get("documents") and results["documents"][0]:
            return " ".join(results["documents"][0])
        return ""
    except Exception:
        return ""
    
IST = timezone("Asia/Kolkata")
        
def close_all_users_day():
    users = habits_collection.distinct("user_id")

    for user_id in users:
        close_day(str(user_id))

scheduler.add_job(
    close_all_users_day,
    trigger="cron",
    hour=2,
    minute=0,
    timezone=IST
)

def get_today():
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    if "user_id" in doc:
        doc["user_id"] = str(doc["user_id"])
    if "habit_id" in doc:
        doc["habit_id"] = str(doc["habit_id"])
    return doc

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.on_event("startup")
def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        print("✅ Scheduler started")

@app.post("/app-usage/batch")
def store_app_usage_batch(data: AppUsageBatch):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        for app_data in data.apps:
            app_usage_collection.update_one(
                {
                    "user_id": ObjectId(data.user_id),
                    "package_name": app_data.package_name,
                    "date": today
                },
                {
                    "$set": {
                        "usage_duration_sec": app_data.usage_sec,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        return {"success": True, "message": "Usage stored", "date": today}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    # 1. Get memory context
    user_context = get_user_context(request.user_id, request.message)

    # 2. Build the System Prompt
    system_prompt = f"""
    You are Aivis 🤖✨, a world-class Behavioral Coach and Personal Architect. 

    ### CORE MISSION:
    Your goal is to help the user bridge the gap between their current habits and their ideal self. 

    ### USER INTELLIGENCE (Past Context):
    {user_context if user_context else "No prior history. This is a new interaction."}

    ### YOUR OPERATIONAL PROTOCOL:
    1.  **Extract & Remember:** Identify any 'Core Truths' the user reveals (Goals, Fears, Values, Personality traits). 
    2.  **Pattern Recognition:** If they mention a habit, compare it to their stated goals.
    3.  **Tone & Persona:**
        - Be insightful and proactive (don't just react; suggest).
        - Use "Habit Stacking" logic (e.g., "Since you do X, try adding Y").
        - End with ONE high-impact, open-ended question that helps you learn more about their personality.

    ### EXTRACTION TARGETS:
    Always look for and reinforce:
    - **The 'Why':** Why do they want this goal?
    - **The 'Blocker':** What is their specific "friction"? (e.g., "I'm too tired after work").
    - **Communication Style:** Do they prefer tough love or gentle support?
        """

    try:
        # 3. Groq API Call
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=500
        )

        ai_reply = completion.choices[0].message.content

        # 4. Save message to ChromaDB for future context
        memory_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[request.message],
            metadatas=[{"user_id": request.user_id}]
        )

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Aivis is taking a nap. Try again soon!")
    
# 1️⃣ CREATE HABIT (ONCE)
@app.post("/habits")
def create_habit(habit: HabitRequest):
    habit_doc = {
        "user_id": ObjectId(habit.user_id),
        "name": habit.name,
        "icon": habit.icon,
        "goal_type": habit.goal_type,
        "goal_value": habit.goal_value,
        "created_at": datetime.utcnow(),
        "is_active": True
    }

    result = habits_collection.insert_one(habit_doc)

    return {
        "success": True,
        "habit_id": str(result.inserted_id)
    }


# 2️⃣ UPDATE TODAY'S PROGRESS
@app.put("/habits/update")
def update_habit(req: HabitProgressRequest):
    today = get_today()

    habit_log = {
        "habit_id": ObjectId(req.habit_id),
        "date": today,
        "progress": req.progress or 0,
        "completed": req.completed or False,
        "updated_at": datetime.utcnow()
    }

    habit_logs_collection.update_one(
        {
            "habit_id": ObjectId(req.habit_id),
            "date": today
        },
        {"$set": habit_log},
        upsert=True
    )

    return {"success": True}


# 3️⃣ GET TODAY'S HABITS (AUTO RESET LOGIC)
@app.get("/habits/today/{user_id}")
def get_today_habits(user_id: str):
    today = get_today()

    habits = list(habits_collection.find({"user_id": ObjectId(user_id), "is_active": True}))
    habit_ids = [h["_id"] for h in habits]

    logs = {
        log["habit_id"]: log
        for log in habit_logs_collection.find(
            {"habit_id": {"$in": habit_ids}, "date": today}
        )
    }

    response = []

    for habit in habits:
        log = logs.get(habit["_id"], {
            "progress": 0,
            "completed": False
        })

        response.append({
            **serialize(habit),
            "today_progress": log["progress"],
            "completed": log["completed"]
        })

    return {"success": True, "habits": response}


# 4️⃣ GET YESTERDAY / ANY DAY
@app.get("/habits/history/{user_id}")
def get_habit_history(user_id: str, days_ago: int = 1):
    target_day = get_today() - timedelta(days=days_ago)

    logs = habit_logs_collection.find({"date": target_day})

    return {
        "success": True,
        "date": target_day,
        "logs": [serialize(log) for log in logs]
    }


# 5️⃣ DELETE HABIT
@app.delete("/habits/{habit_id}")
def delete_habit(habit_id: str):
    habits_collection.update_one(
        {"_id": ObjectId(habit_id)},
        {"$set": {"is_active": False}}
    )
    return {"success": True}

@app.post("/daily/close-day/{user_id}")
def close_day(user_id: str):
    try:
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        today_date = get_today()

        user_obj_id = ObjectId(user_id)

        # 🛑 Prevent duplicate close
        if daily_summaries_collection.find_one({
            "user_id": user_obj_id,
            "date": today_str
        }):
            return {
                "success": True,
                "message": "Day already closed",
                "date": today_str
            }

        # 1️⃣ FETCH TODAY'S APP USAGE
        app_usage = list(app_usage_collection.find({
            "user_id": user_obj_id,
            "date": today_str
        }))

        app_usage_data = [
            {
                "package_name": a["package_name"],
                "usage_duration_sec": a["usage_duration_sec"]
            }
            for a in app_usage
        ]

        # 2️⃣ FETCH ALL ACTIVE USER HABITS
        user_habits = list(
            habits_collection.find({
                "user_id": user_obj_id,
                "is_active": True
            })
        )

        habit_ids = [h["_id"] for h in user_habits]

        # 3️⃣ FETCH TODAY'S HABIT LOGS
        logs = {
            log["habit_id"]: log
            for log in habit_logs_collection.find({
                "date": today_date,
                "habit_id": {"$in": habit_ids}
            })
        }

        # 4️⃣ MERGE HABITS + LOGS
        habits_summary = []

        for habit in user_habits:
            log = logs.get(habit["_id"])

            progress = log["progress"] if log else 0
            completed = log["completed"] if log else False

            if completed:
                status = "completed"
            elif progress > 0:
                status = "partial"
            else:
                status = "not_done"

            habits_summary.append({
                "habit_id": habit["_id"],
                "name": habit["name"],
                "icon": habit["icon"],
                "goal_type": habit["goal_type"],
                "goal_value": habit.get("goal_value"),
                "progress": progress,
                "completed": completed,
                "status": status
            })

        # 5️⃣ SAVE DAILY SNAPSHOT
        daily_summaries_collection.insert_one({
            "user_id": user_obj_id,
            "date": today_str,
            "app_usage": app_usage_data,
            "habits": habits_summary,
            "created_at": datetime.utcnow()
        })

        # 6️⃣ RESET APP USAGE (HABIT LOGS ARE DATE-BASED, NO RESET NEEDED)
        app_usage_collection.delete_many({
            "user_id": user_obj_id,
            "date": today_str
        })

        print(f"✅ Day closed for user {user_id}")

        return {
            "success": True,
            "date": today_str,
            "apps_logged": len(app_usage_data),
            "habits_logged": len(habits_summary)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

