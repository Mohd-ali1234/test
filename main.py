from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from bson import ObjectId
from datetime import datetime
from pydantic import BaseModel
from google import genai
from google.genai import types
import json
import chromadb
from fastapi import HTTPException, Form
from bson import ObjectId
import bcrypt
from datetime import datetime
from database import users_collection
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Import DB from database.py
from database import users_collection, health_records_collection, chat_history_collection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    user_id: str
    user_message: str
    mode: str

load_dotenv()  # loads .env file

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_memory")
chat_collection = chroma_client.get_or_create_collection(
    name="chat_memory",
    metadata={"hnsw:space": "cosine"}
)



app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "..", "static")), name="static")

# ---------------------------------------------------------
# 🔥 TEST ROUTE (make sure FastAPI is working)
# ----------------------------  -----------------------------
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "..", "static/index.html"))

@app.get("/login")
def login_page():
    return FileResponse(os.path.join(BASE_DIR, "..", "static/login.html"))

@app.get("/register")
def register_page():
    return FileResponse(os.path.join(BASE_DIR, "..", "static/register.html"))

def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    if "user_id" in doc:
        doc["user_id"] = str(doc["user_id"])
    return doc

@app.post("/auth/register")
def register_user(email: str = Form(...), password: str = Form(...)):
    # Check if user already exists
    existing = users_collection.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

    # Hash password
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    user_doc = {
        "_id": ObjectId(),
        "email": email,
        "password_hash": hashed_pw,
        "created_at": datetime.utcnow()
    }

    users_collection.insert_one(user_doc)

    return {
        "success": True,
        "message": "User registered successfully",
        "user_id": str(user_doc["_id"])
    }

@app.post("/auth/login")
def login_user(email: str = Form(...), password: str = Form(...)):
    user = users_collection.find_one({"email": email})

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "success": True,
        "message": "Login successful",
        "user_id": str(user["_id"])
    }


@app.get("/debug/all_chats")
def debug_all_chats():
    chats = list(chat_history_collection.find())
    return [serialize_doc(chat) for chat in chats]

# ---------------------------------------------------------
# 🔥 GET CHAT HISTORY FOR specific user
# ---------------------------------------------------------
@app.get("/chats/{user_id}")
def get_chat_history(user_id: str):
    try:
        chats = list(
            chat_history_collection.find({
                "$or": [
                    {"user_id": user_id},               # string match
                    {"user_id": ObjectId(user_id)}      # ObjectId match
                ]
            })
        )

        # Convert ObjectId to string
        for chat in chats:
            chat["_id"] = str(chat["_id"])
            if isinstance(chat.get("user_id"), ObjectId):
                chat["user_id"] = str(chat["user_id"])

        return {"user_id": user_id, "chats": chats}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 🔥 Add a new chat message
# ---------------------------------------------------------
@app.post("/chats/add")
def add_chat(user_id: str, user_message: str, ai_response: str, category: str):
    """Insert a new chat in DB."""

    chat = {
        "user_id": user_id,
        "user_message": user_message,
        "ai_response": ai_response,
        "category": category,
        "created_at": datetime.utcnow()
    }

    result = chat_history_collection.insert_one(chat)

    return {
        "message": "Chat added",
        "chat_id": str(result.inserted_id)
    }

@app.post("/chat/send")
def send_message(msg: UserMessage):
    try:
        # NORMAL MODE → Ask Gemini
        if msg.mode == "normal":
            ai_response = get_ai_response(msg.user_message,msg.user_id)

        # AGENT MODE → Extract filters + search doctors + AI recommendation
        elif msg.mode == "agent":
            ai_response = doctor_agent(msg.user_message,msg.user_id)

        else:
            ai_response = "Invalid mode. Valid options are 'normal' or 'agent'."

        # Save to DB
        chat_doc = {
            "user_id": msg.user_id,
            "user_message": msg.user_message,
            "ai_response": ai_response,
            "category": msg.mode,
            "created_at": datetime.utcnow()
        }

        result = chat_history_collection.insert_one(chat_doc)
        chat_doc["_id"] = str(result.inserted_id)

        save_to_memory(msg.user_id, msg.user_message, message_id=f"user-{chat_doc['_id']}")
        save_to_memory(msg.user_id, ai_response, message_id=f"ai-{chat_doc['_id']}")

        return {"success": True, "chat": chat_doc}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/analyze-drug")
async def analyze_drug(
    drug_name: str = Form(None),
    drug_image: UploadFile = File(None),
    custom_analysis_focus: str = Form(None)
):
    """
    Analyze a drug by name or image.
    """
    if not drug_name and not drug_image:
        return JSONResponse(
            status_code=400,
            content={"error": "Please provide either drug_name or drug_image"}
        )

    try:
        if drug_image:
            # FIX 1: Rewind the file pointer to the start to prevent empty reads
            await drug_image.seek(0)
            
            # FIX 2: Await the read to get actual bytes
            image_bytes = await drug_image.read()
            
            # FIX 3: Check if file is empty
            if not image_bytes:
                 return JSONResponse(status_code=400, content={"error": "Uploaded file is empty"})

            # FIX 4: Sanitize Mime Type
            # If the browser sends 'application/octet-stream', default to 'image/jpeg'
            mime_type = drug_image.content_type
            if not mime_type or mime_type == "application/octet-stream":
                mime_type = "image/jpeg"
            
            result = get_drug_analysis_from_image(image_bytes, mime_type,custom_analysis_focus)
        else:
            result = get_drug_analysis_from_text(drug_name,custom_analysis_focus)

        return {"analysis": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def get_drug_analysis_from_image(image_bytes, mime_type,custom_focus):
    """
    Helper for Image Analysis
    """
    if custom_focus:
        prompt += f"\n\nAlso, please specifically address this user focus: {custom_focus}"

    prompt = """
    Identify this pill and provide:
    - Name or closest matches
    - Drug class
    - Uses
    - Common side effects
    - Serious side effects
    - Interactions
    - Safety warnings
    """

    if custom_focus:
        prompt += f"\n\nAlso, please specifically address this user focus: {custom_focus}"

    prompt += "\n\nIf not fully certain, mention uncertainty and give likely possibilities."


    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        return response.text.strip()
    except Exception as e:
        # Pass the specific API error back to the main handler
        raise e


def get_drug_analysis_from_text(drug_name,custom_focus):
    """
    Helper for Text Analysis
    """
    prompt = f"""
    You are DrugAnalyzer AI.
    User provided this drug name: "{drug_name}"

    Provide:
    - Drug identification
    - What it's used for
    - Common side effects
    - Serious side effects
    - Interactions
    - General dosage info (not personalized)
    - Precautions + warnings
    - If name may be misspelled, give closest matches
    """

    if custom_focus:
        prompt += f"\n\nAlso, please specifically address this user focus: {custom_focus}"

    prompt += "\n\nIf not fully certain, mention uncertainty and give likely possibilities."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        raise e


@app.post("/analyze-report")
async def analyze_report(
    report_file: UploadFile = File(...),
    custom_analysis_focus: str = Form(None),
    context: str = Form(None) # ADDED context parameter to analyze_report
):
    """
    Analyze an uploaded report (PDF, text, etc.) using the Gemini model.
    """
    try:
        await report_file.seek(0)
        file_bytes = await report_file.read()
        
        if not file_bytes:
             return JSONResponse(status_code=400, content={"error": "Uploaded report file is empty"})

        mime_type = report_file.content_type
        
        # Passed context to get_report_analysis
        result = get_report_analysis(file_bytes, mime_type, custom_analysis_focus, context)

        return {"analysis": result}

    except Exception as e:
        print(f"Error analyzing report: {e}") 
        return JSONResponse(status_code=500, content={"error": str(e)})

def get_report_analysis(file_bytes, mime_type, custom_focus=None, context=None):
    """
    Helper for Document Analysis
    """
    # Base prompt for medical report analysis
    prompt = """
    You are MedicalReportAnalyzer AI. Analyze the uploaded medical document (e.g., lab report, medical history, clinical summary).
    
    Provide a concise, neutral summary of the document's contents, focusing on key findings and official diagnoses (if present).
    
    Extract the following data points:
    - Patient Name/ID (if visible)
    - Document Type (e.g., Lab Report, X-Ray Scan, Discharge Summary)
    - Date of Service/Report Date
    - Key (Abnormal) Results and Reference Ranges (if applicable)
    - Official Diagnosis or Impression (if applicable)
    """
    
    # NEW: Add context to the prompt if provided
    if context:
        prompt += f"\n\nContext provided by the user (use this for supplementary information, e.g., patient age, history): {context}"

    if custom_focus:
        # Add the custom focus to the prompt
        prompt += f"\n\nIMPORTANT: Please specifically address the user's focus in your analysis: {custom_focus}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        return response.text.strip()
    except Exception as e:
        raise e


def get_ai_response(user_message: str,user_id: str):
    """
    Call Gemini API to get AI response.
    """
    past_context = retrieve_memory(user_id, user_message)
    memory_block = "\n".join(past_context) if past_context else "No previous context."

    prompt = f"""
You are MediAssist AI, a virtual AI doctor. 

The following is past conversation context relevant to the user:
---
{memory_block}
---

Now continue the conversation.
Answer the user's medical question carefully. 
- Give clear and practical advice for mild problems.
- If the symptoms indicate something serious or require a doctor, explicitly advise the user to consult a licensed doctor.
- Keep your answer safe, polite, and professional.

User Message: "{user_message}"
"""  # or you can wrap with context if needed

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        # The Gemini response object usually contains 'content' inside response output
        ai_text = response.candidates[0].content.parts[0].text
        return ai_text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return "Sorry, I couldn't process your request at the moment."        


def embed_text(text: str):
    try:
        resp = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )

        # IMPORTANT: embeddings are inside 'embeddings[0].values'
        vector = resp.embeddings[0].values

        return vector

    except Exception as e:
        print("Embedding error:", e)
        return None


def save_to_memory(user_id: str, text: str, message_id: str):
    embedding = embed_text(text)
    if embedding:
        chat_collection.add(
            ids=[message_id],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id}],
            documents=[text]
        )

def retrieve_memory(user_id: str, query: str, limit=5):
    embedding = embed_text(query)

    results = chat_collection.query(
        query_embeddings=[embedding],
        n_results=limit,
        where={"user_id": user_id} 
    )

    return results.get("documents", [[]])[0]

import requests

def doctor_agent(user_message: str,user_id: str):

    """
    AI Agent:
    1. Extract filters from user natural language.
    2. Query your doctor search API.
    3. Generate final recommendation.
    """

    past_context = retrieve_memory(user_id, user_message)
    memory_block = "\n".join(past_context) if past_context else "No previous context."

    # -------------------------------------------------------
    # 1️⃣ Extract search filters from user message using Gemini
    # -------------------------------------------------------

    extraction_prompt = f"""
You are an AI that extracts structured filters from user text.

User said: "{user_message}"

Extract ONLY these fields in pure JSON:
{{
  "name": string or null,
  "city": string or null,
  "min_rating": number or null,
  "max_fees": number or null,
  "specialization": string or null,
  "symptoms": string or null
}}

If the user did not specify something, use null.
Return ONLY JSON. No extra text.
"""

    try:
        extraction_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=extraction_prompt
        )

        raw_text = extraction_resp.candidates[0].content.parts[0].text
        filters = extract_json(raw_text)

        if filters is None:
            raise ValueError("Gemini did not return valid JSON")

    except Exception as e:
        print("Filter extraction failed:", e)
        filters = {
        "name":None,    
        "city": None,
        "min_rating": None,
        "max_fees": None,
        "specialization": None,
        "symptoms": None
    }


    # -------------------------------------------------------
    # 2️⃣ Build query params for doctor API
    # -------------------------------------------------------

    api_params = {}

    if filters.get("name"):
        api_params["name"] = filters["name"]
    if filters.get("city"):
        api_params["location"] = filters["city"]
    if filters.get("min_rating"):
        api_params["rating"] = filters["min_rating"]   # your API uses ?rating=
    if filters.get("specialization"):
        api_params["specialization"] = filters["specialization"]
    if filters.get("max_fees"):
        api_params["fees"] = filters["max_fees"]  # only if your API supports this

    api_params["page"] = 1
    api_params["limit"] = 20

    # -------------------------------------------------------
    # 3️⃣ Call your doctor search API (Correct Port + Path)
    # -------------------------------------------------------

    try:
        response = requests.get(
            "http://127.0.0.1:8001/api/doctors/search",
            params=api_params
        )
        doctor_results = response.json()

    except Exception as e:
        print("Doctor API error:", e)
        doctor_results = {"doctors": []}

    # -------------------------------------------------------
    # 4️⃣ Ask Gemini to format the final recommendation
    # -------------------------------------------------------

    summary_prompt = f"""
You are MediAssist AI Agent.

The following is past conversation context relevant to the user:
---
{memory_block}
---

Now continue the conversation.
Your task:
1. Briefly explain the user's symptoms.
2. Recommend the top 2–3 best matching doctors.
3. For each doctor, show:
   - Name
   - Specialization
   - Rating
   - Fees
   - Location
   - **Source URL (clickable link)** → MUST USE this key from data: source_url
4. Format with clean bullet points.
5. Be short, helpful, and friendly.

User Message:
{user_message}

Extracted Filters:
{filters}

Doctor Data (includes source_url):
{doctor_results}
"""


    try:
        final_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt
        )
        final_text = final_resp.candidates[0].content.parts[0].text

    except:
        final_text = "Sorry, I couldn't process your request."

    return final_text


import re

def extract_json(text):
    """
    Extract the first valid JSON object from a Gemini response.
    Removes markdown, extra text, and stops at the first closing brace.
    """
    try:
        # Remove markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # Find JSON object using regex
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            return json.loads(json_text)
        else:
            return None
    except:
        return None

