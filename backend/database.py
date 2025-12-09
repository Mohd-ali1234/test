# database.py
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from datetime import datetime
import bcrypt

# --- Configuration ---
DB_USERNAME = "Mohd"
DB_PASSWORD = "mohd1234"
DB_NAME = "myDocDataBase"

CONNECTION_STRING = (
    f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}"
    f"@cluster0.8ikmq2h.mongodb.net/{DB_NAME}"
    f"?retryWrites=true&w=majority&appName=Cluster0"
)

# -----------------------------
# 1. CONNECT TO MONGODB ATLAS
# -----------------------------
db = None
client = None

try:
    client = MongoClient(
        CONNECTION_STRING,
        server_api=ServerApi('1'),
        serverSelectionTimeoutMS=20000
    )

    client.admin.command("ping")
    print("✅ Successfully connected to MongoDB Atlas.")

    db = client[DB_NAME]  # Assign database

except Exception as e:
    print(f"❌ CRITICAL CONNECTION ERROR: {e}")
    db = None
    client = None


# -----------------------------
# 2. CREATE COLLECTIONS
# -----------------------------
if db is not None:
    users_collection = db["users"]
    health_records_collection = db["health_records"]
    chat_history_collection = db["chat_history"]
    print("📁 Collections Ready: users, health_records, chat_history")
else:
    print("⚠️ Collections NOT created — DB connection failed.")


# -----------------------------
# 3. INSERT SAMPLE DATA (once)
# -----------------------------
def insert_sample_data():
    if db is None:
        print("⚠️ No DB connection. Sample data not inserted.")
        return

    # ----- SAMPLE USER -----
    existing_user = users_collection.find_one({"email": "john@example.com"})

    if existing_user:
        print("👤 Sample user already exists.")
        user_id = existing_user["_id"]
    else:
        sample_user = {
            "_id": ObjectId(),
            "name": "John Doe",
            "email": "john@example.com",
            "password_hash": bcrypt.hashpw("password123".encode("utf-8"), bcrypt.gensalt()),
            "created_at": datetime.utcnow()
        }
        users_collection.insert_one(sample_user)
        print("👤 Sample user created.")
        user_id = sample_user["_id"]

    # ----- SAMPLE HEALTH RECORD -----
    if not health_records_collection.find_one({"user_id": user_id}):
        sample_health_record = {
            "_id": ObjectId(),
            "user_id": user_id,
            "age": 28,
            "gender": "Male",
            "weight": 72,
            "height": 178,
            "medical_history": ["asthma"],
            "medications": ["cetirizine"],
            "allergies": ["peanuts"],
            "updated_at": datetime.utcnow()
        }
        health_records_collection.insert_one(sample_health_record)
        print("🩺 Sample health record added.")
    else:
        print("🩺 Health record already exists.")

    # ----- SAMPLE CHAT HISTORY -----
   
    # chat_history_collection.insert_many([sample_chat_1, sample    _chat_2])
    print("💬 Sample chat history added.")


# RUN sample data insertion
if db is not None:
    insert_sample_data()
