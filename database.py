# database.py
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

DB_USERNAME = os.getenv("MONGO_USERNAME")
DB_PASSWORD = os.getenv("MONGO_PASSWORD")
DB_NAME = os.getenv("MONGO_DB_NAME")

CONNECTION_STRING = (
    f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}"
    f"@cluster0.8ikmq2h.mongodb.net/{DB_NAME}"
    f"?retryWrites=true&w=majority&appName=Cluster0"
)

client = None
db = None

try:
    client = MongoClient(
        CONNECTION_STRING,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=20000
    )

    client.admin.command("ping")
    db = client[DB_NAME]

    print("✅ Successfully connected to MongoDB Atlas.")
    print("📁 Collections Ready")

except Exception as e:
    print(f"❌ DATABASE ERROR: {e}")
    client = None
    db = None
