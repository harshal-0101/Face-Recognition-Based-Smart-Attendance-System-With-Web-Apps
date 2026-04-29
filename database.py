import os
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
import numpy as np

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'attendance_system')

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]
users_collection = db['users']
attendance_collection = db['attendance']

# Test connection on startup
try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
except Exception as e:
    print("❌ MongoDB Connection Error!")
    if "bad auth" in str(e).lower() or "authentication failed" in str(e).lower():
        print("   👉 Error: Authentication Failed.")
        print("   👉 Check your .env file: ensure MONGO_URI has the correct username and password.")
        print("   👉 Also verify that the user exists in MongoDB Atlas 'Database Access' and has the right permissions.")
    else:
        print(f"   👉 Error Details: {e}")
    # We don't exit here to allow the app to potentially start, 
    # but most features will fail until the connection is fixed.

def insert_user(user_id, name, department, encoding):
    """Inserts a new user into the database."""
    # Convert numpy array to list for MongoDB
    if isinstance(encoding, np.ndarray):
        encoding = encoding.tolist()
    
    # Check if user already exists
    if users_collection.find_one({"user_id": user_id}):
        return False, "User with this ID already exists."
        
    users_collection.insert_one({
        "user_id": user_id,
        "name": name,
        "department": department,
        "encoding": encoding,
        "registered_at": datetime.datetime.now()
    })
    return True, "User registered successfully."

def get_all_users():
    """Retrieves all registered users."""
    users = list(users_collection.find({}, {"_id": 0}))
    for user in users:
        # Convert list back to numpy array for facial recognition
        if 'encoding' in user:
            user['encoding'] = np.array(user['encoding'])
    return users

def get_user_by_id(user_id):
    """Retrieves a single user by ID."""
    return users_collection.find_one({"user_id": user_id}, {"_id": 0})

def update_user(user_id, name=None, department=None, encoding=None):
    """Updates an existing user's profile."""
    update_fields = {}
    if name is not None:
        update_fields['name'] = name
    if department is not None:
        update_fields['department'] = department
    if encoding is not None:
        if isinstance(encoding, np.ndarray):
            encoding = encoding.tolist()
        update_fields['encoding'] = encoding

    if not update_fields:
        return False, "Nothing to update."

    result = users_collection.update_one({"user_id": user_id}, {"$set": update_fields})
    if result.matched_count == 0:
        return False, "User not found."
    return True, "User updated successfully."


def delete_user(user_id):
    """Deletes a user by ID."""
    result = users_collection.delete_one({"user_id": user_id})
    if result.deleted_count == 0:
        return False, "User not found."
    return True, "User deleted successfully."


def mark_attendance(user_id, name):
    """Marks attendance for a given user if not already marked today."""
    today = datetime.date.today().isoformat() # YYYY-MM-DD
    
    # Check if already marked for today
    existing = attendance_collection.find_one({
        "user_id": user_id,
        "date": today
    })
    
    if existing:
        return False, "Attendance already marked for today."
        
    attendance_collection.insert_one({
        "user_id": user_id,
        "name": name,
        "date": today,
        "timestamp": datetime.datetime.now(),
        "status": "Present"
    })
    return True, f"Attendance marked for {name}."

def get_attendance_records(date_str=None):
    """Retrieves attendance records."""
    query = {}
    if date_str:
        query = {"date": date_str}
        
    records = list(attendance_collection.find(query, {"_id": 0}).sort("timestamp", -1))
    return records
