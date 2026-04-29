import pyttsx3
import threading
import pandas as pd
from database import get_attendance_records
import os

# Thread-safe TTS function so it doesn't block the Flask video stream
def speak_async(text):
    def run_tts(t):
        try:
            # Initialize pyttsx3 internally to avoid loop/thread com error on Windows
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(t)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    
    threading.Thread(target=run_tts, args=(text,), daemon=True).start()

def export_attendance_csv(date_str=None):
    """Exports attendance to a CSV file and returns the path."""
    records = get_attendance_records(date_str)
    if not records:
        return None
        
    df = pd.DataFrame(records)
    # Reorder columns
    columns = ['user_id', 'name', 'date', 'timestamp', 'status']
    df = df[columns]
    
    filename = f"attendance_records_{date_str if date_str else 'all'}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    df.to_csv(filepath, index=False)
    return filepath
