import os
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv('MONGO_URI')
if uri:
    # Mask password for safety
    parts = uri.split(':')
    if len(parts) > 2:
        username = parts[1].replace('//', '')
        password_part = parts[2].split('@')[0]
        masked_uri = uri.replace(password_part, '****')
        print(f"Loaded URI: {masked_uri}")
    else:
        print(f"Loaded URI: {uri}")
else:
    print("MONGO_URI not found in environment")
