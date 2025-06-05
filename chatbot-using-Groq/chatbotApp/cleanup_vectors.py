# utils/cleanup_vectors.py
import os
import time
from datetime import datetime, timedelta

VECTOR_DIR = "vector_store"
EXPIRY_TIME = 3600  # seconds

def cleanup_old_vectors():
    now = time.time()
    for subdir in os.listdir(VECTOR_DIR):
        path = os.path.join(VECTOR_DIR, subdir)
        if os.path.isdir(path):
            last_modified = os.path.getmtime(path)
            if now - last_modified > EXPIRY_TIME:
                print(f"Deleting expired vector store: {path}")
                os.system(f"rm -rf '{path}'")

# To run this manually or via a periodic task (e.g., cron or Celery)
if __name__ == "__main__":
    cleanup_old_vectors()
