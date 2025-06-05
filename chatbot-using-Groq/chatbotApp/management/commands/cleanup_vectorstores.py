from django.core.management.base import BaseCommand
import os
import shutil
from datetime import timedelta
from django.utils.timezone import now

VECTOR_STORE_ROOT = "vector_store"
EXPIRATION_DAYS = 3  # Same as session expiration

class Command(BaseCommand):
    help = "Clean up old vector stores not used in the last 3 days"

    def handle(self, *args, **kwargs):
        cutoff = now() - timedelta(days=EXPIRATION_DAYS)
        deleted = 0

        for dirname in os.listdir(VECTOR_STORE_ROOT):
            full_path = os.path.join(VECTOR_STORE_ROOT, dirname)
            if os.path.isdir(full_path):
                last_modified = os.path.getmtime(full_path)
                if now().timestamp() - last_modified > EXPIRATION_DAYS * 86400:
                    shutil.rmtree(full_path)
                    deleted += 1

        self.stdout.write(f"✅ Deleted {deleted} old vector stores.")
