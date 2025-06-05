from django.db import models
from django.utils import timezone

class ChatHistory(models.Model):
    session_id = models.CharField(max_length=255)
    role = models.CharField(max_length=50)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    hidden = models.BooleanField(default=False)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.role} - {self.session_id} - {self.timestamp}"

class ChatLog(models.Model):
    session_id = models.CharField(max_length=255)
    user_message = models.TextField()
    bot_response = models.TextField()
    used_pdf = models.CharField(max_length=255, null=True, blank=True)
    source_pages = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Chat Log - {self.session_id} - {self.timestamp}"
