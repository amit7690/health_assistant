from django.contrib import admin
from .models import ChatLog, ChatHistory

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'role', 'timestamp')
    list_filter = ('role', 'timestamp')
    search_fields = ('session_id', 'message')
    ordering = ('-timestamp',)

@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'timestamp', 'used_pdf')
    list_filter = ('timestamp', 'used_pdf')
    search_fields = ('session_id', 'user_message', 'bot_response')
    ordering = ('-timestamp',)
