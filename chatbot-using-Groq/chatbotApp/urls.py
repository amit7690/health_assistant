from chatbotApp.views import *
from django.urls import path



urlpatterns = [
    path('', index, name='index'),
    path('api/chatbot/', chatbot_api, name='chatbot_api'),
    path("api/upload_file/", upload_file, name="upload_file"),
    path("api/clear_chat_history/", clear_chat_history, name="clear_chat_history"),
    path('api/get_chat_history/', get_chat_history, name='get_chat_history'),
    path('api/get_detailed_chat_history/', get_detailed_chat_history, name='get_detailed_chat_history'),
    path('api/add_pdf_to_history/', add_pdf_to_history, name='add_pdf_to_history'),
    path('api/create_new_session/', create_new_session, name='create_new_session'),
    path('api/update_session_name/', update_session_name, name='update_session_name'),
]



