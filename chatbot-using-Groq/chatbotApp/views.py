import os
import uuid
import json
import traceback
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .health_chatbot import (
    ask_medical_query, 
    get_session_upload_dir,
    process_file
)
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from chatbotApp.models import ChatLog, ChatHistory  # Added ChatHistory import
from django.utils.timezone import now
import pytz
import time


# Render frontend
def index(request):
    if not request.session.session_key:
        request.session.create()
    return render(request, "chatbotApp/index.html")


# Upload PDF and process it into vector store
@api_view(['POST'])
def upload_file(request):
    print("upload_file function called")
    """
    Handle file uploads (PDFs and images) and process them into vector stores.
    Supports multiple file uploads in a single request.
    """
    try:
        print("\n=== Starting File Upload Process ===")
        print(f"Request method: {request.method}")
        print(f"Request FILES: {request.FILES}")
        print(f"Request data: {request.data}")
        
        if request.method == "POST" and request.FILES.get("files"):
            files = request.FILES.getlist("files")
            session_id = request.data.get("session_id", "default-session")
            
            print(f"\n=== File Upload Details ===")
            print(f"Session ID: {session_id}")
            
            # Define valid file types
            valid_types = {
                'application/pdf': 'pdf',
                'image/png': 'image',
                'image/jpeg': 'image',
                'image/jpg': 'image',
                'image/gif': 'image',
                'image/bmp': 'image'
            }
            
            try:
                # Create session-specific upload directory in media/uploads
                session_upload_dir = os.path.abspath(get_session_upload_dir(session_id))
                print(f"Creating upload directory: {session_upload_dir}")
                
                # Ensure the base media directory exists
                media_dir = os.path.abspath(os.path.join("media", "uploads"))
                if not os.path.exists(media_dir):
                    print(f"Creating base media directory: {media_dir}")
                    os.makedirs(media_dir, exist_ok=True)
                
                # Create session directory
                if not os.path.exists(session_upload_dir):
                    print(f"Creating session directory: {session_upload_dir}")
                    os.makedirs(session_upload_dir, exist_ok=True)
                print("Directory created successfully")
                
                # Verify directory permissions
                if not os.access(session_upload_dir, os.W_OK):
                    raise PermissionError(f"No write permission for directory: {session_upload_dir}")
                
            except Exception as dir_error:
                print(f"\n=== Directory Creation Error ===")
                print(f"Error type: {type(dir_error).__name__}")
                print(f"Error details: {str(dir_error)}")
                print(f"Traceback: {traceback.format_exc()}")
                return Response({
                    "error": f"Failed to create upload directory: {str(dir_error)}"
                }, status=500)
            
            uploaded_files = []
            for file in files:
                try:
                    print(f"\n=== Processing File: {file.name} ===")
                    print(f"File size: {file.size} bytes")
                    print(f"Content type: {file.content_type}")
                    
                    # Validate file type
                    if file.content_type not in valid_types:
                        print(f"Invalid file type: {file.content_type}")
                        continue
                    
                    file_type = valid_types[file.content_type]
                    print(f"File type: {file_type}")
                    
                    # Save file to session directory
                    print("\n=== Saving File ===")
                    fs = FileSystemStorage(location=session_upload_dir)
                    filename = fs.save(file.name, file)
                    file_path = os.path.abspath(fs.path(filename))
                    print(f"Saved file to: {file_path}")
                    print(f"File exists: {os.path.exists(file_path)}")
                    print(f"File size on disk: {os.path.getsize(file_path)} bytes")
                    
                    # Verify file permissions
                    if not os.access(file_path, os.R_OK):
                        raise PermissionError(f"No read permission for file: {file_path}")
                    
                    # Process file based on type
                    try:
                        # Process file and create vector store
                        print("\n=== Starting File Processing ===")
                        print(f"Using file path: {file_path}")
                        
                        # Ensure chroma_db directory exists
                        chroma_base_dir = os.path.abspath("chroma_db")
                        if not os.path.exists(chroma_base_dir):
                            print(f"Creating chroma_db directory: {chroma_base_dir}")
                            os.makedirs(chroma_base_dir, exist_ok=True)
                        
                        # Process the file using the process_file function
                        print("Processing file...")
                        vectorstore = process_file(file_path, session_id=session_id)
                        print("File processing completed successfully")
                        
                        uploaded_files.append({
                            "file_name": filename,
                            "file_path": os.path.join("uploads", session_id, filename),
                            "vector_dir": os.path.join("chroma_db", session_id, os.path.splitext(filename)[0]),
                            "type": file_type
                        })
                    except Exception as process_error:
                        print(f"Error processing file: {str(process_error)}")
                        print(f"Traceback: {traceback.format_exc()}")
                        continue
                    
                except Exception as save_error:
                    print(f"\n=== File Save Error ===")
                    print(f"Error type: {type(save_error).__name__}")
                    print(f"Error details: {str(save_error)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
            
            if not uploaded_files:
                return Response({
                    "error": "No files were successfully processed"
                }, status=400)
            
            return Response({
                "message": "Files uploaded successfully!",
                "files": uploaded_files,
                "session_id": session_id
            })
            
        else:
            print("\n=== Upload Error ===")
            print("No files found in request")
            print(f"Request FILES: {request.FILES}")
            print(f"Request data: {request.data}")
            return Response({"error": "No files uploaded."}, status=400)
            
    except Exception as e:
        print(f"\n=== Unexpected Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return Response({
            "error": f"Unexpected error: {str(e)}",
            "details": traceback.format_exc()
        }, status=500)


@csrf_exempt
@require_POST
def chatbot_api(request):
    try:
        # Parse request data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as json_error:
            return JsonResponse({
                "status": "error",
                "code": 400,
                "message": "Invalid JSON format",
                "data": None,
                "error": {
                    "type": "JSONDecodeError",
                    "details": str(json_error)
                }
            }, status=400)

        # Validate and normalize query
        raw_query = data.get("query", "")
        query = ' '.join(raw_query.strip().split())

        if not query:
            return JsonResponse({
                "status": "error",
                "code": 400,
                "message": "Missing required field: query",
                "data": None,
                "error": {
                    "type": "ValidationError",
                    "details": "Query field is required"
                }
            }, status=400)

        # Extract parameters
        file_name = data.get("file_name", "").strip()
        session_id = data.get("session_id", "default-session")
        stream = data.get("stream", False)

        # Handle PDF path
        vector_dir = None
        if file_name:
            file_path = os.path.join("media", "uploads", session_id, file_name)
            vector_dir = os.path.join("chroma_db", session_id, os.path.splitext(file_name)[0])
            if not os.path.exists(file_path):
                vector_dir = None
                file_name = None

        # Save human message if new
        if not ChatHistory.objects.filter(session_id=session_id, role="human", message=query).exists():
            ChatHistory.objects.create(
                session_id=session_id,
                role="human",
                message=query,
                timestamp=now()
            )

        # === Streamed Mode ===
        if stream:
            def stream_response():
                full_response = ""
                try:
                    for chunk in ask_medical_query(query, vector_dir, session_id, stream=True):
                        full_response += chunk
                        yield chunk

                    # Save AI message
                    if full_response.strip() and not ChatHistory.objects.filter(session_id=session_id, role="ai", message=full_response).exists():
                        ChatHistory.objects.create(
                            session_id=session_id,
                            role="ai",
                            message=full_response,
                            timestamp=now()
                        )

                    # Save chat log
                    ChatLog.objects.create(
                        session_id=session_id,
                        user_message=query,
                        bot_response=full_response,
                        used_pdf=file_name if file_name else None,
                        source_pages=None  # you can pass if available
                    )
                except Exception as stream_error:
                    yield f"\nError: {str(stream_error)}"

            return StreamingHttpResponse(stream_response(), content_type='text/plain')

        # === Non-Streamed Mode ===
        else:
            answer, source_pages = ask_medical_query(query, vector_dir, session_id)

            if answer.strip() and not ChatHistory.objects.filter(session_id=session_id, role="ai", message=answer).exists():
                ChatHistory.objects.create(
                    session_id=session_id,
                    role="ai",
                    message=answer,
                    timestamp=now()
                )

            chat_log = ChatLog.objects.create(
                session_id=session_id,
                user_message=query,
                bot_response=answer,
                used_pdf=file_name if file_name else None,
                source_pages=",".join(str(p) for p in source_pages) if source_pages else None
            )

            response_data = {
                "status": "success",
                "code": 200,
                "message": "Chat response generated successfully",
                "data": {
                    "response": answer,
                    "metadata": {
                        "session_id": session_id,
                        "timestamp": chat_log.timestamp.isoformat() if chat_log.timestamp else None,
                        "used_pdf": file_name if file_name else None,
                        "source_pages": source_pages or None
                    }
                },
                "error": None
            }

            return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "code": 500,
            "message": "Internal server error",
            "data": None,
            "error": {
                "type": type(e).__name__,
                "details": str(e),
                "traceback": traceback.format_exc()
            }
        }, status=500)


@csrf_exempt
@require_POST
def clear_chat_history(request):
    try:
        data = json.loads(request.body)
        clear_all = data.get('clear_all', False)
        
        if clear_all:
            # Clear all chat history
            ChatHistory.objects.all().delete()
            ChatLog.objects.all().delete()
            return JsonResponse({
                "status": "success",
                "message": "All chat history cleared successfully"
            })
        else:
            # Clear specific session
            session_id = data.get("session_id")
            if not session_id:
                return JsonResponse({"error": "Session ID not provided"}, status=400)
            
            # Clear chat history for this session
            ChatHistory.objects.filter(session_id=session_id).delete()
            ChatLog.objects.filter(session_id=session_id).delete()
            
            return JsonResponse({
                "status": "success",
                "message": f"Chat history cleared for session: {session_id}"
            })
            
    except Exception as e:
        print(f"Error in clear_chat_history: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)


@api_view(['GET'])
def get_chat_history(request):
    session_id = request.GET.get('session_id')
    if not session_id:
        return Response({"error": "Session ID is required"}, status=400)
    
    try:
        # Get all messages for the session, ordered by timestamp
        chat_history = ChatHistory.objects.filter(
            session_id=session_id
        ).order_by('timestamp')
        
        # Convert to list of dictionaries with proper formatting
        messages = []
        for msg in chat_history:
            try:
                # Convert timestamp to local timezone
                local_tz = pytz.timezone('Asia/Kolkata')
                local_time = msg.timestamp.astimezone(local_tz)
                
                messages.append({
                    'role': msg.role,
                    'message': msg.message,
                    'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p')
                })
            except Exception as msg_error:
                print(f"Error processing message: {str(msg_error)}")
                continue
        
        return Response(messages)
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return Response({"error": str(e)}, status=500)


@csrf_exempt
def add_pdf_to_history(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            message = data.get('message', '')  # Allow empty messages
            role = data.get('role', 'ai')  # Default to 'ai' if not specified
            
            if not session_id:
                return JsonResponse({'error': 'Session ID is required'}, status=400)
            
            # Add the message to chat history with local time
            from django.utils import timezone
            import pytz
            
            # Get current time in local timezone
            local_tz = pytz.timezone('Asia/Kolkata')  # Using IST timezone
            current_time = timezone.now().astimezone(local_tz)
            
            # Create the chat history entry
            chat_entry = ChatHistory.objects.create(
                session_id=session_id,
                role=role,
                message=message,
                timestamp=current_time
            )
            print(f"Added message to history: {message} (role: {role})")
            
            return JsonResponse({
                'status': 'success',
                'timestamp': current_time.isoformat(),
                'message_id': chat_entry.id
            })
            
        except Exception as e:
            print(f"Error adding message to history: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@api_view(['GET'])
def get_detailed_chat_history(request):
    try:
        # Get all sessions with their messages
        sessions = ChatHistory.objects.values('session_id').distinct()
        
        # Get the most recent message for each session to use as the session name
        session_data = []
        for session in sessions:
            session_id = session['session_id']
            # Get the most recent message for this session
            latest_message = ChatHistory.objects.filter(
                session_id=session_id
            ).order_by('-timestamp').first()
            
            if latest_message:
                # Convert timestamp to local timezone
                local_tz = pytz.timezone('Asia/Kolkata')
                local_time = latest_message.timestamp.astimezone(local_tz)
                
                session_data.append({
                    'session_id': session_id,
                    'user_message': latest_message.message if latest_message.role == 'human' else "",
                    'timestamp': local_time.strftime('%Y-%m-%d %I:%M %p')
                })
        
        # Sort sessions by timestamp (newest first)
        session_data.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
        
        return Response({
            'sidebar_history': session_data
        })
        
    except Exception as e:
        print(f"Error in get_detailed_chat_history: {str(e)}")
        return Response({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def create_new_session(request):
    try:
        # Check if there's already a session being created in the last 2 seconds
        last_session_time = request.session.get('last_session_creation_time', 0)
        current_time = time.time()
        
        if current_time - last_session_time < 2:
            return JsonResponse({
                'status': 'error',
                'error': 'Please wait before creating another session'
            }, status=429)
        
        # Update the last session creation time
        request.session['last_session_creation_time'] = current_time
        
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # Get current time in local timezone
        local_tz = pytz.timezone('Asia/Kolkata')
        current_time = now().astimezone(local_tz)
        
        # Check if session already exists
        if ChatHistory.objects.filter(session_id=session_id).exists():
            return JsonResponse({
                'status': 'error',
                'error': 'Session already exists'
            }, status=400)
        
        # Create a new session with an empty message
        ChatHistory.objects.create(
            session_id=session_id,
            role='system',
            message='',
            timestamp=current_time
        )
        
        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'timestamp': current_time.strftime('%Y-%m-%d %I:%M %p')
        })
        
    except Exception as e:
        print(f"Error creating new session: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_POST
def update_session_name(request):
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        name = data.get('name')
        
        if not session_id or not name:
            return JsonResponse({
                'status': 'error',
                'error': 'Session ID and name are required'
            }, status=400)
        
        # Update the session name in the database
        # You might want to add a name field to your ChatHistory model
        # For now, we'll store it in the first message of the session
        first_message = ChatHistory.objects.filter(
            session_id=session_id
        ).order_by('timestamp').first()
        
        if first_message:
            first_message.message = name
            first_message.save()
        
        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'name': name
        })
        
    except Exception as e:
        print(f"Error updating session name: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

