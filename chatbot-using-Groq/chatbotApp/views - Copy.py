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
    process_pdf, 
    clear_memory,
    get_session_upload_dir,
    get_session_vector_dir,
    clear_session_data
)
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from chatbotApp.models import ChatLog, ChatHistory  # Added ChatHistory import
from django.utils.timezone import now


# Render frontend
def index(request):
    if not request.session.session_key:
        request.session.create()
    return render(request, "chatbotApp/index.html")


# Upload PDF and process it into vector store
@api_view(['POST'])
def upload_pdf(request):
    try:
        print("\n=== Starting PDF Upload Process ===")
        print(f"Request method: {request.method}")
        print(f"Request FILES: {request.FILES}")
        print(f"Request data: {request.data}")
        
        if request.method == "POST" and request.FILES.get("pdf"):
            pdf_file = request.FILES["pdf"]
            session_id = request.data.get("session_id", "default-session")
            
            print(f"\n=== PDF Upload Details ===")
            print(f"Session ID: {session_id}")
            print(f"Original filename: {pdf_file.name}")
            print(f"File size: {pdf_file.size} bytes")
            print(f"Content type: {pdf_file.content_type}")
            
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
            
            try:
                # Save PDF to session directory
                print("\n=== Saving PDF File ===")
                fs = FileSystemStorage(location=session_upload_dir)
                filename = fs.save(pdf_file.name, pdf_file)
                file_path = os.path.abspath(fs.path(filename))
                print(f"Saved PDF to: {file_path}")
                print(f"File exists: {os.path.exists(file_path)}")
                print(f"File size on disk: {os.path.getsize(file_path)} bytes")
                
                # Verify file permissions
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"No read permission for file: {file_path}")
                
            except Exception as save_error:
                print(f"\n=== File Save Error ===")
                print(f"Error type: {type(save_error).__name__}")
                print(f"Error details: {str(save_error)}")
                print(f"Traceback: {traceback.format_exc()}")
                return Response({
                    "error": f"Failed to save PDF: {str(save_error)}"
                }, status=500)

            try:
                # Process PDF and create vector store
                print("\n=== Starting PDF Processing ===")
                print(f"Using file path: {file_path}")
                
                # Ensure chroma_db directory exists
                chroma_base_dir = os.path.abspath("chroma_db")
                if not os.path.exists(chroma_base_dir):
                    print(f"Creating chroma_db directory: {chroma_base_dir}")
                    os.makedirs(chroma_base_dir, exist_ok=True)
                
                vectorstore = process_pdf(file_path, session_id=session_id)
                print("PDF processing completed successfully")
                
                # Store the relative path for frontend use
                relative_path = os.path.join("uploads", session_id, filename)
                
                return Response({
                    "message": "PDF uploaded and processed successfully!",
                    "file_name": filename,
                    "file_path": relative_path,
                    "session_id": session_id,
                    "vector_dir": os.path.join("chroma_db", session_id, os.path.splitext(filename)[0])
                })
                
            except Exception as e:
                print(f"\n=== PDF Processing Error ===")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                
                # Clean up the uploaded file if processing fails
                if os.path.exists(file_path):
                    print(f"Cleaning up file: {file_path}")
                    try:
                        os.remove(file_path)
                        print("File cleanup successful")
                    except Exception as cleanup_error:
                        print(f"Error during file cleanup: {str(cleanup_error)}")
                
                return Response({
                    "error": f"Failed to process PDF: {str(e)}",
                    "details": traceback.format_exc()
                }, status=500)
        else:
            print("\n=== Upload Error ===")
            print("No PDF file found in request")
            print(f"Request FILES: {request.FILES}")
            print(f"Request data: {request.data}")
            return Response({"error": "No file uploaded."}, status=400)
            
    except Exception as e:
        print(f"\n=== Unexpected Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return Response({
            "error": f"Unexpected error: {str(e)}",
            "details": traceback.format_exc()
        }, status=500)


# @csrf_exempt
# @require_POST
# def chatbot_api(request):
#     try:
#         # Parse request data
#         try:
#             data = json.loads(request.body)
#         except json.JSONDecodeError as json_error:
#             return JsonResponse({
#                 "status": "error",
#                 "code": 400,
#                 "message": "Invalid JSON format",
#                 "data": None,
#                 "error": {
#                     "type": "JSONDecodeError",
#                     "details": str(json_error)
#                 }
#             }, status=400)
            
#         # Validate required fields
#         if not data.get("query"):
#             return JsonResponse({
#                 "status": "error",
#                 "code": 400,
#                 "message": "Missing required field: query",
#                 "data": None,
#                 "error": {
#                     "type": "ValidationError",
#                     "details": "Query field is required"
#                 }
#             }, status=400)
            
#         # Extract request parameters
#         query = data.get("query")
#         file_name = data.get("file_name", "").strip()
#         session_id = data.get("session_id", "default-session")
#         stream = data.get("stream", False)

#         print("\n=== Request Parameters ===")
#         print(f"Query: {query}")
#         print(f"File name: {file_name}")
#         print(f"Session ID: {session_id}")
#         print(f"Stream: {stream}")

#         # Initialize vector_dir as None by default
#         vector_dir = None
        
#         # Handle PDF if provided
#         if file_name and file_name.strip():
#             file_path = os.path.join("media", "uploads", session_id, file_name)
#             vector_dir = os.path.join("chroma_db", session_id, os.path.splitext(file_name)[0])
            
#             print(f"\n=== PDF Details ===")
#             print(f"File path: {file_path}")
#             print(f"Vector dir: {vector_dir}")
            
#             if not os.path.exists(file_path):
#                 print(f"PDF not found at: {file_path}")
#                 vector_dir = None
#                 file_name = None
#             else:
#                 print(f"PDF found and will be used")

#         # Store user message if not exists
#         if not ChatHistory.objects.filter(session_id=session_id, role="human", message=query).exists():
#             ChatHistory.objects.create(
#                 session_id=session_id,
#                 role="human",
#                 message=query,
#                 timestamp=now()
#             )
#             print(f"\n=== Stored User Message ===")
#             print(f"Session ID: {session_id}")
#             print(f"Message: {query}")

#         try:
#             if stream:
#                 def stream_response():
#                     try:
#                         full_response = ""
#                         for chunk in ask_medical_query(query, vector_dir, session_id, stream=True):
#                             full_response += chunk
#                             yield chunk
                        
#                         # Store bot response
#                         if not ChatHistory.objects.filter(session_id=session_id, role="ai", message=full_response).exists():
#                             ChatHistory.objects.create(
#                                 session_id=session_id,
#                                 role="ai",
#                                 message=full_response,
#                                 timestamp=now()
#                             )
                        
#                         # Store in ChatLog
#                         ChatLog.objects.create(
#                             session_id=session_id,
#                             user_query=query,
#                             response=full_response,
#                             used_pdf=file_name if file_name else None,
#                             source_pages=None
#                         )
#                     except Exception as stream_error:
#                         print(f"\n=== Stream Error ===")
#                         print(f"Error type: {type(stream_error).__name__}")
#                         print(f"Error details: {str(stream_error)}")
#                         print(f"Traceback: {traceback.format_exc()}")
#                         yield f"\nError: {str(stream_error)}"
                
#                 return StreamingHttpResponse(stream_response(), content_type='text/plain')
#             else:
#                 # Get response from chatbot
#                 print("\n=== Calling ask_medical_query ===")
#                 print(f"Query: {query}")
#                 print(f"Vector dir: {vector_dir}")
#                 print(f"Session ID: {session_id}")
                
#                 answer, source_pages = ask_medical_query(query, vector_dir, session_id)
                
#                 print("\n=== Received Response ===")
#                 print(f"Answer: {answer}")
#                 print(f"Source pages: {source_pages}")
                
#                 # Store bot response
#                 if not ChatHistory.objects.filter(session_id=session_id, role="ai", message=answer).exists():
#                     ChatHistory.objects.create(
#                         session_id=session_id,
#                         role="ai",
#                         message=answer,
#                         timestamp=now()
#                     )
                
#                 # Store in ChatLog
#                 chat_log = ChatLog.objects.create(
#                     session_id=session_id,
#                     user_query=query,
#                     response=answer,
#                     used_pdf=file_name if file_name else None,
#                     source_pages=",".join(str(p) for p in source_pages) if source_pages else None
#                 )
                
#                 # Prepare success response
#                 response_data = {
#                     "status": "success",
#                     "code": 200,
#                     "message": "Chat response generated successfully",
#                     "data": {
#                         "response": answer,
#                         "metadata": {
#                             "session_id": session_id,
#                             "timestamp": chat_log.timestamp.isoformat() if chat_log.timestamp else None,
#                             "used_pdf": file_name if file_name else None,
#                             "source_pages": source_pages if source_pages else None
#                         }
#                     },
#                     "error": None
#                 }
                
#                 print("\n=== Final JSON Response ===")
#                 print(json.dumps(response_data, indent=2))
                
#                 return JsonResponse(response_data)
                
#         except Exception as query_error:
#             print(f"\n=== Query Error ===")
#             print(f"Error type: {type(query_error).__name__}")
#             print(f"Error details: {str(query_error)}")
#             print(f"Traceback: {traceback.format_exc()}")
            
#             return JsonResponse({
#                 "status": "error",
#                 "code": 500,
#                 "message": "Error generating response",
#                 "data": None,
#                 "error": {
#                     "type": type(query_error).__name__,
#                     "details": str(query_error),
#                     "traceback": traceback.format_exc()
#                 }
#             }, status=500)

#     except Exception as e:
#         print(f"\n=== General Error ===")
#         print(f"Error type: {type(e).__name__}")
#         print(f"Error details: {str(e)}")
#         print(f"Traceback: {traceback.format_exc()}")
        
#         return JsonResponse({
#             "status": "error",
#             "code": 500,
#             "message": "Internal server error",
#             "data": None,
#             "error": {
#                 "type": type(e).__name__,
#                 "details": str(e),
#                 "traceback": traceback.format_exc()
#             }
#         }, status=500)


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
                        user_query=query,
                        response=full_response,
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
                user_query=query,
                response=answer,
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
        session_id = data.get("session_id")
        if session_id:
            # Clear chat history from database
            ChatHistory.objects.filter(session_id=session_id).delete()
            ChatLog.objects.filter(session_id=session_id).delete()
            
            # Clear PDFs and vector stores
            try:
                # Get the session upload directory
                session_upload_dir = os.path.join("media", "uploads", session_id)
                if os.path.exists(session_upload_dir):
                    # Delete all files in the upload directory
                    for file in os.listdir(session_upload_dir):
                        file_path = os.path.join(session_upload_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {str(e)}")
                    
                    # Remove the upload directory
                    try:
                        os.rmdir(session_upload_dir)
                        print(f"Removed directory: {session_upload_dir}")
                    except Exception as e:
                        print(f"Error removing directory {session_upload_dir}: {str(e)}")

                # Get the session vector store directory
                session_vector_dir = os.path.join("chroma_db", session_id)
                if os.path.exists(session_vector_dir):
                    # Delete all files in the vector store directory
                    for file in os.listdir(session_vector_dir):
                        file_path = os.path.join(session_vector_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                print(f"Deleted vector store file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting vector store file {file_path}: {str(e)}")
                    
                    # Remove the vector store directory
                    try:
                        os.rmdir(session_vector_dir)
                        print(f"Removed vector store directory: {session_vector_dir}")
                    except Exception as e:
                        print(f"Error removing vector store directory {session_vector_dir}: {str(e)}")

            except Exception as e:
                print(f"Error clearing session data: {str(e)}")
                return JsonResponse({"error": f"Error clearing session data: {str(e)}"}, status=500)

            return JsonResponse({
                "message": f"Chat history, PDFs, and vector stores cleared for session: {session_id}"
            })
        return JsonResponse({"error": "Session ID not provided"}, status=400)
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
        chat_history = ChatHistory.objects.filter(session_id=session_id).order_by('timestamp')
        
        # Convert to list of dictionaries
        messages = [
            {
                'role': msg.role,
                'message': msg.message,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
            }
            for msg in chat_history
        ]
        
        print(f"Retrieved {len(messages)} messages for session {session_id}")
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
            message = data.get('message')
            role = data.get('role', 'ai')  # Default to 'ai' if not specified
            
            if not session_id or not message:
                return JsonResponse({'error': 'Missing required fields'}, status=400)
            
            # Check if this message already exists
            existing_message = ChatHistory.objects.filter(
                session_id=session_id,
                message=message,
                role=role
            ).first()
            
            if not existing_message:
                # Add the message to chat history
                ChatHistory.objects.create(
                    session_id=session_id,
                    role=role,
                    message=message,
                    timestamp=now()
                )
                print(f"Added message to history: {message} (role: {role})")
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            print(f"Error adding message to history: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@api_view(['GET'])
def get_detailed_chat_history(request):
    session_id = request.GET.get('session_id')
    if not session_id:
        return Response({"error": "Session ID is required"}, status=400)
    
    try:
        # Get all messages for the session, ordered by timestamp in descending order (newest first)
        chat_history = ChatHistory.objects.filter(session_id=session_id).order_by('-timestamp')
        
        # Get all chat logs for the session, ordered by timestamp in descending order
        chat_logs = ChatLog.objects.filter(session_id=session_id).order_by('-timestamp')
        
        # Convert to list of dictionaries with detailed information
        messages = []
        for msg in chat_history:
            message_data = {
                'role': msg.role,
                'message': msg.message,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else None,
                'session_id': msg.session_id
            }
            messages.append(message_data)
        
        # Get chat logs with analytics
        logs = []
        for log in chat_logs:
            log_data = {
                'user_query': log.user_query,
                'response': log.response,
                'used_pdf': log.used_pdf,
                'source_pages': log.source_pages,
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'session_id': log.session_id
            }
            logs.append(log_data)
        
        print("\n=== Detailed Chat History ===")
        print(f"Session ID: {session_id}")
        print(f"Number of messages: {len(messages)}")
        print(f"Number of chat logs: {len(logs)}")
        
        # Organize conversations for sidebar display
        conversations = []
        current_conversation = []
        
        for msg in messages:
            if msg['role'] == 'human':
                if current_conversation:
                    # Add the completed conversation to the list
                    conversations.append({
                        'user_query': current_conversation[0]['message'],
                        'bot_response': current_conversation[-1]['message'],
                        'timestamp': current_conversation[-1]['timestamp']
                    })
                current_conversation = [msg]
            else:
                current_conversation.append(msg)
        
        # Add the last conversation if it exists
        if current_conversation:
            conversations.append({
                'user_query': current_conversation[0]['message'],
                'bot_response': current_conversation[-1]['message'],
                'timestamp': current_conversation[-1]['timestamp']
            })
        
        # Sort conversations by timestamp (newest first)
        conversations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        response_data = {
            'session_id': session_id,
            'conversation_history': messages,
            'chat_logs': logs,
            'sidebar_history': conversations  # Now contains full conversation pairs
        }
        
        print("\n=== Response JSON ===")
        print(json.dumps(response_data, indent=2))
        
        return Response(response_data)
        
    except Exception as e:
        print(f"Error retrieving detailed chat history: {str(e)}")
        return Response({"error": str(e)}, status=500)

