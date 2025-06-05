import os
import traceback
import time
from typing import Optional, Union
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from chatbotApp.models import ChatLog, ChatHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from functools import lru_cache
import hashlib
import shutil
import json

# Initialize Groq API client
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_groq_response(messages, temperature=0.3, max_tokens=4096, stream=False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    if stream:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield {"content": delta['content']}
                                elif data['choices'][0].get('finish_reason') == 'stop':
                                    break
                        except json.JSONDecodeError:
                            continue
        else:
            raise Exception(f"Groq API error: {response.text}")
    else:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API error: {response.text}")

# Custom LLM class to work with Groq
class GroqLLM:
    def __init__(self, temperature=0.3, max_tokens=4096):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, input_text):
        messages = [{"role": "user", "content": input_text}]
        return get_groq_response(messages, self.temperature, self.max_tokens, stream=False)

    def stream(self, input_text):
        messages = [{"role": "user", "content": input_text}]
        return get_groq_response(messages, self.temperature, self.max_tokens, stream=True)

# Initialize LLM with Groq
llm = GroqLLM(
    temperature=0.3,
    max_tokens=4096
)

# Cache for storing common medical responses
@lru_cache(maxsize=1000)
def get_cached_response(query_hash: str) -> Optional[str]:
    return None  # Implement actual caching logic if needed

def hash_query(query: str) -> str:
    """Create a hash of the query for caching"""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()



chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful and empathetic medical assistant. Only answer health-related questions (medicine, wellness, biology). Politely decline unrelated topics.

Behavior:
- Greet only if the user says hi/hello/hey. Otherwise, skip greetings and respond directly.
- Never give diagnoses or treatment plans.
- For medicines: explain general uses, dosage ranges, side effects, and precautions. Never suggest specific drugs or doses. Always recommend consulting a doctor or pharmacist.

PDF Context:
{context}

Rules:
- If PDF context is provided, answer using ONLY that. Say clearly if it lacks relevant info. Cite context if possible.
- If no context, use your medical knowledge. Don't mention documents.

Always be accurate, concise, safe, and respectful.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])



# Chain
def process_chat(input_text, context="", history=None):
    messages = []
    
    # Add system message with context
    system_message = f"""You are a helpful and empathetic medical assistant. Only answer health-related questions (medicine, wellness, biology). Politely decline unrelated topics.

Behavior:
- Greet only if the user says hi/hello/hey. Otherwise, skip greetings and respond directly.
- Never give diagnoses or treatment plans.
- For medicines: explain general uses, dosage ranges, side effects, and precautions. Never suggest specific drugs or doses. Always recommend consulting a doctor or pharmacist.
- For non-health related questions (like programming, technology, etc.): Politely redirect the user to ask health-related questions instead. Do not provide any information about the unrelated topic.

PDF Context:
{context}

Rules:
- If PDF context is provided, answer using ONLY that. Say clearly if it lacks relevant info. Cite context if possible.
- If no context, use your medical knowledge. Don't mention documents.
- For non-health questions, simply say: "I'm a health-focused assistant. I can help you with medical and wellness questions. What health-related topic would you like to know about?"

Always be accurate, concise, safe, and respectful."""
    
    messages.append({"role": "system", "content": system_message})
    
    # Add history messages
    if history:
        for msg in history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({"role": role, "content": msg.content})
    
    # Add current input
    messages.append({"role": "user", "content": input_text})
    
    return get_groq_response(messages)

# Use DB-based session memory
class DBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self):
        msgs = ChatHistory.objects.filter(session_id=self.session_id).order_by("timestamp")
        return [
            HumanMessage(content=m.message) if m.role == "human" else AIMessage(content=m.message)
            for m in msgs
        ]

    def add_user_message(self, message: str):
        ChatHistory.objects.create(session_id=self.session_id, role="human", message=message)

    def add_ai_message(self, message: str):
        ChatHistory.objects.create(session_id=self.session_id, role="ai", message=message)

    def add_messages(self, messages):
        for msg in messages:
            if isinstance(msg, HumanMessage):
                self.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                self.add_ai_message(msg.content)

    def clear(self):
        ChatHistory.objects.filter(session_id=self.session_id).delete()

# Modified conversation chain to work with Groq
def get_conversation_chain(session_id: str):
    history = DBChatMessageHistory(session_id)
    return lambda input_text, context="": process_chat(input_text, context, history.messages)

def clear_memory(session_id):
    DBChatMessageHistory(session_id).clear()
    print(f"Chat history cleared for session: {session_id}")

def get_session_upload_dir(session_id: str) -> str:
    """Get the upload directory for a specific session"""
    return os.path.join("media", "uploads", session_id)

def get_session_vector_dir(session_id: str) -> str:
    """Get the vector store directory for a specific session"""
    return os.path.join("chroma_db", session_id)

def clear_session_data(session_id: str):
    """Clear all data (PDFs and vectors) for a specific session"""
    # Clear vector stores
    vector_dir = get_session_vector_dir(session_id)
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
        print(f"Cleared vector stores for session: {session_id}")
    
    # Clear uploaded PDFs
    upload_dir = get_session_upload_dir(session_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        print(f"Cleared uploaded PDFs for session: {session_id}")

def save_pdf_to_session(pdf_path: str, session_id: str) -> str:
    """Save a PDF to the session's upload directory and return the new path"""
    # Create session upload directory
    session_upload_dir = get_session_upload_dir(session_id)
    os.makedirs(session_upload_dir, exist_ok=True)
    
    # Get the original filename
    original_filename = os.path.basename(pdf_path)
    
    # Create new path in session directory
    new_pdf_path = os.path.join(session_upload_dir, original_filename)
    
    # Copy the PDF to the session directory
    shutil.copy2(pdf_path, new_pdf_path)
    print(f"✅ PDF uploaded: {original_filename}")
    
    return new_pdf_path

# Process single PDF
def process_pdf(pdf_path: str, session_id: str = "default-session") -> Chroma:
    start_time = time.time()
    print(f"\n=== Processing PDF for session {session_id} ===")
    print(f"Input PDF path: {pdf_path}")
    
    try:
        # Verify PDF file exists and is readable
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found at: {pdf_path}")
        
        if not os.access(pdf_path, os.R_OK):
            raise ValueError(f"PDF file is not readable: {pdf_path}")
            
        print(f"PDF file exists and is readable")
        print(f"File size: {os.path.getsize(pdf_path)} bytes")
        
        # Create session-specific vector directory
        persist_directory = get_session_vector_dir(session_id)
        print(f"Creating vector directory: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Load and split the PDF
        print("Loading PDF...")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages")
        except Exception as pdf_error:
            print(f"Error loading PDF: {str(pdf_error)}")
            raise ValueError(f"Failed to load PDF: {str(pdf_error)}")
        
        print("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        try:
            embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        except Exception as embed_error:
            print(f"Error initializing embedding model: {str(embed_error)}")
            raise ValueError(f"Failed to initialize embedding model: {str(embed_error)}")
        
        # Create a unique name for this PDF's vector store
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_vector_dir = os.path.join(persist_directory, pdf_name)
        print(f"Creating vector store at: {pdf_vector_dir}")
        
        # Create and persist vector store
        try:
            vectorstore = Chroma.from_documents(
                docs, 
                embedding=embedding_model, 
                persist_directory=pdf_vector_dir
            )
            # The vector store is automatically persisted when created with persist_directory
            print("Vector store created and persisted successfully")
        except Exception as vector_error:
            print(f"Error creating vector store: {str(vector_error)}")
            raise ValueError(f"Failed to create vector store: {str(vector_error)}")
        
        end_time = time.time()
        print(f"PDF Processing Time: {end_time - start_time:.2f} seconds")
        print(f"Vector store created successfully at: {pdf_vector_dir}")
        return vectorstore
        
    except Exception as e:
        print(f"\n=== Error in process_pdf ===")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

# Process multiple PDFs and merge into one vectorstore
def process_multiple_pdfs(pdf_paths: List[str], session_id: str = "default-session") -> Chroma:
    start_time = time.time()
    print(f"\n=== Processing multiple PDFs for session {session_id} ===")
    
    # Save all PDFs to session directory
    session_pdf_paths = [save_pdf_to_session(pdf_path, session_id) for pdf_path in pdf_paths]
    
    # Create session-specific directory
    persist_directory = get_session_vector_dir(session_id)
    os.makedirs(persist_directory, exist_ok=True)
    
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    
    for pdf_path in session_pdf_paths:
        pdf_start_time = time.time()
        loader = PyPDFLoader(pdf_path)
        docs = splitter.split_documents(loader.load())
        all_docs.extend(docs)
        pdf_end_time = time.time()
        print(f"Individual PDF Processing Time for {pdf_path}: {pdf_end_time - pdf_start_time:.2f} seconds")
    
    embedding_start_time = time.time()
    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # Create a merged vector store for all PDFs
    merged_vector_dir = os.path.join(persist_directory, "merged_pdfs")
    vectorstore = Chroma.from_documents(
        all_docs, 
        embedding=embedding_model, 
        persist_directory=merged_vector_dir
    )
    
    embedding_end_time = time.time()
    print(f"Embedding and Vectorstore Creation Time: {embedding_end_time - embedding_start_time:.2f} seconds")
    
    end_time = time.time()
    print(f"Total Multiple PDFs Processing Time: {end_time - start_time:.2f} seconds")
    print(f"Merged vector store created at: {merged_vector_dir}")
    return vectorstore

def get_session_pdfs(session_id: str) -> List[str]:
    """Get list of PDFs uploaded for a specific session"""
    session_upload_dir = get_session_upload_dir(session_id)
    if not os.path.exists(session_upload_dir):
        return []
    
    return [f for f in os.listdir(session_upload_dir) if f.lower().endswith('.pdf')]

# Ask query with optional streaming
def ask_medical_query(
    query: str,
    vector_dir: Optional[Union[str, list]] = None,
    session_id: str = "default-session",
    stream: bool = False
):
    try:
        total_start_time = time.time()
        print("\n=== Starting Query Processing ===")
        print(f"Query: {query}")
        print(f"Session ID: {session_id}")
        
        SIMILARITY_THRESHOLD = 0.75
        context = ""
        source_pages = []
        use_vector_context = False
        all_docs = []

        # First, try to find relevant content in PDFs
        if vector_dir:
            print("\n=== Vector Search Process ===")
            vector_search_start_time = time.time()
            embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
            
            # Handle both single directory and list of directories
            vector_dirs = [vector_dir] if isinstance(vector_dir, str) else vector_dir
            
            # If vector_dir is a session ID, use the session's vector stores
            if isinstance(vector_dir, str) and vector_dir == session_id:
                session_vector_dir = get_session_vector_dir(session_id)
                if os.path.exists(session_vector_dir):
                    vector_dirs = [os.path.join(session_vector_dir, d) for d in os.listdir(session_vector_dir) 
                                 if os.path.isdir(os.path.join(session_vector_dir, d))]

            for dir_path in vector_dirs:
                dir_start_time = time.time()
                if os.path.exists(dir_path):
                    print(f"\nProcessing directory: {dir_path}")
                    chroma_vs = Chroma(embedding_function=embedding_model, persist_directory=dir_path)
                    retriever = chroma_vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 5,
                            "fetch_k": 10
                        }
                    )

                    doc_search_start = time.time()
                    retrieved_docs = retriever.get_relevant_documents(query)
                    doc_search_end = time.time()
                    print(f"Document search time: {doc_search_end - doc_search_start:.2f} seconds")
                    
                    all_docs.extend(retrieved_docs)
                
                dir_end_time = time.time()
                print(f"Directory processing time: {dir_end_time - dir_start_time:.2f} seconds")
            
            vector_search_end_time = time.time()
            print(f"\nTotal Vector Search Time: {vector_search_end_time - vector_search_start_time:.2f} seconds")

            if all_docs:
                context_start_time = time.time()
                max_chunks = 3
                selected_docs = all_docs[:max_chunks]
                context = "\n".join([doc.page_content for doc in selected_docs])
                source_pages = [
                    {
                        "page": doc.metadata.get("page"),
                        "source": doc.metadata.get("source", "unknown")
                    }
                    for doc in selected_docs
                ]
                use_vector_context = True
                context_end_time = time.time()
                print('context----------->>>>', context)
                print(f"Context preparation time: {context_end_time - context_start_time:.2f} seconds")
                print("\n=== Using RAG (Retrieval Augmented Generation) ===")
                print(f"Found relevant context from {len(selected_docs)} document chunks")
                print(f"Sources: {[doc.metadata.get('source', 'unknown') for doc in selected_docs]}")
            else:
                print("\n=== No relevant PDF context found - Using General LLM Knowledge ===")
        else:
            print("\n=== No PDF context available - Using General LLM Knowledge ===")

        print("\n=== Response Generation Process ===")
        input_prep_start = time.time()
        
        # Get conversation chain for this session
        conversation_chain = get_conversation_chain(session_id)
        
        if stream:
            def generate():
                print("\n=== Starting Stream Generation ===")
                llm_start_time = time.time()
                
                # Get response from Groq with streaming
                messages = []
                
                # Add system message with context
                system_message = f"""You are a helpful and empathetic medical assistant. Only answer health-related questions (medicine, wellness, biology). Politely decline unrelated topics.

Behavior:
- Greet only if the user says hi/hello/hey. Otherwise, skip greetings and respond directly.
- Never give diagnoses or treatment plans.
- For medicines: explain general uses, dosage ranges, side effects, and precautions. Never suggest specific drugs or doses. Always recommend consulting a doctor or pharmacist.
- For non-health related questions (like programming, technology, etc.): Politely redirect the user to ask health-related questions instead. Do not provide any information about the unrelated topic.

PDF Context:
{context}

Rules:
- If PDF context is provided, answer using ONLY that. Say clearly if it lacks relevant info. Cite context if possible.
- If no context, use your medical knowledge. Don't mention documents.
- For non-health questions, simply say: "I'm a health-focused assistant. I can help you with medical and wellness questions. What health-related topic would you like to know about?"

Always be accurate, concise, safe, and respectful."""
                
                messages.append({"role": "system", "content": system_message})
                
                # Add history messages
                history = DBChatMessageHistory(session_id).messages
                if history:
                    for msg in history:
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        messages.append({"role": role, "content": msg.content})
                
                # Add current input
                messages.append({"role": "user", "content": query})
                
                # Stream the response
                accumulated_response = ""
                for chunk in get_groq_response(messages, stream=True):
                    if isinstance(chunk, dict) and "content" in chunk:
                        content = chunk["content"]
                        accumulated_response += content
                        yield content
                
                # Save the complete response to chat history
                DBChatMessageHistory(session_id).add_ai_message(accumulated_response)
                
                llm_end_time = time.time()
                print(f"LLM Streaming Time: {llm_end_time - llm_start_time:.2f} seconds")
            
            return generate()
        else:
            print("\n=== Starting Non-Stream Generation ===")
            llm_start_time = time.time()
            
            # Get response from Groq
            # Get response from Groq
            final_answer = conversation_chain(query, context)

            # In case final_answer is a generator, convert it to a string
            if hasattr(final_answer, "__iter__") and not isinstance(final_answer, str):
                final_answer = "".join(final_answer)

            llm_end_time = time.time()
            print(f"LLM Processing Time: {llm_end_time - llm_start_time:.2f} seconds")

            # Post-process the response to remove any PDF references if using general knowledge
            if not use_vector_context:
                final_answer = final_answer.replace("PDF", "medical information").replace("document", "information")
                sentences = final_answer.split('.')
                final_answer = '. '.join([s for s in sentences if 'pdf' not in s.lower() and 'document' not in s.lower()])

            print("\n=== Database Logging ===")
            db_start_time = time.time()
            ChatLog.objects.create(
                session_id=session_id,
                user_message=query,
                bot_response=final_answer,
                source_pages=source_pages
            )
            db_end_time = time.time()
            print(f"Database Logging Time: {db_end_time - db_start_time:.2f} seconds")
            
            total_end_time = time.time()
            print("\n=== Summary ===")
            print(f"Response Type: {'RAG with PDF Context' if use_vector_context else 'General LLM Knowledge'}")
            print(f"Total Query Processing Time: {total_end_time - total_start_time:.2f} seconds")
            print("=== End of Processing ===\n")
            
            return final_answer, source_pages

    except Exception as e:
        print(f"\n=== Error Occurred ===")
        print(f"Error in ask_medical_query: {traceback.format_exc()}")
        raise ValueError("Something went wrong while generating the response.")

