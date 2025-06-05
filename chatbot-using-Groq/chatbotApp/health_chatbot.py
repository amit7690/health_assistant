import os, time
import uuid
import json
import traceback
from datetime import datetime
from typing import Optional, Union, List, Dict, Any
import threading

# Document processing imports
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document

# Django and other imports
from chatbotApp.models import ChatLog, ChatHistory
import requests
from functools import lru_cache
import hashlib
import shutil
import io
import pandas as pd
import numpy as np

# Configure pytesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\amikushw\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using pytesseract OCR with enhanced preprocessing"""
    try:
        print(f"\n=== Extracting Text from Image ===")
        print(f"Image path: {image_path}")
        
        # Open and preprocess the image
        image = Image.open(image_path)
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Resize if too small (helps with OCR accuracy)
        min_size = 1000
        if min(image.size) < min_size:
            ratio = min_size / min(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        print(f"Processed image size: {image.size}")
        
        # Extract text with multiple configurations
        configs = [
            '--oem 3 --psm 6',  # Assume uniform text block
            '--oem 3 --psm 3',  # Auto page segmentation
            '--oem 3 --psm 4'   # Assume single column of text
        ]
        
        extracted_texts = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text.strip():
                    extracted_texts.append(text.strip())
            except Exception as config_error:
                print(f"Error with config {config}: {str(config_error)}")
        
        # Combine all extracted texts
        final_text = ' '.join(extracted_texts)
        final_text = ' '.join(final_text.split())  # Normalize whitespace
        
        print("\n=== Extracted Text ===")
        print("Text content:")
        print("-" * 50)
        print(final_text)
        print("-" * 50)
        
        if not final_text:
            print("Warning: No text was extracted from the image")
            return "No text content could be extracted from this image. Please ensure the image is clear and contains readable text."
        else:
            print(f"Successfully extracted {len(final_text)} characters")
            return final_text
    except Exception as e:
        print(f"Error extracting text from image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return "Error processing the image. Please ensure the image is in a supported format and contains readable text."

def extract_tables_from_pdf(pdf_path: str, page_number: int) -> List[str]:
    """Extract tables from a specific page of a PDF using pdfplumber"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if 0 <= page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                tables = page.extract_tables()
                return [str(table) for table in tables]
    except Exception as e:
        print(f"Error extracting tables from PDF: {str(e)}")
    return []

def extract_images_from_pdf(pdf_path: str, page_number: int) -> List[Dict[str, Any]]:
    """Extract images from a specific page of a PDF using PyMuPDF"""
    try:
        pdf_document = fitz.open(pdf_path)
        if 0 <= page_number < len(pdf_document):
            page = pdf_document[page_number]
            image_list = page.get_images()
            images = []
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image)
                images.append({
                    "image_bytes": image_bytes,
                    "extracted_text": text.strip(),
                    "format": base_image["ext"]
                })
            pdf_document.close()
            return images
    except Exception as e:
        print(f"Error extracting images from PDF: {str(e)}")
    return []

def create_document_chunks(text: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """Create optimized chunks from text using RecursiveCharacterTextSplitter"""
    try:
        print("\n=== Creating Document Chunks ===")
        print(f"Input text length: {len(text)} characters")
        
        # Configure splitter for optimal chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for better precision
            chunk_overlap=200,  # Increased overlap for better context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )
        
        # Create document with metadata
        doc = Document(page_content=text, metadata=metadata or {})
        chunks = splitter.split_documents([doc])
        
        # Validate chunks
        if not chunks:
            raise ValueError("No chunks were created from the text")
            
        print(f"Created {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"\n=== Error in create_document_chunks ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

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
- When referencing tables or images from the PDF:
  * For tables: Present the data in a clear, structured format. If the table is complex, break it down into smaller, more digestible parts.
  * For images: Describe the visual content and any text extracted from it. If the image contains important medical information, highlight key points.
  * Always maintain the original meaning and context of tables and images.

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
        self._lock = threading.Lock()
        self._cache = {}
        self._cache_timeout = 30  # seconds
        self._last_update = 0

    def _update_cache(self):
        current_time = time.time()
        if current_time - self._last_update > self._cache_timeout:
            with self._lock:
                msgs = ChatHistory.objects.filter(session_id=self.session_id).order_by("timestamp")
                self._cache = {
                    'messages': [
                        HumanMessage(content=m.message) if m.role == "human" else AIMessage(content=m.message)
                        for m in msgs
                    ],
                    'timestamp': current_time
                }
                self._last_update = current_time

    @property
    def messages(self):
        self._update_cache()
        return self._cache.get('messages', [])

    def add_user_message(self, message: str):
        with self._lock:
            ChatHistory.objects.create(
                session_id=self.session_id,
                role="human",
                message=message,
                timestamp=datetime.now().isoformat()
            )
            self._last_update = 0  # Force cache update

    def add_ai_message(self, message: str):
        with self._lock:
            ChatHistory.objects.create(
                session_id=self.session_id,
                role="ai",
                message=message,
                timestamp=datetime.now().isoformat()
            )
            self._last_update = 0  # Force cache update

    def add_messages(self, messages):
        with self._lock:
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    self.add_user_message(msg.content)
                elif isinstance(msg, AIMessage):
                    self.add_ai_message(msg.content)
            self._last_update = 0  # Force cache update

    def clear(self):
        with self._lock:
            ChatHistory.objects.filter(session_id=self.session_id).delete()
            self._cache = {}
            self._last_update = 0

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
        
        # Load and process the PDF with UnstructuredPDFLoader for better table and image handling
        print("Loading PDF with UnstructuredPDFLoader...")
        try:
            loader = UnstructuredPDFLoader(pdf_path, mode="elements")
            documents = loader.load()
            print(f"Loaded {len(documents)} elements")
            
            # Process tables and images
            processed_docs = []
            for doc in documents:
                if hasattr(doc, 'type'):
                    if doc.type == 'Table':
                        # Extract tables using helper function
                        tables = extract_tables_from_pdf(pdf_path, doc.page_number - 1)
                        if tables:
                            table_text = "Tables found in document:\n"
                            for table in tables:
                                table_text += f"{table}\n\n"
                            doc.page_content = table_text
                        else:
                            doc.page_content = "Table: [Unable to extract table data]"
                            
                    elif doc.type == 'Image':
                        # Extract images using helper function
                        images = extract_images_from_pdf(pdf_path, doc.page_number - 1)
                        if images:
                            image_text = "Images found in document:\n"
                            for img in images:
                                if img['extracted_text']:
                                    image_text += f"Image text: {img['extracted_text']}\n"
                                else:
                                    image_text += "Image: [No text could be extracted]\n"
                            doc.page_content = image_text
                        else:
                            doc.page_content = "Image: [Unable to extract content]"
                
                processed_docs.append(doc)
            
            documents = processed_docs
            
        except Exception as pdf_error:
            print(f"Error loading PDF with UnstructuredPDFLoader: {str(pdf_error)}")
            print("Falling back to PyPDFLoader...")
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                print(f"Loaded {len(documents)} pages with PyPDFLoader")
            except Exception as fallback_error:
                print(f"Error loading PDF with PyPDFLoader: {str(fallback_error)}")
                raise ValueError(f"Failed to load PDF: {str(fallback_error)}")
        
        # Create document chunks
        print("Creating document chunks...")
        all_chunks = []
        for doc in documents:
            chunks = create_document_chunks(
                doc.page_content,
                {
                    "source": pdf_path,
                    "type": "pdf",
                    "filename": os.path.basename(pdf_path),
                    "page": doc.metadata.get("page", 0),
                    "timestamp": datetime.now().isoformat()
                }
            )
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} total chunks")
        
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
                all_chunks, 
                embedding=embedding_model, 
                persist_directory=pdf_vector_dir
            )
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

def process_image(image_path: str, session_id: str = "default-session") -> Chroma:
    start_time = time.time()
    print(f"\n=== Processing Image for session {session_id} ===")
    print(f"Input Image path: {image_path}")
    
    try:
        # Verify image file exists and is readable
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found at: {image_path}")
        
        if not os.access(image_path, os.R_OK):
            raise ValueError(f"Image file is not readable: {image_path}")
            
        print(f"Image file exists and is readable")
        print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Create session-specific vector directory
        persist_directory = get_session_vector_dir(session_id)
        print(f"Creating vector directory: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Process the image
        print("\n=== Processing image with pytesseract ===")
        try:
            # Extract text from image using helper function
            image_text = extract_text_from_image(image_path)
            
            print("\n=== Extracted Text Content ===")
            print("Raw extracted text:")
            print("-" * 50)
            print(image_text)
            print("-" * 50)
            
            if not image_text:
                print("Warning: No text could be extracted from the image")
                image_text = "No text content could be extracted from this image."
            else:
                print(f"Successfully extracted {len(image_text)} characters of text")
            
            # Create metadata
            metadata = {
                "source": image_path,
                "type": "image",
                "filename": os.path.basename(image_path),
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(image_path),
                "has_text": bool(image_text and image_text != "No text content could be extracted from this image.")
            }
            
            # Create document chunks
            print("\n=== Creating document chunks ===")
            docs = create_document_chunks(image_text, metadata)
            print(f"Created {len(docs)} chunks")
            
            if not docs:
                raise ValueError("Failed to create document chunks from image text")
            
            # Print chunk contents for debugging
            print("\n=== Document Chunks ===")
            for i, doc in enumerate(docs):
                print(f"\nChunk {i+1}:")
                print("-" * 30)
                print(doc.page_content)
                print("-" * 30)
            
            # Create embeddings
            print("\n=== Creating embeddings ===")
            try:
                embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
            except Exception as embed_error:
                print(f"Error initializing embedding model: {str(embed_error)}")
                raise ValueError(f"Failed to initialize embedding model: {str(embed_error)}")
            
            # Create a unique name for this image's vector store
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_vector_dir = os.path.join(persist_directory, image_name)
            print(f"Creating vector store at: {image_vector_dir}")
            
            # Create and persist vector store
            try:
                vectorstore = Chroma.from_documents(
                    docs, 
                    embedding=embedding_model, 
                    persist_directory=image_vector_dir
                )
                print("Vector store created and persisted successfully")
                
                # Verify vector store contents with meaningful test queries
                print("\n=== Verifying Vector Store Contents ===")
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3}  # Get top 3 most relevant documents
                )
                
                # Use the first few words of the extracted text as test queries
                test_queries = []
                if image_text and image_text != "No text content could be extracted from this image.":
                    # Split text into words and create test queries
                    words = image_text.split()
                    if len(words) > 3:
                        test_queries = [
                            " ".join(words[:3]),  # First 3 words
                            " ".join(words[-3:]),  # Last 3 words
                            words[0] if words else "test"  # First word
                        ]
                else:
                    test_queries = ["test"]
                
                print("\nTesting vector store with queries:")
                for query in test_queries:
                    print(f"\nQuery: '{query}'")
                    test_docs = retriever.get_relevant_documents(query)
                    if test_docs:
                        print("Found matching documents:")
                        for i, doc in enumerate(test_docs):
                            print(f"\nDocument {i+1}:")
                            print("-" * 30)
                            print(f"Content: {doc.page_content}")
                            print(f"Score: {doc.metadata.get('score', 'N/A')}")
                            print("-" * 30)
                    else:
                        print("No matching documents found")
                
                if not any(test_docs for test_docs in [retriever.get_relevant_documents(q) for q in test_queries]):
                    print("\nWarning: Vector store verification failed - no documents found for any test query")
                else:
                    print("\nVector store verification successful - documents are searchable")
                
            except Exception as vector_error:
                print(f"Error creating vector store: {str(vector_error)}")
                raise ValueError(f"Failed to create vector store: {str(vector_error)}")
            
            end_time = time.time()
            print(f"\n=== Image Processing Summary ===")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print(f"Vector store created at: {image_vector_dir}")
            print(f"Number of chunks created: {len(docs)}")
            return vectorstore
            
        except Exception as img_error:
            print(f"\n=== Error processing image ===")
            print(f"Error type: {type(img_error).__name__}")
            print(f"Error details: {str(img_error)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Failed to process image: {str(img_error)}")
        
    except Exception as e:
        print(f"\n=== Error in process_image ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to process image: {str(e)}")

def process_file(file_path: str, session_id: str = "default-session") -> Chroma:
    """Process a file (PDF or image) and create a vector store with enhanced content handling"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            raise ValueError(f"File not found at: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise ValueError(f"File is not readable: {file_path}")
            
        # Create session-specific vector directory
        persist_directory = get_session_vector_dir(session_id)
        os.makedirs(persist_directory, exist_ok=True)
        
        # Process based on file type
        if file_extension == '.pdf':
            return process_pdf(file_path, session_id)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return process_image(file_path, session_id)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        print(f"\n=== Error in process_file ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to process file: {str(e)}")

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
        
        # Extract file name from vector_dir if it exists
        file_name = None
        if vector_dir and isinstance(vector_dir, str):
            file_name = os.path.basename(vector_dir)
        
        SIMILARITY_THRESHOLD = 0.65  # Lowered threshold for better recall
        context = ""
        source_pages = []
        use_vector_context = False
        all_docs = []
        is_image_query = False

        # First, try to find relevant content in files
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
                        search_type="mmr",  # Using MMR for better diversity
                        search_kwargs={
                            "k": 10,  # Increased number of documents to retrieve
                            "fetch_k": 20,  # Increased number of documents to fetch
                            "lambda_mult": 0.5  # Adjusted for better diversity
                        }
                    )

                    doc_search_start = time.time()
                    retrieved_docs = retriever.get_relevant_documents(query)
                    doc_search_end = time.time()
                    print(f"Document search time: {doc_search_end - doc_search_start:.2f} seconds")
                    
                    # Filter documents by similarity threshold
                    filtered_docs = []
                    for doc in retrieved_docs:
                        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                            if doc.metadata['score'] >= SIMILARITY_THRESHOLD:
                                filtered_docs.append(doc)
                        else:
                            filtered_docs.append(doc)
                    
                    # Check if this is an image query
                    if filtered_docs and filtered_docs[0].metadata.get('type') == 'image':
                        is_image_query = True
                        print("Detected image-based query")
                    
                    all_docs.extend(filtered_docs)
                
                dir_end_time = time.time()
                print(f"Directory processing time: {dir_end_time - dir_start_time:.2f} seconds")
            
            vector_search_end_time = time.time()
            print(f"\nTotal Vector Search Time: {vector_search_end_time - vector_search_start_time:.2f} seconds")

            # Process retrieved documents
            if all_docs:
                print(f"\nFound {len(all_docs)} relevant documents")
                use_vector_context = True
                
                # Sort documents by score if available
                if all_docs[0].metadata.get('score'):
                    all_docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
                
                # Combine document content with metadata
                context_parts = []
                for doc in all_docs:
                    source = doc.metadata.get('source', 'Unknown source')
                    page = doc.metadata.get('page', 'Unknown page')
                    score = doc.metadata.get('score', 0)
                    doc_type = doc.metadata.get('type', 'unknown')
                    has_text = doc.metadata.get('has_text', True)
                    
                    if doc_type == 'image':
                        if has_text:
                            context_part = f"Image content from {source} (Relevance: {score:.2f}):\n{doc.page_content}\n"
                        else:
                            context_part = f"Image from {source} (Relevance: {score:.2f}): No text could be extracted from this image.\n"
                    else:
                        context_part = f"Content from {source} (Page {page}, Relevance: {score:.2f}):\n{doc.page_content}\n"
                    
                    context_parts.append(context_part)
                
                context = "\n---\n".join(context_parts)
                print(f"Total context length: {len(context)} characters")
            else:
                print("No relevant documents found in vector store")
                context = "No relevant information found in the uploaded documents."
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": f"""You are a helpful and empathetic medical assistant. Only answer health-related questions (medicine, wellness, biology). Politely decline unrelated topics.

Behavior:
- Greet only if the user says hi/hello/hey. Otherwise, skip greetings and respond directly.
- Never give diagnoses or treatment plans.
- For medicines: explain general uses, dosage ranges, side effects, and precautions. Never suggest specific drugs or doses. Always recommend consulting a doctor or pharmacist.

Context from uploaded documents:
{context}

Rules:
- If context is provided, answer using ONLY that. Say clearly if it lacks relevant info. Cite context if possible.
- If no context, use your medical knowledge. Don't mention documents.
- If the context is about an image:
  * If text was extracted, use that text to answer the question.
  * If no text was extracted, explain that the image might not contain readable text or might need better quality.
  * If the image appears to be medical-related (like a medical chart, scan, or diagram), describe what you can infer from the context about its medical relevance.
- Always maintain a helpful and informative tone, even when explaining limitations in image processing.

Always be accurate, concise, safe, and respectful."""},
            {"role": "user", "content": query}
        ]

        # Get response from Groq
        if stream:
            def generate():
                try:
                    for chunk in get_groq_response(messages, stream=True):
                        if chunk.get("content"):
                            yield chunk["content"]
                except Exception as e:
                    print(f"Error in stream generation: {str(e)}")
                    yield f"Error generating response: {str(e)}"
            return generate()
        else:
            response = get_groq_response(messages)
            return response

    except Exception as e:
        print(f"\n=== Error in ask_medical_query ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error processing your query: {str(e)}"

