import os
import shutil
import tempfile
import json
import base64
import pickle
from typing import List, Dict, Any, Optional

import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import requests
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vapi imports
from vapi_python import Vapi

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ========== Environment Variables ==========
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_DEPLOYMENT_EMBEDDINGS = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")

if not VAPI_API_KEY:
    raise ValueError("VAPI_API_KEY environment variable is required for Vapi.")
if not all(
    [
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_DEPLOYMENT,
        AZURE_DEPLOYMENT_EMBEDDINGS,
    ]
):
    raise ValueError("Missing some Azure OpenAI environment variables. Make sure all are set.")

# ========== FastAPI and Templating Setup ==========
app = FastAPI(title="RAG PDF Document Assistant with Voice")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========== Vapi Client ==========
vapi_client = Vapi(api_key=VAPI_API_KEY)

# ========== Data/Index Paths ==========
os.makedirs("faiss_index_upgraded_english", exist_ok=True)
INDEX_PATH = "faiss_index_upgraded_english/index.faiss"
METADATA_PATH = "faiss_index_upgraded_english/index.pkl"

# ========== Document Model for Storage ==========
class DocumentChunk(BaseModel):
    doc_id: str                 # Unique doc ID or filename
    topic: Optional[str]        # Topic/category assigned to this doc
    chunk_id: int               # The nth chunk of the doc
    text: str                   # The chunk text
    title: Optional[str] = None # Optional title for display

# ========== Document Processing Class ==========
class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        extracted_text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                extracted_text += page.get_text()
        return extracted_text

    def split_into_chunks(self, text: str, doc_id: str, topic: str) -> List[DocumentChunk]:
        """Split a single document's text into chunks, return as DocumentChunk list."""
        chunks_list = []
        splitted = self.text_splitter.split_text(text)
        for idx, chunk_text in enumerate(splitted):
            chunk = DocumentChunk(
                doc_id=doc_id,
                topic=topic,
                chunk_id=idx,
                text=chunk_text,
                title=doc_id  # optional
            )
            chunks_list.append(chunk)
        return chunks_list

# ========== Embedding Generator Class (Azure OpenAI) ==========
class AzureOpenAIEmbedding:
    def __init__(self):
        self.api_key = AZURE_OPENAI_API_KEY
        self.endpoint = AZURE_OPENAI_ENDPOINT
        self.api_version = AZURE_OPENAI_API_VERSION
        self.deployment = AZURE_DEPLOYMENT_EMBEDDINGS
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts via Azure OpenAI Embeddings."""
        embeddings = []
        batch_size = 5
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            url = f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
            
            # Azure can do batch embeddings:
            payload = {"input": batch}
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                for data_item in result["data"]:
                    embedding = data_item["embedding"]
                    embeddings.append(embedding)
            else:
                raise ValueError(f"Azure Embedding error: {response.text}")
        return np.array(embeddings, dtype=np.float32)

# ========== FAISS Vector Database ==========
class FAISSVectorStore:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, doc_chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add documents + embeddings to FAISS index."""
        # Convert doc_chunks to dict
        chunk_dicts = [chunk.dict() for chunk in doc_chunks]

        if not self.index.is_trained and len(chunk_dicts) > 0:
            # Re-init index
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        self.documents.extend(chunk_dicts)

    def search(self, query_embedding: np.ndarray, k=5, topic_filter: str = "") -> List[Dict[str, Any]]:
        """Search for similar documents, optionally filter by topic."""
        D, I = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx, distance in zip(I[0], D[0]):
            if idx < len(self.documents):
                doc_info = self.documents[idx]
                # If topic filter is given and doesn't match doc's topic, skip
                if topic_filter and doc_info.get("topic", "") != topic_filter:
                    continue
                doc_copy = {**doc_info}
                doc_copy["score"] = float(distance)
                results.append(doc_copy)
        return results

    def save_index(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                self.documents = pickle.load(f)
        else:
            raise FileNotFoundError("No existing index.faiss or index.pkl found in faiss_index_upgraded_english folder")

# ========== Vapi AI Integration ==========
class VapiAIIntegration:
    def __init__(self):
        self.vapi_client = vapi_client
    
    def generate_completion(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Use Vapi completions to get an LLM-based answer."""
        context_text = ""
        if context:
            # Concatenate relevant chunks
            context_text = "\n\nRelevant context:\n" + "\n---\n".join([doc["text"] for doc in context])

        full_prompt = f"{prompt}\n{context_text}"
        # Possibly some system instructions or role prompt can be added:
        messages = [
            {"role": "system", "content": "You are a helpful AI Assistant."},
            {"role": "user", "content": full_prompt}
        ]
        resp = self.vapi_client.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        return resp.choices[0].message.content

    def generate_speech(self, text: str, voice_id: str = "echo") -> Dict[str, Any]:
        """Generate TTS audio using Vapi TTS."""
        try:
            response = self.vapi_client.tts.create(
                text=text,
                voice_id=voice_id,  # default "echo"
                output_format="mp3"
            )
            # Construct return payload
            result = {"success": True}
            if hasattr(response, "audio") and response.audio:
                audio_base64 = base64.b64encode(response.audio).decode("utf-8")
                result["audio_data"] = audio_base64
            if hasattr(response, "url") and response.url:
                result["audio_url"] = response.url
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_voices(self) -> List[Dict[str, Any]]:
        """List available TTS voices in Vapi."""
        try:
            return self.vapi_client.tts.list_voices()
        except Exception as e:
            print(f"Could not list voices: {e}")
            return []

# ========== RAG Pipeline ==========
class RAGPipeline:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedding_generator = AzureOpenAIEmbedding()
        self.vector_store = FAISSVectorStore()
        self.vapi = VapiAIIntegration()

        # Load existing index if available
        try:
            self.vector_store.load_index()
            print("Loaded existing FAISS index from faiss_index_upgraded_english.")
        except:
            print("No existing index found; starting fresh.")

    def index_pdf_documents(self, pdf_files: List[bytes], file_names: List[str], topic: str):
        """
        For each PDF file:
          - Extract text
          - Split into chunks
          - Embed
          - Save to FAISS
        """
        all_chunks = []
        for pdf_content, filename in zip(pdf_files, file_names):
            text = self.processor.extract_text_from_pdf(pdf_content)
            # Split text
            chunks = self.processor.split_into_chunks(text, doc_id=filename, topic=topic)
            all_chunks.extend(chunks)

        # Generate embeddings for all chunks
        if all_chunks:
            texts = [c.text for c in all_chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            self.vector_store.add_documents(all_chunks, embeddings)
            self.vector_store.save_index()

        return len(all_chunks)

    def query_docs(self, question: str, top_k: int = 5, voice_id: Optional[str] = None, topic_filter: str = ""):
        """
        1) Embed question
        2) Retrieve top_k documents (optionally topic-filtered)
        3) Use Vapi completions to get final answer
        4) Optionally TTS the answer
        """
        # Embed user question
        q_embed = self.embedding_generator.generate_embeddings([question])[0]

        # Retrieve docs
        docs = self.vector_store.search(q_embed, k=top_k, topic_filter=topic_filter)
        
        # Get answer from LLM
        answer_text = self.vapi.generate_completion(question, context=docs)

        # Optionally TTS
        speech_result = None
        if voice_id:
            speech_result = self.vapi.generate_speech(answer_text, voice_id=voice_id)

        return {
            "answer": answer_text,
            "sources": docs,
            "speech": speech_result
        }

    def get_voices(self):
        return self.vapi.list_voices()

    def get_topics(self):
        """Return list of unique topics in docs."""
        topic_set = set()
        for doc in self.vector_store.documents:
            if doc.get("topic"):
                topic_set.add(doc["topic"])
        return list(topic_set)

rag_pipeline = RAGPipeline()

# ========== Pydantic models for query request ==========
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    voice_id: Optional[str] = None
    topic: Optional[str] = ""

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    speech: Optional[Dict[str, Any]] = None

# ========== Routes ==========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    topic: str = Form(...)
):
    """Upload PDF files with associated topic, index them in FAISS."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    temp_dir = tempfile.mkdtemp()
    pdf_contents = []
    file_names = []
    try:
        for f in files:
            file_path = os.path.join(temp_dir, f.filename)
            with open(file_path, "wb") as bf:
                shutil.copyfileobj(f.file, bf)
            with open(file_path, "rb") as rf:
                pdf_bytes = rf.read()
            pdf_contents.append(pdf_bytes)
            file_names.append(f.filename)
    finally:
        shutil.rmtree(temp_dir)

    # Process & Index PDFs
    chunks_added = rag_pipeline.index_pdf_documents(pdf_contents, file_names, topic)
    return {"message": f"Processed {chunks_added} chunks from {len(files)} PDF(s)."}

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        result = rag_pipeline.query_docs(
            question=request.question,
            top_k=request.top_k,
            voice_id=request.voice_id,
            topic_filter=request.topic
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def get_voices():
    """Return available voices from Vapi."""
    voices = rag_pipeline.get_voices()
    return {"voices": voices}

@app.get("/api/topics")
async def get_topics():
    """Return the list of unique document topics."""
    topics = rag_pipeline.get_topics()
    return {"topics": topics}

# ========== Main Entry Point ==========
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)