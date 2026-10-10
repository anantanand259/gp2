"""
═══════════════════════════════════════════════════════════════
  GPA RAG Backend — Production Flask Server
  Hybrid Retrieval (ChromaDB + BM25) + Gemini LLM
  Supports: Images (JPG/PNG), PDFs, JSON, Plain Text

  Usage:
    1. pip install -r requirements.txt
    2. Set your API key:  set GOOGLE_API_KEY=your_key_here
    3. python server.py
    4. Place documents in ./knowledge_base/input_docs/
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import shutil
import logging
import traceback
import requests
import hashlib
from threading import RLock
from werkzeug.utils import secure_filename
from backend_errors import describe_query_error
from pathlib import Path
from typing import List, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
try:
    from pyngrok import ngrok, conf
    NGROK_SUPPORT = True
except ImportError:
    NGROK_SUPPORT = False

# Load environment variables from .env file in parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('GPA-RAG')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# ┌──────────────────────────────────────────────────────────┐
# │  🔑  Place your Gemini API Key here or set env var       │
# └──────────────────────────────────────────────────────────┘
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or 'YOUR_GOOGLE_KEY_HERE'
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'YOUR_OPENROUTER_KEY_HERE')
OPENROUTER_MODEL = 'meta-llama/llama-3.3-70b-instruct'
NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', 'YOUR_NGROK_AUTH_TOKEN')

# ┌──────────────────────────────────────────────────────────┐
# │  📂  STORAGE CONFIGURATION                               │
# └──────────────────────────────────────────────────────────┘
# If you use Google Drive for Desktop, set this to your Drive path.
# Example: 'G:/My Drive/GPA_Knowledge_Base'
DRIVE_PATH = os.environ.get('GOOGLE_DRIVE_PATH')

if DRIVE_PATH:
    BASE_DIR = Path(DRIVE_PATH)
    log.info(f'📁 Using Google Drive storage: {BASE_DIR}')
else:
    BASE_DIR = Path(__file__).parent
    log.info(f'📁 Using local storage: {BASE_DIR}')

KB_DIR         = BASE_DIR / 'knowledge_base'
INPUT_DIR      = KB_DIR / 'input_docs'
PROCESSED_DIR  = KB_DIR / 'processed_docs'
CHROMA_DIR     = KB_DIR / 'chroma_db'
LOG_FILE       = KB_DIR / 'processed_log.json'
BM25_FILE      = KB_DIR / 'bm25_docs.json'

# Server config
HOST = '0.0.0.0'
PORT = int(os.environ.get('RAG_PORT', 5000))

# Allowed origins for CORS
ALLOWED_ORIGINS = [
    'https://anantanand259.github.io',
    'http://localhost:3000',
    'http://localhost:5500',
    'http://127.0.0.1:5500',
    'http://localhost:8080',
    'http://127.0.0.1:8080',
    '*',  # Allow all during development — restrict in production
]

# Logger moved up to avoid NameError during initialization

# ─────────────────────────────────────────────────────────────
# CREATE DIRECTORIES
# ─────────────────────────────────────────────────────────────
for d in [INPUT_DIR, PROCESSED_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# os.environ['GOOGLE_API_KEY'] is already set from .env or default above

# ─────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

client = genai.Client(http_options=types.HttpOptions(timeout=25000))
log.info('✅ Gemini client ready.')

# ─────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA for structured VLM extraction
# ─────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

class SignatoryEntity(BaseModel):
    name_or_designation: str = Field(description="Official designation like 'प्राचार्य'")
    organization: Optional[str] = Field(default=None, description='Institution name if mentioned')
    date_signed:  Optional[str] = Field(default=None, description='Date near signature if visible')

class AcademicNoticeSchema(BaseModel):
    issuing_authority:  Optional[str]                    = None
    reference_number:   Optional[str]                    = None
    date_issued:        Optional[str]                    = None
    subject_line:       Optional[str]                    = None
    target_audience:    Optional[List[str]]              = None
    main_body_content:  Optional[str]                    = None
    signatories:        Optional[List[SignatoryEntity]]   = None
    distribution_list:  Optional[List[str]]              = None
    document_type:      Optional[str]                    = None
    extra_fields:       Optional[str]                    = None

log.info('✅ Pydantic schema defined.')

# ─────────────────────────────────────────────────────────────
# VLM EXTRACTION MODULE (for images)
# ─────────────────────────────────────────────────────────────
from PIL import Image

EXTRACTION_PROMPT = '''
You are an intelligent document understanding system for academic/administrative documents.

STEP 1 — Classify document: notice | office_order | exam_schedule | email | tabular_data | unknown
STEP 2 — HIGH PRIORITY: extract reference_number, date_issued, issuing_authority accurately.
STEP 3 — main_body_content MUST contain ALL readable text. Never return null here.
STEP 4 — Preserve Hindi and English exactly. Tables in Markdown. No hallucination.
STEP 5 — extra_fields: JSON string for tables or unknown structures.
Return valid JSON only.
'''

def extract_from_image(image_path: str) -> AcademicNoticeSchema:
    """Extract structured data from an image document using Gemini VLM."""
    document_image = Image.open(image_path)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[document_image, EXTRACTION_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=AcademicNoticeSchema,
            temperature=0.0
        )
    )
    parsed = response.parsed
    if not parsed.main_body_content:
        parsed.main_body_content = ''
    return parsed

log.info('✅ VLM extraction module ready.')

# ─────────────────────────────────────────────────────────────
# PDF EXTRACTION MODULE
# ─────────────────────────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
    log.info('✅ PDF support enabled (PyPDF2).')
except ImportError:
    PDF_SUPPORT = False
    log.warning('⚠️  PyPDF2 not installed — PDF ingestion disabled.')

def extract_from_pdf(pdf_path: str) -> AcademicNoticeSchema:
    """Extract text from a PDF file."""
    if not PDF_SUPPORT:
        raise RuntimeError('PyPDF2 not installed. Run: pip install PyPDF2')

    reader = PdfReader(pdf_path)
    full_text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + '\n\n'

    full_text = full_text.strip()
    if not full_text or any(not (page.extract_text() or '').strip() for page in reader.pages):
        # Scanned PDFs (or mixed text/scanned pages) need visual extraction.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=Path(pdf_path).read_bytes(), mime_type='application/pdf'), EXTRACTION_PROMPT],
            config=types.GenerateContentConfig(response_mime_type='application/json',
                                              response_schema=AcademicNoticeSchema, temperature=0)
        )
        if not response.parsed or not response.parsed.main_body_content:
            raise ValueError(f'No readable content extracted from PDF: {Path(pdf_path).name}')
        return response.parsed

    return AcademicNoticeSchema(
        main_body_content=full_text,
        document_type='pdf_document',
        subject_line=Path(pdf_path).stem.replace('_', ' ').replace('-', ' ').title()
    )

# ─────────────────────────────────────────────────────────────
# JSON EXTRACTION MODULE
# ─────────────────────────────────────────────────────────────
def extract_from_json(json_path: str) -> List[AcademicNoticeSchema]:
    """Extract structured data from a JSON file. Supports single object or array."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    results = []
    for item in data:
        if isinstance(item, str):
            # Plain text entries in an array
            results.append(AcademicNoticeSchema(
                main_body_content=item,
                document_type='json_text_entry',
                subject_line='Knowledge Base Entry'
            ))
        elif isinstance(item, dict):
            # Try to map common fields
            content = item.get('content') or item.get('text') or item.get('body') or json.dumps(item, ensure_ascii=False)
            title = item.get('title') or item.get('subject') or item.get('name') or 'Knowledge Base Entry'
            results.append(AcademicNoticeSchema(
                main_body_content=content,
                subject_line=title,
                date_issued=item.get('date') or item.get('date_issued'),
                reference_number=item.get('reference') or item.get('ref') or item.get('id'),
                issuing_authority=item.get('authority') or item.get('author') or item.get('source'),
                document_type='json_entry'
            ))

    return results

# ─────────────────────────────────────────────────────────────
# TEXT EXTRACTION MODULE
# ─────────────────────────────────────────────────────────────
def extract_from_text(txt_path: str) -> AcademicNoticeSchema:
    """Extract content from a plain text file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f'Empty text file: {txt_path}')

    return AcademicNoticeSchema(
        main_body_content=content,
        document_type='text_document',
        subject_line=Path(txt_path).stem.replace('_', ' ').replace('-', ' ').title()
    )

# ─────────────────────────────────────────────────────────────
# CHUNKING MODULE
# ─────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_chunks(structured_notice: AcademicNoticeSchema) -> List[Document]:
    """Create contextual chunks from structured notice data."""
    global_metadata = {
        'issuing_authority': structured_notice.issuing_authority or 'Unknown',
        'reference_number':  structured_notice.reference_number or 'UNKNOWN_REF',
        'date_issued':       structured_notice.date_issued or 'Unknown',
        'subject':           structured_notice.subject_line or 'General Administrative Notice'
    }

    context_header = (
        f"Notice Reference: {global_metadata['reference_number']}, "
        f"Date: {global_metadata['date_issued']}, "
        f"Subject: {global_metadata['subject']}.\nContent: "
    )

    body_text = (structured_notice.main_body_content or '').strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=['\n\n', '\n', '।', '.', ' ', '']
    )

    return [
        Document(page_content=context_header + chunk, metadata=global_metadata)
        for chunk in splitter.split_text(body_text)
    ]

log.info('✅ Chunking module ready.')

# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL (BAAI/bge-m3 — multilingual)
# ─────────────────────────────────────────────────────────────
from langchain_core.embeddings import Embeddings
import numpy as np

class OnnxMiniLMEmbeddings(Embeddings):
    """
    Lightweight, high-performance ONNX embeddings for all-MiniLM-L6-v2.
    Runs on CPU via onnxruntime + tokenizers without requiring PyTorch,
    avoiding Windows 11 Application Control DLL policy blocks.
    """
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', subfolder='onnx', filename='model.onnx')
        tokenizer_path = hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', filename='tokenizer.json')

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(length=256)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        encodings = [self.tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self.session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        })
        token_embeddings = outputs[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-12, a_max=None)
        return (pooled / norm).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

log.info('⏳ Loading embedding model (all-MiniLM-L6-v2 ONNX)...')
embedding_model = OnnxMiniLMEmbeddings()
log.info('✅ Embedding model loaded.')

# ─────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL SYSTEM (ChromaDB + BM25)
# ─────────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever

def load_processed_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_processed_log(log_data: list):
    pending = LOG_FILE.with_suffix('.tmp')
    with open(pending, 'w', encoding='utf-8') as f:
        json.dump(log_data, f)
    pending.replace(LOG_FILE)

def load_bm25_docs() -> list:
    if BM25_FILE.exists():
        with open(BM25_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_bm25_docs(docs: list):
    pending = BM25_FILE.with_suffix('.tmp')
    with open(pending, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    pending.replace(BM25_FILE)

def build_hybrid_retriever(new_documents: list, replace_entry_id=None):
    """Build or rebuild the hybrid retriever with optional new documents."""
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model
    )

    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': 12})

    # Merge BM25 docs
    existing_docs = load_bm25_docs()
    if replace_entry_id:
        existing_docs = [d for d in existing_docs if d.get('metadata', {}).get('entry_id') != replace_entry_id]
    new_doc_dicts = [{'content': d.page_content, 'metadata': d.metadata} for d in new_documents]
    unique_docs = {}
    for doc in existing_docs + new_doc_dicts:
        doc['metadata'] = {k: v for k, v in doc.get('metadata', {}).items() if v is not None}
        identity = hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        unique_docs[identity] = doc
    all_docs = list(unique_docs.values())
    if all_docs:
        # Stable IDs make retries safe and restore vector data from the lexical
        # index if an earlier upload only populated one of the two stores.
        vector_store.add_documents([
            Document(page_content=d['content'], metadata=d['metadata']) for d in all_docs
        ], ids=list(unique_docs))
    if replace_entry_id:
        old_ids = vector_store.get(where={'entry_id': replace_entry_id})['ids']
        obsolete = [doc_id for doc_id in old_ids if doc_id not in unique_docs]
        if obsolete:
            vector_store.delete(ids=obsolete)
    save_bm25_docs(all_docs)

    retrievers = [semantic_retriever]
    weights    = [1.0]

    if all_docs:
        bm25_retriever = BM25Retriever.from_documents([
            Document(page_content=d['content'], metadata=d.get('metadata', {})) for d in all_docs
        ])
        bm25_retriever.k = 12
        retrievers.append(bm25_retriever)
        weights = [0.6, 0.4]

    return EnsembleRetriever(retrievers=retrievers, weights=weights)

log.info('✅ Hybrid retrieval module ready.')

# ─────────────────────────────────────────────────────────────
# DOCUMENT INGESTION — Process all files in input_docs/
# ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    'image': ('.jpg', '.jpeg', '.png', '.webp', '.bmp'),
    'pdf':   ('.pdf',),
    'json':  ('.json',),
    'text':  ('.txt', '.md', '.csv'),
}

def get_file_type(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    for ftype, extensions in SUPPORTED_EXTENSIONS.items():
        if ext in extensions:
            return ftype
    return None

def ingest_documents(fail_on_error=False):
    """Scan input_docs/ for new files and process them."""
    processed_files = load_processed_log()
    all_files = [f for f in os.listdir(INPUT_DIR) if not f.startswith('.')]
    new_files = [f for f in all_files if f not in processed_files and get_file_type(f)]

    log.info(f'🔍 Found {len(new_files)} new file(s) to process')

    documents_to_add = []

    for i, filename in enumerate(new_files, 1):
        filepath = str(INPUT_DIR / filename)
        ftype = get_file_type(filename)
        log.info(f'  [{i}/{len(new_files)}] Processing: {filename} ({ftype})')

        try:
            notices = []

            if ftype == 'image':
                notices = [extract_from_image(filepath)]
            elif ftype == 'pdf':
                notices = [extract_from_pdf(filepath)]
            elif ftype == 'json':
                notices = extract_from_json(filepath)
            elif ftype == 'text':
                notices = [extract_from_text(filepath)]

            file_chunks = []
            for notice in notices:
                chunks = create_chunks(notice)
                for chunk in chunks:
                    chunk.metadata['filename'] = filename
                file_chunks.extend(chunks)

            if not file_chunks:
                raise ValueError(f'No readable content in {filename}')

            # Index first. A file is marked processed only after indexing succeeds.
            build_hybrid_retriever(file_chunks)
            documents_to_add.extend(file_chunks)

            # Move to processed
            shutil.move(filepath, str(PROCESSED_DIR / filename))
            processed_files.append(filename)
            log.info(f'     ✅ Extracted {sum(len(create_chunks(n)) for n in notices)} chunk(s)')

        except Exception as e:
            log.error(f'     ❌ Failed: {e}')
            log.debug(traceback.format_exc())
            if fail_on_error:
                raise

    save_processed_log(processed_files)
    return documents_to_add

# Run initial ingestion
log.info('⏳ Running initial document ingestion...')
initial_docs = ingest_documents()
hybrid_retriever = build_hybrid_retriever([])
kb_lock = RLock()
log.info(f'✅ Retriever ready. New chunks: {len(initial_docs)}, Total BM25: {len(load_bm25_docs())}')

# ─────────────────────────────────────────────────────────────
# RAG QUERY FUNCTION
# ─────────────────────────────────────────────────────────────
from rag_answers import answer_query


def generate_rag_answer(user_query: str) -> dict:
    with kb_lock:
        retriever = hybrid_retriever
        documents = [Document(page_content=d['content'], metadata=d.get('metadata', {}))
                     for d in load_bm25_docs()]
    return answer_query(user_query, retriever, client, OPENROUTER_API_KEY, OPENROUTER_MODEL,
                        all_documents=documents)


log.info('✅ RAG query function ready.')

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
CORS(app, resources={r'/api/*': {'origins': ALLOWED_ORIGINS}})


@app.route('/')
def root():
    return jsonify({
        'service': 'GPA RAG Backend',
        'version': '3.0',
        'status': 'running',
        'endpoints': [
            'GET  /api/health',
            'GET  /api/rag/stats',
            'POST /api/rag/query',
            'POST /api/rag/ingest',
            'POST /api/rag/add-text',
        ]
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'GPA RAG Backend',
        'version': '3.0',
        'retriever_ready': hybrid_retriever is not None,
        'total_chunks': len(load_bm25_docs()),
        'total_documents': len(load_processed_log())
    })


@app.route('/api/rag/stats', methods=['GET'])
def stats():
    try:
        bm25_docs = load_bm25_docs()
        processed = load_processed_log()
        return jsonify({
            'total_chunks':    len(bm25_docs),
            'total_documents': len(processed),
            'processed_files': processed,
            'status':          'ready' if hybrid_retriever else 'not_initialized'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    """Main RAG query endpoint — called by the chatbot frontend."""
    if hybrid_retriever is None:
        return jsonify({'error': 'RAG not initialized. Please restart the server.'}), 503

    data = request.get_json(silent=True)
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" field.'}), 400

    user_query = str(data['query']).strip()
    if not user_query:
        return jsonify({'error': 'Query cannot be empty.'}), 400

    try:
        log.info(f'📩 Query: {user_query[:80]}...')
        result = generate_rag_answer(user_query)
        log.info(f'📤 Response: {len(result["answer"])} chars, {result["chunk_count"]} sources')
        return jsonify(result)
    except Exception as e:
        log.error(f'❌ Query failed: {e}')
        log.debug(traceback.format_exc())
        payload, status = describe_query_error(e)
        return jsonify(payload), status


@app.route('/api/rag/ingest', methods=['POST'])
def trigger_ingest():
    """Re-scan input_docs/ and process new files. Optionally accepts a file upload."""
    global hybrid_retriever
    try:
        # Check if a file was uploaded in the request
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                # Save the uploaded file to the input_docs directory
                filename = secure_filename(file.filename)
                if not filename or not get_file_type(filename):
                    return jsonify({'error': 'Unsupported file type. Use PDF, image, JSON, TXT, MD or CSV.'}), 400
                if filename in load_processed_log():
                    return jsonify({'error': 'A file with this name is already indexed. Rename the new notice before uploading.'}), 409
                save_path = INPUT_DIR / filename
                file.save(str(save_path))
                log.info(f'📥 Uploaded new file to input_docs: {file.filename}')

        # Run ingestion on the directory
        with kb_lock:
            new_docs = ingest_documents(fail_on_error=True)
            hybrid_retriever = build_hybrid_retriever([])
        if 'file' in request.files and not new_docs:
            return jsonify({'error': 'No readable content was indexed from the uploaded file.'}), 422
        
        return jsonify({
            'status': 'ok',
            'new_chunks': len(new_docs),
            'total_chunks': len(load_bm25_docs()),
            'total_documents': len(load_processed_log())
        })
    except Exception as e:
        log.error(f'❌ Ingestion failed: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag/add-text', methods=['POST'])
def add_text_entry():
    """Add a plain-text knowledge base entry (no file needed)."""
    global hybrid_retriever
    data = request.get_json(silent=True)
    if not data or 'content' not in data:
        return jsonify({'error': 'Missing "content" field.'}), 400

    title   = data.get('title', 'Manual Entry')
    content = data['content']
    if not isinstance(content, str) or not content.strip():
        return jsonify({'error': 'Content must be non-empty text.'}), 400
    date    = data.get('date', 'N/A')
    ref     = data.get('reference', 'MANUAL')

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks   = splitter.split_text(content)
        header   = f'Notice Reference: {ref}, Date: {date}, Subject: {title}.\nContent: '
        metadata = {
            'reference_number': ref,
            'date_issued': date,
            'subject': title,
            'issuing_authority': 'Manual Entry'
        }
        entry_id = data.get('entry_id')
        if entry_id:
            if not isinstance(entry_id, str):
                return jsonify({'error': 'entry_id must be text.'}), 400
            metadata['entry_id'] = entry_id
        docs = [Document(page_content=header + c, metadata=metadata) for c in chunks]
        with kb_lock:
            hybrid_retriever = build_hybrid_retriever(docs, replace_entry_id=entry_id)

        return jsonify({
            'status': 'ok',
            'title': title,
            'chunks_added': len(docs),
            'total_chunks': len(load_bm25_docs())
        })
    except Exception as e:
        log.error(f'❌ Add text failed: {e}')
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start ngrok tunnel if token is provided
    public_url = None
    if NGROK_SUPPORT and NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != 'YOUR_NGROK_AUTH_TOKEN':
        try:
            print('[WAIT] Starting Ngrok tunnel...', flush=True)
            conf.get_default().auth_token = NGROK_AUTH_TOKEN
            ngrok.kill()
            public_url = ngrok.connect(PORT).public_url
            log.info(f'🌐 Ngrok tunnel LIVE: {public_url}')
        except Exception as e:
            log.error(f'❌ Ngrok failed: {e}')
            print(f'❌ Ngrok failed: {e}. Check your NGROK_AUTH_TOKEN.', flush=True)
    else:
        print('[INFO] Ngrok token not provided. Server will only be accessible locally.', flush=True)
        print('[INFO] To enable public access, set NGROK_AUTH_TOKEN in your .env file.', flush=True)

    print()
    print('=' * 62, flush=True)
    print('  GPA RAG Backend - Production Server', flush=True)
    print('=' * 62, flush=True)
    print(f'  Local URL   : http://localhost:{PORT}', flush=True)
    if public_url:
        print(f'  Public URL  : {public_url}', flush=True)
    print(f'  Input docs  : {INPUT_DIR}', flush=True)
    print(f'  ChromaDB    : {CHROMA_DIR}', flush=True)
    print(f'  Total chunks: {len(load_bm25_docs())}', flush=True)
    print(f'  Total files : {len(load_processed_log())}', flush=True)
    print(flush=True)
    if public_url:
        print('  📋 NEXT STEP: Update your Cloudflare Worker Secret:', flush=True)
        print(f'     wrangler secret put RAG_BACKEND_URL', flush=True)
        print(f'     (Then paste: {public_url})', flush=True)
        print(flush=True)
    print('  Endpoints:', flush=True)
    print(f'    GET  /api/health', flush=True)
    print(f'    GET  /api/rag/stats', flush=True)
    print(f'    POST /api/rag/query', flush=True)
    print(f'    POST /api/rag/ingest', flush=True)
    print(f'    POST /api/rag/add-text', flush=True)
    print('=' * 62, flush=True)
    print(flush=True)

    app.run(host=HOST, port=PORT, debug=False)
