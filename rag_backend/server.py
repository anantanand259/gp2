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
from pathlib import Path
from typing import List, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# ┌──────────────────────────────────────────────────────────┐
# │  🔑  Place your Gemini API Key here or set env var       │
# └──────────────────────────────────────────────────────────┘
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyB3VfdenK3ejp9EmEJdQLx7lw8NLkMQ590')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_MODEL = 'meta-llama/llama-3.3-70b-instruct'

# Paths — relative to this script
BASE_DIR       = Path(__file__).parent
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
# CREATE DIRECTORIES
# ─────────────────────────────────────────────────────────────
for d in [INPUT_DIR, PROCESSED_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SET API KEY
# ─────────────────────────────────────────────────────────────
if GOOGLE_API_KEY == 'YOUR_GOOGLE_API_KEY_HERE':
    log.warning('⚠️  GOOGLE_API_KEY not set! Set it via environment variable or edit server.py line 35.')
    log.warning('   Example: set GOOGLE_API_KEY=AIzaSy...')

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# ─────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

client = genai.Client()
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
    if not full_text:
        raise ValueError(f'No text extracted from PDF: {pdf_path}')

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
        'issuing_authority': structured_notice.issuing_authority,
        'reference_number':  structured_notice.reference_number or 'UNKNOWN_REF',
        'date_issued':       structured_notice.date_issued,
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
from langchain_huggingface import HuggingFaceEmbeddings

log.info('⏳ Loading embedding model (first run downloads ~80MB)...')
embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
log.info('✅ Embedding model loaded.')

# ─────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL SYSTEM (ChromaDB + BM25)
# ─────────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def load_processed_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_processed_log(log_data: list):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f)

def load_bm25_docs() -> list:
    if BM25_FILE.exists():
        with open(BM25_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_bm25_docs(docs: list):
    with open(BM25_FILE, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

def build_hybrid_retriever(new_documents: list):
    """Build or rebuild the hybrid retriever with optional new documents."""
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model
    )

    if new_documents:
        vector_store.add_documents(new_documents)

    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    # Merge BM25 docs
    existing_docs = load_bm25_docs()
    new_doc_dicts = [{'content': d.page_content, 'metadata': d.metadata} for d in new_documents]
    all_docs = existing_docs + new_doc_dicts
    save_bm25_docs(all_docs)

    retrievers = [semantic_retriever]
    weights    = [1.0]

    if all_docs:
        bm25_retriever   = BM25Retriever.from_texts([d['content'] for d in all_docs])
        bm25_retriever.k = 4
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

def ingest_documents():
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

            for notice in notices:
                chunks = create_chunks(notice)
                documents_to_add.extend(chunks)

            # Move to processed
            shutil.move(filepath, str(PROCESSED_DIR / filename))
            processed_files.append(filename)
            log.info(f'     ✅ Extracted {sum(len(create_chunks(n)) for n in notices)} chunk(s)')

        except Exception as e:
            log.error(f'     ❌ Failed: {e}')
            log.debug(traceback.format_exc())

    save_processed_log(processed_files)
    return documents_to_add

# Run initial ingestion
log.info('⏳ Running initial document ingestion...')
initial_docs = ingest_documents()
hybrid_retriever = build_hybrid_retriever(initial_docs)
log.info(f'✅ Retriever ready. New chunks: {len(initial_docs)}, Total BM25: {len(load_bm25_docs())}')

# ─────────────────────────────────────────────────────────────
# RAG QUERY FUNCTION
# ─────────────────────────────────────────────────────────────
GPA_SYSTEM_PROMPT = '''
You are "GPA Assistant" — an intelligent chatbot for
Government Polytechnic Adityapur (GPA), Jamshedpur, Jharkhand.

- Answer ONLY using the provided CONTEXT. Do not use outside knowledge or internet surfing.
- If the context lacks the answer, politely say: "This specific information is not available in our knowledge base. Please contact the college directly or visit gpa.ac.in."
- Keep answers concise, factual, and extremely professional. Use well-formatted markdown.
- Preserve Hindi text as-is. Do NOT hallucinate dates, numbers, or names.
- Use bullet points and bold text for readability.
- IMPORTANT: Under no circumstances should you answer general questions (math, science, general knowledge) that are not in the context. Always politely decline them.
'''

def generate_rag_answer(user_query: str) -> dict:
    """Generate answer using RAG pipeline: retrieve → augment → generate."""
    global hybrid_retriever

    retrieved_docs = hybrid_retriever.invoke(user_query)

    if not retrieved_docs:
        return {
            'answer': 'This specific information is not available in our knowledge base. Please contact the college directly or visit [gpa.ac.in](https://www.gpa.ac.in).',
            'sources': [],
            'chunk_count': 0,
            'source_type': 'none'
        }

    context_blocks = []
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(f'[Source {i}]: {doc.page_content}')
        meta = doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {}
        sources.append({
            'index':     i,
            'reference': meta.get('reference_number', 'N/A'),
            'date':      meta.get('date_issued', 'N/A'),
            'subject':   meta.get('subject', 'N/A'),
            'authority': meta.get('issuing_authority', 'N/A')
        })

    context = '\n\n---\n\n'.join(context_blocks)
    
    # ── Generate Answer via OpenRouter (Llama 3.3 70B) ──
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://gpa.ac.in",
            "X-Title": "GPA Assistant RAG",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": GPA_SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nUSER QUERY: {user_query}\n\nAnswer:"}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            answer_text = response.json()['choices'][0]['message']['content']
        else:
            log.error(f"OpenRouter Error: {response.status_code} - {response.text}")
            raise Exception(f"OpenRouter API failed: {response.status_code}")

    except Exception as e:
        log.warning(f"⚠️ OpenRouter failed, falling back to Gemini: {str(e)}")
        # Fallback to Gemini
        prompt = f'''{GPA_SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nUSER QUERY: {user_query}\n\nAnswer:'''
        gemini_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        answer_text = gemini_response.text

    return {
        'answer': answer_text,
        'sources': sources,
        'chunk_count': len(retrieved_docs),
        'source_type': 'rag'
    }

log.info('✅ RAG query function ready.')

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
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
        return jsonify({'error': str(e)}), 500


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
                save_path = INPUT_DIR / file.filename
                file.save(str(save_path))
                log.info(f'📥 Uploaded new file to input_docs: {file.filename}')

        # Run ingestion on the directory
        new_docs = ingest_documents()
        hybrid_retriever = build_hybrid_retriever(new_docs)
        
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
        docs = [Document(page_content=header + c, metadata=metadata) for c in chunks]
        hybrid_retriever = build_hybrid_retriever(docs)

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
    print()
    print('=' * 62)
    print('  GPA RAG Backend - Production Server')
    print('=' * 62)
    print(f'  URL         : http://localhost:{PORT}')
    print(f'  Input docs  : {INPUT_DIR}')
    print(f'  ChromaDB    : {CHROMA_DIR}')
    print(f'  Total chunks: {len(load_bm25_docs())}')
    print(f'  Total files : {len(load_processed_log())}')
    print()
    print('  Endpoints:')
    print(f'    GET  http://localhost:{PORT}/api/health')
    print(f'    GET  http://localhost:{PORT}/api/rag/stats')
    print(f'    POST http://localhost:{PORT}/api/rag/query')
    print(f'    POST http://localhost:{PORT}/api/rag/ingest')
    print(f'    POST http://localhost:{PORT}/api/rag/add-text')
    print('=' * 62)
    print()

    app.run(host=HOST, port=PORT, debug=False)
