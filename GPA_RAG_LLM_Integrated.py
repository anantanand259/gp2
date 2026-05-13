# 1.1  Mount Google Drive
# Google Drive mount removed — set BASE_PATH below to your local folder
print('[OK] Running in local mode.')

# 1.2  Install all dependencies
# import subprocess, sys
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U',
#     'google-genai', 'langchain==0.2.0', 'langchain-community==0.2.0',
#     'langchain-core==0.2.0', 'langchain-text-splitters', 'langchain-huggingface',
#     'chromadb', 'rank_bm25', 'flask', 'flask-cors', 'pyngrok', 'pydantic'])

print('[OK] All dependencies installed.')

# 1.3  Configuration — EDIT THESE BEFORE RUNNING
import os, json, requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

GOOGLE_API_KEY   = os.getenv('GOOGLE_API_KEY', 'YOUR_GOOGLE_API_KEY_HERE')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'YOUR_OPENROUTER_KEY_HERE')
OPENROUTER_MODEL = 'meta-llama/llama-3.3-70b-instruct'
NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', 'YOUR_NGROK_AUTH_TOKEN')

# Paths (Local)
BASE_PATH      = os.path.join(os.path.dirname(__file__), 'rag_backend', 'knowledge_base')
INPUT_PATH     = os.path.join(BASE_PATH, 'input_docs')
PROCESSED_PATH = os.path.join(BASE_PATH, 'processed_docs')
CHROMA_PATH    = os.path.join(BASE_PATH, 'chroma_db')
LOG_PATH       = os.path.join(BASE_PATH, 'processed_log.json')
BM25_DOCS_PATH = os.path.join(BASE_PATH, 'bm25_docs.json')

for path in [INPUT_PATH, PROCESSED_PATH, CHROMA_PATH]:
    os.makedirs(path, exist_ok=True)

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
print('[OK] Configuration complete.')
print(f'   Input docs  -> {INPUT_PATH}')
print(f'   ChromaDB    -> {CHROMA_PATH}')

# 2.1  Initialise Gemini Client
from PIL import Image
from google import genai
from google.genai import types

client = genai.Client()
print('[OK] Gemini client ready.')

# 2.2  Pydantic schema for structured VLM extraction
from pydantic import BaseModel, Field
from typing import List, Optional

class SignatoryEntity(BaseModel):
    name_or_designation: str = Field(description="Official designation like 'प्राचार्य'")
    organization: Optional[str] = Field(description='Institution name if mentioned')
    date_signed:  Optional[str] = Field(description='Date near signature if visible')

class AcademicNoticeSchema(BaseModel):
    issuing_authority:  Optional[str]           = None
    reference_number:   Optional[str]           = None
    date_issued:        Optional[str]           = None
    subject_line:       Optional[str]           = None
    target_audience:    Optional[List[str]]     = None
    main_body_content:  Optional[str]           = None
    signatories:        Optional[List[SignatoryEntity]] = None
    distribution_list:  Optional[List[str]]     = None
    document_type:      Optional[str]           = None
    extra_fields:       Optional[str]           = None

print('[OK] Pydantic schema defined.')

# 2.3  VLM Extraction Module
EXTRACTION_PROMPT = '''
You are an intelligent document understanding system for academic/administrative documents.

STEP 1 — Classify document: notice | office_order | exam_schedule | email | tabular_data | unknown
STEP 2 — HIGH PRIORITY: extract reference_number, date_issued, issuing_authority accurately.
STEP 3 — main_body_content MUST contain ALL readable text. Never return null here.
STEP 4 — Preserve Hindi and English exactly. Tables in Markdown. No hallucination.
STEP 5 — extra_fields: JSON string for tables or unknown structures.
Return valid JSON only.
'''

def extract_structured_metadata(image_path: str) -> AcademicNoticeSchema:
    try:
        document_image = Image.open(image_path)
    except FileNotFoundError:
        raise ValueError(f'Image not found: {image_path}')

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

print('[OK] VLM extraction module ready.')

# 2.4  Chunking Module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_intelligent_context_chunks(structured_notice: AcademicNoticeSchema) -> List[Document]:
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=['\n\n', '\n', '।', '.', ' ', '']
    )

    return [
        Document(page_content=context_header + chunk, metadata=global_metadata)
        for chunk in text_splitter.split_text(body_text)
    ]

print('[OK] Chunking module ready.')

# 2.5  Multilingual Embedding Model (BAAI/bge-m3)
# NOTE: First load takes ~2-3 minutes to download the model weights.
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

print('[WAIT] Loading BAAI/bge-m3 embedding model (first run ~2-3 min)...')
embedding_model = initialize_embedding_model()
print('[OK] Embedding model loaded.')

# 2.6  Hybrid Retrieval System (ChromaDB + BM25)
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def load_processed_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            return json.load(f)
    return []

def save_processed_log(log):
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

def load_bm25_docs():
    if os.path.exists(BM25_DOCS_PATH):
        with open(BM25_DOCS_PATH, 'r') as f:
            return json.load(f)
    return []

def save_bm25_docs(docs):
    with open(BM25_DOCS_PATH, 'w') as f:
        json.dump(docs, f, indent=2)

def build_hybrid_retriever(new_documents, embeddings):
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    if new_documents:
        vector_store.add_documents(new_documents)
        vector_store.persist()

    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': 4})

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

print('[OK] Hybrid retrieval module ready.')

# 2.7  Run Document Ingestion
import shutil

processed_files = load_processed_log()
all_files       = os.listdir(INPUT_PATH)
new_files       = [
    f for f in all_files
    if f not in processed_files and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f'[SEARCH] Found {len(new_files)} new file(s): {new_files}')

documents_to_add = []

for i, filename in enumerate(new_files, 1):
    image_path = os.path.join(INPUT_PATH, filename)
    print(f'  [{i}/{len(new_files)}] Processing: {filename}')
    try:
        structured_notice = extract_structured_metadata(image_path)
        chunks = create_intelligent_context_chunks(structured_notice)
        documents_to_add.extend(chunks)
        shutil.move(image_path, os.path.join(PROCESSED_PATH, filename))
        processed_files.append(filename)
        print(f'     [OK] {len(chunks)} chunk(s) extracted')
    except Exception as e:
        print(f'     [ERROR] Failed: {e}')

save_processed_log(processed_files)

print('\n[WAIT] Building hybrid retriever...')
hybrid_retriever = build_hybrid_retriever(documents_to_add, embedding_model)
print(f'\n[OK] Done. New chunks: {len(documents_to_add)}, Total BM25: {len(load_bm25_docs())}')

# 3.1  RAG Query Function
GPA_SYSTEM_PROMPT = '''
You are "GPA Assistant" — an intelligent chatbot for
Government Polytechnic Adityapur (GPA), Jamshedpur, Jharkhand.

RULES:
- Answer ONLY using the provided CONTEXT when relevant.
- If context lacks the answer, say: "Information not available in knowledge base."
- Keep answers concise and factual. Preserve Hindi text as-is.
- Do NOT hallucinate dates, numbers, or names.
'''

def generate_rag_answer(user_query: str, retriever) -> dict:
    retrieved_docs = retriever.invoke(user_query)

    if not retrieved_docs:
        return {
            'answer': 'Information regarding this query is not available in our knowledge base.',
            'sources': [],
            'chunk_count': 0
        }

    context_blocks = []
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(f'[Source {i}]: {doc.page_content}')
        meta = doc.metadata
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
            "temperature": 0.3
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            answer_text = response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"OpenRouter error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ OpenRouter fallback to Gemini: {e}")
        prompt = f'''{GPA_SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nUSER QUERY: {user_query}\n\nAnswer:'''
        gemini_res = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        answer_text = gemini_res.text

    return {'answer': answer_text, 'sources': sources, 'chunk_count': len(retrieved_docs)}

print('[OK] RAG query function ready.')

# 3.2  Start Flask API Server with ngrok
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok, conf

def add_text_document(title, content, date='N/A', ref='MANUAL'):
    global hybrid_retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    chunks   = splitter.split_text(content)
    header   = f'Notice Reference: {ref}, Date: {date}, Subject: {title}.\nContent: '
    metadata = {'reference_number': ref, 'date_issued': date, 'subject': title, 'issuing_authority': 'Manual'}
    docs = [Document(page_content=header + c, metadata=metadata) for c in chunks]
    hybrid_retriever = build_hybrid_retriever(docs, embedding_model)
    processed = load_processed_log()
    if title not in processed:
        processed.append(title)
        save_processed_log(processed)
    print(f'✅ Added "{title}" with {len(docs)} chunk(s).')

app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': '*'}})

_retriever = hybrid_retriever

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'GPA RAG Backend', 'version': '2.0'})

@app.route('/api/rag/stats', methods=['GET'])
def stats():
    try:
        return jsonify({
            'total_chunks':    len(load_bm25_docs()),
            'total_documents': len(load_processed_log()),
            'status':          'ready'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    if _retriever is None:
        return jsonify({'error': 'RAG not initialised. Run Section 2 first.'}), 503
    data = request.get_json(silent=True)
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" field.'}), 400
    user_query = str(data['query']).strip()
    if not user_query:
        return jsonify({'error': 'Query cannot be empty.'}), 400
    try:
        return jsonify(generate_rag_answer(user_query, _retriever))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/ingest', methods=['POST'])
def trigger_ingest():
    global _retriever, hybrid_retriever
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                save_path = os.path.join(INPUT_PATH, file.filename)
                file.save(save_path)
                print(f'📥 Uploaded file saved: {file.filename}')
                
                # Process plain text/JSON/Markdown directly if uploaded
                if file.filename.lower().endswith(('.txt', '.md', '.json')):
                    with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    add_text_document(file.filename, content, ref='FILE_UPLOAD')
                    _retriever = hybrid_retriever
                elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Process image using Gemini VLM extractor
                    import shutil
                    structured_notice = extract_structured_metadata(save_path)
                    chunks = create_intelligent_context_chunks(structured_notice)
                    hybrid_retriever = build_hybrid_retriever(chunks, embedding_model)
                    _retriever = hybrid_retriever
                    shutil.move(save_path, os.path.join(PROCESSED_PATH, file.filename))
                    processed = load_processed_log()
                    processed.append(file.filename)
                    save_processed_log(processed)
                    print(f'✅ Image processed successfully into RAG database.')

        return jsonify({
            'status': 'ok',
            'new_chunks': 1,
            'total_chunks': len(load_bm25_docs()),
            'total_documents': len(load_processed_log())
        })
    except Exception as e:
        print(f'❌ Ingest error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/add-text', methods=['POST'])
def add_text_entry():
    global _retriever, hybrid_retriever
    data = request.get_json(silent=True)
    if not data or 'content' not in data:
        return jsonify({'error': 'Missing "content" field.'}), 400

    title   = data.get('title', 'Manual Entry')
    content = data['content']
    date_str = data.get('date', 'N/A')
    ref     = data.get('reference', 'MANUAL_ENTRY')

    try:
        add_text_document(title, content, date=date_str, ref=ref)
        _retriever = hybrid_retriever
        return jsonify({
            'status': 'ok',
            'title': title,
            'total_chunks': len(load_bm25_docs())
        })
    except Exception as e:
        print(f'❌ Add text error: {e}')
        return jsonify({'error': str(e)}), 500

# Start ngrok tunnel
conf.get_default().auth_token = NGROK_AUTH_TOKEN
ngrok.kill()
public_url = ngrok.connect(5000).public_url

print('=' * 60)
print('[LIVE]  GPA RAG Backend is LIVE!')
print('=' * 60)
print(f'   Public URL : {public_url}')
print()
print('📋  Paste into chatbot.js:')
print(f'    RAG_BACKEND_URL: \'{public_url}\',')
print()
print(f'   GET  {public_url}/api/health')
print(f'   GET  {public_url}/api/rag/stats')
print(f'   POST {public_url}/api/rag/query')
print('=' * 60)

threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True).start()
print('\n⚡ Server running. Keep this notebook open.')

# 4.1  Test a query directly
test_query = 'What is the information about E-Kalyan scholarship?'
result = generate_rag_answer(test_query, hybrid_retriever)
print(f'Query   : {test_query}')
print(f'Chunks  : {result["chunk_count"]}')
print(f'Sources : {result["sources"]}')
print()
print('─── Answer ───')
print(result['answer'])

# 4.2  Add a plain-text document manually (no image needed)
# Helper function add_text_document is initialized in section 3.2 above.
# Example (uncomment to use):
# add_text_document(
#     title='E-Kalyan Scholarship 2026',
#     content='Students from SC/ST/OBC must apply at ekalyan.cgg.gov.in before March 31, 2026.',
#     date='2026-02-01',
#     ref='EKalyan-2026'
# )
print('[OK] add_text_document() helper ready.')

# 4.3  Export / Import knowledge base snapshot
EXPORT_PATH = f'{BASE_PATH}/kb_export.json'

def export_knowledge_base():
    snapshot = {'bm25_docs': load_bm25_docs(), 'processed_files': load_processed_log()}
    with open(EXPORT_PATH, 'w') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    size_kb = os.path.getsize(EXPORT_PATH) / 1024
    print(f'[OK] Exported to {EXPORT_PATH}  ({size_kb:.1f} KB)')

def import_knowledge_base(path=EXPORT_PATH):
    global hybrid_retriever
    with open(path, 'r') as f:
        snapshot = json.load(f)
    save_bm25_docs(snapshot['bm25_docs'])
    save_processed_log(snapshot['processed_files'])
    hybrid_retriever = build_hybrid_retriever([], embedding_model)
    print(f'[OK] Imported from {path}')
    print(f'   Chunks: {len(snapshot["bm25_docs"])}, Files: {len(snapshot["processed_files"])}')

# Uncomment to use:
# export_knowledge_base()
# import_knowledge_base()
print('[OK] Export/Import helpers ready.')

# 4.4  Stop the server (run when done)
# ngrok.kill()
# print('[STOP] Server stopped.')


if __name__ == "__main__":
    print("[OK] GPA RAG Backend script loaded. Run cells in order.")
