import os

bind = '0.0.0.0:' + (os.getenv('PORT') or os.getenv('RAG_PORT') or '5000')
# Locks, provider cooldowns and the embedded Chroma store are process-local.
# Multiple workers would independently modify the same JSON/index files.
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 180
graceful_timeout = 30
accesslog = '-'
errorlog = '-'
