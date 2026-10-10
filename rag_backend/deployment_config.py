"""Hosting settings shared by the local server and Render."""
from pathlib import Path


def storage_base(environ, default):
    # Render's free filesystem is temporary. A paid disk or other mounted
    # volume can opt in via RAG_STORAGE_DIR without changing local paths.
    return Path(environ.get('RAG_STORAGE_DIR') or
                environ.get('GOOGLE_DRIVE_PATH') or default)


def server_port(environ):
    return int(environ.get('PORT') or environ.get('RAG_PORT') or 5000)


def allowed_origins(environ, defaults):
    configured = environ.get('RAG_ALLOWED_ORIGINS')
    return [value.strip() for value in configured.split(',') if value.strip()] if configured else defaults
