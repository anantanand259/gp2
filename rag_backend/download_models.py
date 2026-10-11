"""Cache the two embedding assets at build time, without loading server/KB."""
from huggingface_hub import hf_hub_download
import os
from deployment_config import embedding_filename

if __name__ == '__main__':
    hf_hub_download('sentence-transformers/all-MiniLM-L6-v2',
                    subfolder='onnx', filename=embedding_filename(os.environ))
    hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', filename='tokenizer.json')
