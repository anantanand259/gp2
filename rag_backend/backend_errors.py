"""Safe, actionable errors for clients; never return raw provider payloads."""
import math
import re


def describe_query_error(error, operation='query'):
    message = str(error).lower()
    if any(marker in message for marker in ('resource_exhausted', 'quota', 'rate limit')) or re.search(r'\b429\b', message):
        payload = {'code': 'PROVIDER_QUOTA_ERROR',
                   'error': 'The AI provider quota or rate limit has been reached.'}
        retry_match = re.search(r'retrydelay[\"\x27]?\s*:\s*[\"\x27]?(\d+(?:\.\d+)?)s', message)
        if retry_match:
            seconds = math.ceil(float(retry_match.group(1)))
            payload['retry_after_seconds'] = seconds
            minutes = math.ceil(seconds / 60)
            hours, minutes = divmod(minutes, 60)
            payload['error'] += f' The provider asks you to retry in about {hours}h {minutes}m.'
        else:
            payload['error'] += ' Wait for the provider quota to reset or check its billing and usage limits.'
        if operation == 'upload':
            payload['error'] += ' This upload was not indexed. Plain-text PDFs, JSON, TXT, MD and CSV can be uploaded without Gemini visual extraction.'
        return payload, 429
    if any(marker in message for marker in (
        'api_key_invalid', 'api key not valid', 'invalid api key',
        'invalid_api_key', 'unauthorized', 'authentication',
    )) or re.search(r'\b401\b', message):
        return {'code': 'PROVIDER_AUTH_ERROR',
                'error': 'The knowledge base is reachable, but its AI provider API key is missing or invalid. The administrator must update the backend API key and restart the Python server.'}, 503
    if any(marker in message for marker in ('timeout', 'timed out')):
        return {'code': 'PROVIDER_TIMEOUT',
                'error': 'The knowledge base was reached, but the AI provider took too long to respond. Please retry.'}, 504
    return {'code': 'RAG_QUERY_ERROR',
            'error': 'The knowledge-base query failed. Please ask the administrator to check the Python server logs.'}, 500
