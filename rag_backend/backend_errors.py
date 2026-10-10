"""Safe, actionable errors for clients; never return raw provider payloads."""


def describe_query_error(error):
    message = str(error).lower()
    if any(marker in message for marker in (
        'api_key_invalid', 'api key not valid', 'invalid api key',
        'invalid_api_key', 'unauthorized', '401', 'authentication',
    )):
        return {'code': 'PROVIDER_AUTH_ERROR',
                'error': 'The knowledge base is reachable, but its AI provider API key is missing or invalid. The administrator must update the backend API key and restart the Python server.'}, 503
    if any(marker in message for marker in ('429', 'resource_exhausted', 'quota', 'rate limit')):
        return {'code': 'PROVIDER_QUOTA_ERROR',
                'error': 'The AI provider quota or rate limit has been reached. Please try again later or ask the administrator to check the provider account.'}, 503
    if any(marker in message for marker in ('timeout', 'timed out')):
        return {'code': 'PROVIDER_TIMEOUT',
                'error': 'The knowledge base was reached, but the AI provider took too long to respond. Please retry.'}, 504
    return {'code': 'RAG_QUERY_ERROR',
            'error': 'The knowledge-base query failed. Please ask the administrator to check the Python server logs.'}, 500
