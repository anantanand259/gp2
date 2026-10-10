"""Shared provider failover, quota cooldowns, and bounded concurrency."""
import logging
import math
import os
import threading
import time
from dataclasses import dataclass

import requests
from google.genai import types
from backend_errors import describe_query_error

log = logging.getLogger('GPA-RAG')


class ProvidersUnavailableError(RuntimeError):
    code = 'LLM_PROVIDERS_UNAVAILABLE'
    def __init__(self, retry_after=60):
        super().__init__('All configured AI models are temporarily unavailable.')
        self.retry_after = max(1, math.ceil(retry_after))


@dataclass
class Route:
    name: str
    model: str
    key: str = ''
    url: str = ''


def configured_key(value):
    return bool(value and not value.startswith('YOUR_'))


class LLMPool:
    def __init__(self, client, routes, max_concurrency=2, clock=time.monotonic):
        self.client = client
        self.routes = routes
        self.clock = clock
        self.lock = threading.Lock()
        self.blocked_until = {}
        self.last_success = None
        # Gemini models share a project: bound concurrency across both models.
        self.slots = {r.name: threading.BoundedSemaphore(max_concurrency) for r in routes}

    @classmethod
    def from_environment(cls, client, openrouter_key='', openrouter_model='meta-llama/llama-3.3-70b-instruct'):
        routes = []
        providers = os.getenv('LLM_PROVIDER_ORDER', 'openrouter,groq,gemini').split(',')
        for provider in dict.fromkeys(p.strip() for p in providers):
            if provider == 'openrouter' and configured_key(openrouter_key):
                routes.append(Route('openrouter', os.getenv('OPENROUTER_MODEL', openrouter_model), openrouter_key,
                                    'https://openrouter.ai/api/v1/chat/completions'))
            elif provider == 'groq' and configured_key(os.getenv('GROQ_API_KEY', '')):
                routes.append(Route('groq', os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'), os.environ['GROQ_API_KEY'],
                                    'https://api.groq.com/openai/v1/chat/completions'))
            elif provider == 'gemini' and client is not None:
                for model in dict.fromkeys(m.strip() for m in os.getenv('GEMINI_MODELS', 'gemini-2.5-flash,gemini-2.5-flash-lite').split(',') if m.strip()):
                    routes.append(Route('gemini', model))
        return cls(client, routes, max_concurrency=max(1, int(os.getenv('LLM_MAX_CONCURRENCY', '2'))))

    def _identity(self, route):
        return route.name + ':' + route.model

    def _failure(self, route, error):
        payload, status = describe_query_error(error)
        delay = payload.get('retry_after_seconds', 60 if status == 429 else 15)
        response = getattr(error, 'response', None)
        if response is not None:
            try:
                delay = max(delay, float(response.headers.get('Retry-After', 0)))
            except (ValueError, TypeError):
                pass
        if payload['code'] == 'PROVIDER_AUTH_ERROR':
            delay = 300
        identity = self._identity(route)
        with self.lock:
            self.blocked_until[identity] = self.clock() + delay
        # Log route and exception class, never keys, context, or raw payloads.
        log.warning('AI route %s failed (%s); cooling down for %ss', identity, type(error).__name__, math.ceil(delay))

    def run(self, invoke, validator=lambda value: value, gemini_only=False):
        routes = [r for r in self.routes if not gemini_only or r.name == 'gemini']
        errors = []
        for route in routes:
            identity = self._identity(route)
            with self.lock:
                blocked = self.blocked_until.get(identity, 0) > self.clock()
            if blocked or not self.slots[route.name].acquire(blocking=False):
                continue
            try:
                # Recheck after acquiring the slot in case another request hit quota.
                with self.lock:
                    blocked = self.blocked_until.get(identity, 0) > self.clock()
                if blocked:
                    continue
                value = validator(invoke(route))
                with self.lock:
                    self.last_success = identity
                return value
            except Exception as error:
                errors.append(error)
                self._failure(route, error)
            finally:
                self.slots[route.name].release()
        if errors and all(isinstance(error, ValueError) for error in errors):
            raise errors[-1]
        with self.lock:
            delays = [max(0, self.blocked_until.get(self._identity(r), 0) - self.clock()) for r in routes]
        # A route without a cooldown may merely be busy; allow a quick retry.
        raise ProvidersUnavailableError(min(delays) if delays and all(delays) else 5)

    def generate_json(self, system, question, validator):
        def invoke(route):
            if route.name == 'gemini':
                response = self.client.models.generate_content(
                    model=route.model, contents=question,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=0,
                                                       response_mime_type='application/json'))
                return response.text
            response = requests.post(route.url,
                                     headers={'Authorization': 'Bearer ' + route.key, 'Content-Type': 'application/json'},
                                     json={'model': route.model, 'messages': [
                                         {'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
                                           'response_format': {'type': 'json_object'}, 'temperature': 0, 'max_tokens': 2048},
                                     timeout=20)
            if response.status_code >= 400:
                # Include a redacted body internally to extract quota RetryInfo.
                detail = response.text.replace(route.key, '[redacted]')
                failure = RuntimeError(f'HTTP {response.status_code}: {detail}')
                failure.response = response
                raise failure
            return response.json()['choices'][0]['message']['content']
        return self.run(invoke, validator)

    def generate_gemini(self, contents, config, validator=lambda value: value):
        return self.run(lambda route: self.client.models.generate_content(model=route.model, contents=contents, config=config),
                        validator=validator, gemini_only=True)

    def status(self):
        with self.lock:
            return {'configured_models': [self._identity(r) for r in self.routes],
                    'last_success': self.last_success,
                    'cooldowns': {name: math.ceil(until - self.clock()) for name, until in self.blocked_until.items() if until > self.clock()}}
