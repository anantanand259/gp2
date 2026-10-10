import json
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag_backend'))
request_stub = ModuleType('requests')
request_stub.post = Mock()
genai_stub = ModuleType('google.genai')
genai_stub.types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
google_stub = ModuleType('google')
google_stub.genai = genai_stub
with patch.dict(sys.modules, {'requests': request_stub, 'google': google_stub, 'google.genai': genai_stub}):
    import llm_pool as pool_module
    from llm_pool import LLMPool, Route, ProvidersUnavailableError
from query_cache import QueryCache
from answer_policy import parse_kb_decision


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.time = 100
        self.routes = [Route('openrouter', 'first', 'test-key', 'https://test.invalid'),
                       Route('groq', 'second', 'test-key', 'https://test.invalid'), Route('gemini', 'third')]
        self.pool = LLMPool(Mock(), self.routes, clock=lambda: self.time)

    def test_quota_failure_falls_back_and_skips_primary_for_other_users(self):
        calls = []
        def invoke(route):
            calls.append(route.name)
            if route.name == 'openrouter':
                raise RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: '71305s'")
            return 'answer'
        self.assertEqual(self.pool.run(invoke), 'answer')
        self.assertEqual(calls, ['openrouter', 'groq'])
        calls.clear()
        self.pool.run(invoke)
        self.assertEqual(calls, ['groq'])
        self.assertEqual(self.pool.status()['cooldowns']['openrouter:first'], 71305)

    def test_two_failed_providers_use_third(self):
        def invoke(route):
            if route.name != 'gemini':
                raise RuntimeError('429 quota exceeded')
            return 'third works'
        self.assertEqual(self.pool.run(invoke), 'third works')
        self.assertEqual(self.pool.last_success, 'gemini:third')

    def test_same_context_is_sent_to_fallback_and_invalid_json_rejected(self):
        seen = []
        def invoke(route):
            seen.append('same KB context')
            if route.name == 'openrouter':
                return '{invalid json'
            return json.dumps({'supported': True, 'answer': 'Known deadline', 'source_indices': [1]})
        decision = self.pool.run(invoke, lambda raw: parse_kb_decision(raw, 1))
        self.assertEqual(decision['answer'], 'Known deadline')
        self.assertEqual(seen, ['same KB context', 'same KB context'])

    def test_real_adapters_send_identical_context_after_quota_failure(self):
        rejected = SimpleNamespace(status_code=429, text='quota exceeded', headers={'Retry-After': '120'})
        accepted = SimpleNamespace(status_code=200, json=lambda: {'choices': [{'message': {'content': json.dumps({
            'supported': True, 'answer': 'Known deadline', 'source_indices': [1]
        })}}]})
        with patch.object(pool_module.requests, 'post', side_effect=[rejected, accepted]) as post:
            decision = self.pool.generate_json('Only use the KB. Return JSON.', 'Same source and question', lambda raw: parse_kb_decision(raw, 1))
        self.assertEqual(decision['answer'], 'Known deadline')
        self.assertEqual(post.call_args_list[0].kwargs['json']['messages'], post.call_args_list[1].kwargs['json']['messages'])
        self.assertEqual(self.pool.status()['cooldowns']['openrouter:first'], 120)

    def test_unavailable_all_models_returns_retryable_error(self):
        with self.assertRaises(ProvidersUnavailableError) as caught:
            self.pool.run(lambda _: (_ for _ in ()).throw(RuntimeError('429 quota exceeded')))
        self.assertEqual(caught.exception.retry_after, 60)

    def test_cooldown_expires(self):
        self.pool.blocked_until['openrouter:first'] = 120
        self.assertEqual(self.pool.run(lambda r: r.name), 'groq')
        self.time = 121
        self.assertEqual(self.pool.run(lambda r: r.name), 'openrouter')

    def test_busy_primary_uses_another_route(self):
        pool = LLMPool(Mock(), self.routes, max_concurrency=1)
        pool.slots['openrouter'].acquire()
        try:
            self.assertEqual(pool.run(lambda r: r.name), 'groq')
        finally:
            pool.slots['openrouter'].release()

    def test_slots_are_released_after_failed_calls(self):
        self.pool.run(lambda r: (_ for _ in ()).throw(RuntimeError('timeout')) if r.name == 'openrouter' else 'ok')
        self.assertTrue(self.pool.slots['openrouter'].acquire(blocking=False))

    def test_missing_keys_are_skipped(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'YOUR_GROQ_KEY_HERE'}, clear=True):
            pool = LLMPool.from_environment(Mock(), 'YOUR_OPENROUTER_KEY_HERE')
        self.assertEqual([r.name for r in pool.routes], ['gemini', 'gemini'])

    def test_gemini_extraction_and_search_fall_back_to_second_model(self):
        client = Mock()
        client.models.generate_content.side_effect = [RuntimeError('429 quota exceeded'), 'vision result']
        pool = LLMPool(client, [Route('gemini', 'flash'), Route('gemini', 'lite')])
        self.assertEqual(pool.generate_gemini('image or search context', {'same': 'rules'}), 'vision result')
        self.assertEqual([c.kwargs['model'] for c in client.models.generate_content.call_args_list], ['flash', 'lite'])


class CacheTests(unittest.TestCase):
    def test_repeated_questions_reuse_result_until_expiry(self):
        clock = [0]
        cache = QueryCache(ttl=5, clock=lambda: clock[0])
        generate = Mock(return_value={'answer': 'Known answer', 'source_type': 'rag'})
        cache.run(('kb-version', 'question'), generate)
        cache.run(('kb-version', 'question'), generate)
        self.assertEqual(generate.call_count, 1)
        clock[0] = 6
        cache.run(('kb-version', 'question'), generate)
        self.assertEqual(generate.call_count, 2)

    def test_kb_change_invalidates_cache(self):
        cache = QueryCache()
        generate = Mock(return_value={'answer': 'Known answer', 'source_type': 'rag'})
        cache.run(('version-1', 'question'), generate)
        cache.run(('version-2', 'question'), generate)
        self.assertEqual(generate.call_count, 2)

    def test_concurrent_same_question_uses_one_model_call(self):
        cache = QueryCache()
        started, release = threading.Event(), threading.Event()
        def generate():
            started.set()
            self.assertTrue(release.wait(2))
            return {'answer': 'Known answer', 'source_type': 'rag'}
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(cache.run, 'same-query', generate)
            self.assertTrue(started.wait(2))
            other_generate = Mock(side_effect=AssertionError('Duplicate model call'))
            second = executor.submit(cache.run, 'same-query', other_generate)
            release.set()
            self.assertEqual(first.result()['answer'], second.result()['answer'])
            other_generate.assert_not_called()

    def test_errors_and_unverified_answers_are_not_cached(self):
        cache = QueryCache()
        generate = Mock(side_effect=[ValueError('provider error'), {'source_type': 'none'}, {'source_type': 'rag'}])
        with self.assertRaises(ValueError):
            cache.run('question', generate)
        cache.run('question', generate)
        cache.run('question', generate)
        self.assertEqual(generate.call_count, 3)


if __name__ == '__main__':
    unittest.main()
