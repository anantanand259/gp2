"""Offline regressions: no model calls, downloads, or private KB access."""
import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag_backend'))
from answer_policy import parse_kb_decision
from backend_errors import describe_query_error

# Load the actual routing module with provider SDK stand-ins.
requests = ModuleType('requests')
requests.RequestException = type('RequestException', (Exception,), {})
requests.post = Mock()
types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw,
                        Tool=lambda **kw: kw, GoogleSearch=lambda: {})
genai = ModuleType('google.genai')
genai.types = types
google = ModuleType('google')
google.genai = genai
with patch.dict(sys.modules, {'requests': requests, 'google': google, 'google.genai': genai}):
    import rag_answers


class RoutingTests(unittest.TestCase):
    def test_authentication_error_is_actionable_without_raw_payload(self):
        payload, status = describe_query_error(ValueError('400 INVALID_ARGUMENT: API_KEY_INVALID sensitive provider payload'))
        self.assertEqual(status, 503)
        self.assertEqual(payload['code'], 'PROVIDER_AUTH_ERROR')
        self.assertNotIn('sensitive', payload['error'])

    def test_provider_quota_and_timeout_have_distinct_errors(self):
        self.assertEqual(describe_query_error(Exception('429 RESOURCE_EXHAUSTED'))[0]['code'], 'PROVIDER_QUOTA_ERROR')
        self.assertEqual(describe_query_error(Exception('Request timed out'))[1], 504)

    def setUp(self):
        self.doc = SimpleNamespace(page_content='Exam registration closes 14 October 2026.',
                                   metadata={'subject': 'Exam notice', 'filename': 'exam.pdf'})
        self.retriever = Mock()
        self.retriever.invoke.return_value = [self.doc]
        self.client = Mock()

    def test_supported_notice_never_searches(self):
        with patch.object(rag_answers, 'assess_context', return_value={
            'answer': 'Registration closes 14 October 2026 [Source 1].', 'source_indices': [1]
        }), patch.object(rag_answers, 'search_web') as web:
            result = rag_answers.answer_query('What is the deadline?', self.retriever, self.client, 'test', 'test')
            web.assert_not_called()
            self.assertEqual(result['source_type'], 'rag')
            self.assertEqual(result['sources'][0]['filename'], 'exam.pdf')

    def test_explicit_miss_searches(self):
        with patch.object(rag_answers, 'assess_context', return_value=None), patch.object(rag_answers, 'search_web', return_value={'source_type': 'internet'}) as web:
            rag_answers.answer_query('Weather?', self.retriever, self.client, 'test', 'test')
            web.assert_called_once_with('Weather?', self.client)

    def test_empty_kb_searches(self):
        self.retriever.invoke.return_value = []
        with patch.object(rag_answers, 'assess_context') as assess, patch.object(rag_answers, 'search_web') as web:
            rag_answers.answer_query('Weather?', self.retriever, self.client, 'test', 'test')
            assess.assert_not_called()
            web.assert_called_once()

    def test_retrieval_miss_checks_remaining_kb_before_web(self):
        missed = SimpleNamespace(page_content='परीक्षा पंजीकरण की अंतिम तिथि 14 अक्टूबर है।',
                                 metadata={'filename': 'hindi-notice.pdf'})
        with patch.object(rag_answers, 'assess_context', side_effect=[None, {
            'answer': '14 अक्टूबर [Source 1]', 'source_indices': [1]
        }]) as assess, patch.object(rag_answers, 'search_web') as web:
            result = rag_answers.answer_query('Deadline?', self.retriever, self.client, 'test', 'test', all_documents=[self.doc, missed])
            self.assertEqual(assess.call_count, 2)
            self.assertEqual(result['sources'][0]['filename'], 'hindi-notice.pdf')
            web.assert_not_called()

    def test_retrieval_failure_never_searches(self):
        self.retriever.invoke.side_effect = RuntimeError('Index unavailable')
        with patch.object(rag_answers, 'search_web') as web, self.assertRaises(RuntimeError):
            rag_answers.answer_query('Deadline?', self.retriever, self.client, 'test', 'test')
        web.assert_not_called()

    def test_invalid_decision_never_searches(self):
        with patch.object(rag_answers, 'assess_context', side_effect=ValueError('Bad JSON')), patch.object(rag_answers, 'search_web') as web, self.assertRaises(ValueError):
            rag_answers.answer_query('Deadline?', self.retriever, self.client, 'test', 'test')
        web.assert_not_called()

    def test_provider_fallback_keeps_context(self):
        requests.post.side_effect = requests.RequestException('Offline')
        self.client.models.generate_content.return_value.text = json.dumps({'supported': True, 'answer': '14 October [Source 1]', 'source_indices': [1]})
        decision = rag_answers.assess_context('Deadline?', [self.doc], self.client, 'test', 'test')
        self.assertEqual(decision['source_indices'], [1])
        kwargs = self.client.models.generate_content.call_args.kwargs
        self.assertIn('14 October 2026', kwargs['contents'])
        self.assertIn('exclusively', kwargs['config']['system_instruction'])
        requests.post.side_effect = None

    def test_uncited_web_output_is_not_an_internet_answer(self):
        self.client.models.generate_content.return_value = SimpleNamespace(text='Invented deadline', candidates=[])
        result = rag_answers.search_web('Deadline?', self.client)
        self.assertEqual(result['source_type'], 'none')
        self.assertNotIn('Invented deadline', result['answer'])

    def test_grounded_web_sources_are_returned(self):
        grounding = SimpleNamespace(grounding_chunks=[SimpleNamespace(web=SimpleNamespace(title='Official', uri='https://example.edu/notice'))],
                                    grounding_supports=[SimpleNamespace(grounding_chunk_indices=[0])])
        self.client.models.generate_content.return_value = SimpleNamespace(text='Verified detail', candidates=[SimpleNamespace(grounding_metadata=grounding)])
        result = rag_answers.search_web('Question?', self.client)
        self.assertEqual(result['source_type'], 'internet')
        self.assertIn('https://example.edu/notice', result['answer'])

    def test_invalid_grounding_rejected(self):
        for decision in [ {'supported': 'false'}, {'supported': True, 'answer': 'hi', 'source_indices': []},
                          {'supported': True, 'answer': 'hi', 'source_indices': [2]},
                          {'supported': True, 'answer': 'hi', 'source_indices': [True]} ]:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                parse_kb_decision(json.dumps(decision), 1)

    def test_index_failure_does_not_mark_file_processed(self):
        # Execute the real ingestion function in isolation from server startup.
        tree = ast.parse((ROOT / 'rag_backend/server.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ingest_documents')
        namespace = {'load_processed_log': lambda: [], 'save_processed_log': Mock(),
                     'os': SimpleNamespace(listdir=lambda _: ['exam.txt']),
                     'INPUT_DIR': Path('/input'), 'PROCESSED_DIR': Path('/processed'),
                     'get_file_type': lambda _: 'text', 'log': Mock(),
                     'extract_from_text': lambda _: object(),
                     'create_chunks': lambda _: [SimpleNamespace(metadata={})],
                     'build_hybrid_retriever': Mock(side_effect=RuntimeError('Index failure')),
                     'shutil': SimpleNamespace(move=Mock()), 'traceback': SimpleNamespace(format_exc=lambda: '')}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<ingestion>', 'exec'), namespace)
        with self.assertRaises(RuntimeError):
            namespace['ingest_documents'](fail_on_error=True)
        namespace['shutil'].move.assert_not_called()
        namespace['save_processed_log'].assert_not_called()

    def test_selected_upload_ignores_unreadable_pending_image(self):
        tree = ast.parse((ROOT / 'rag_backend/server.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ingest_documents')
        namespace = {'load_processed_log': lambda: [], 'save_processed_log': Mock(),
                     'os': SimpleNamespace(listdir=Mock(return_value=['unreadable.jpeg', 'new-notice.txt'])),
                     'INPUT_DIR': Path('/input'), 'PROCESSED_DIR': Path('/processed'),
                     'get_file_type': lambda name: 'image' if name.endswith('.jpeg') else 'text',
                     'log': Mock(), 'extract_from_text': Mock(return_value=object()),
                     'extract_from_image': Mock(side_effect=ValueError('No readable content')),
                     'create_chunks': lambda _: [SimpleNamespace(metadata={})],
                     'build_hybrid_retriever': Mock(), 'shutil': SimpleNamespace(move=Mock()),
                     'traceback': SimpleNamespace(format_exc=lambda: '')}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<ingestion>', 'exec'), namespace)
        docs = namespace['ingest_documents'](fail_on_error=True, filenames=['new-notice.txt'])
        self.assertEqual(len(docs), 1)
        namespace['extract_from_image'].assert_not_called()
        namespace['os'].listdir.assert_not_called()
        namespace['save_processed_log'].assert_called_once_with(['new-notice.txt'])
        self.assertEqual(docs[0].metadata['filename'], 'new-notice.txt')


if __name__ == '__main__':
    unittest.main()
