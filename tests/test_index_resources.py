"""Regressions for the real indexing code, isolated from private KB startup."""
import ast
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def load_function(name, namespace, class_name=None):
    tree = ast.parse((ROOT / 'rag_backend/server.py').read_text(encoding='utf-8'))
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<server function>', 'exec'), namespace)
    return namespace[name]


class ResourceTests(unittest.TestCase):
    def test_embedding_work_is_partitioned_in_order(self):
        fn = load_function('embed_documents', {'List': list}, 'OnnxMiniLMEmbeddings')
        calls = []
        def batch(texts):
            calls.append(texts)
            return [[text] for text in texts]
        instance = SimpleNamespace(batch_size=2, embed_documents=batch)
        result = fn(instance, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(calls, [['a', 'b'], ['c', 'd'], ['e']])
        self.assertEqual(result, [['a'], ['b'], ['c'], ['d'], ['e']])
        self.assertEqual(fn(instance, []), [])

    def test_rebuild_embeds_only_missing_vectors(self):
        stored = {'content': 'Existing notice', 'metadata': {'filename': 'old.txt'}}
        old_id = hashlib.sha256(json.dumps(stored, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        store = Mock()
        store.get.return_value = {'ids': [old_id]}
        lexical = Mock()
        namespace = {'RETRIEVAL_MODE': 'hybrid', 'Chroma': Mock(return_value=store), 'CHROMA_DIR': Path('/test'),
                     'embedding_model': Mock(), 'load_bm25_docs': lambda: [stored],
                     'save_bm25_docs': Mock(), 'Document': lambda **kw: SimpleNamespace(**kw),
                     'hashlib': hashlib, 'json': json, 'log': Mock(),
                     'BM25Retriever': SimpleNamespace(from_documents=Mock(return_value=lexical)),
                     'EnsembleRetriever': Mock()}
        rebuild = load_function('build_hybrid_retriever', namespace)
        rebuild([])
        store.add_documents.assert_not_called()
        rebuild([SimpleNamespace(page_content='New notice', metadata={'filename': 'new.txt'})])
        added = store.add_documents.call_args.args[0]
        self.assertEqual([doc.page_content for doc in added], ['New notice'])
        self.assertEqual(len(store.add_documents.call_args.kwargs['ids']), 1)

    def test_low_memory_profile_indexes_without_vectors(self):
        saved = Mock()
        lexical = Mock()
        vector = Mock(side_effect=AssertionError('No vector model on free profile'))
        namespace = {'RETRIEVAL_MODE': 'bm25', 'Chroma': vector,
                     'load_bm25_docs': lambda: [], 'save_bm25_docs': saved,
                     'Document': lambda **kw: SimpleNamespace(**kw), 'hashlib': hashlib,
                     'json': json, 'log': Mock(),
                     'EmptyRetriever': lambda: SimpleNamespace(invoke=lambda q: []),
                     'BM25Retriever': SimpleNamespace(from_documents=Mock(return_value=lexical))}
        rebuild = load_function('build_hybrid_retriever', namespace)
        self.assertEqual(rebuild([]).invoke('query'), [])
        result = rebuild([SimpleNamespace(page_content='Notice deadline Thursday', metadata={'filename': 'notice.txt'})])
        self.assertIs(result, lexical)
        self.assertEqual(result.k, 12)
        self.assertEqual(saved.call_args.args[0][0]['metadata']['filename'], 'notice.txt')
        vector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
