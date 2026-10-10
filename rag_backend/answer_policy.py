"""Validate the KB decision before allowing an external search."""
import json


def parse_kb_decision(raw, source_count):
    decision = json.loads(raw)
    if not isinstance(decision, dict) or type(decision.get('supported')) is not bool:
        raise ValueError('Missing explicit KB support decision')
    if decision['supported'] is False:
        return None
    answer = decision.get('answer')
    citations = decision.get('source_indices')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Empty grounded answer')
    if not isinstance(citations, list) or not citations:
        raise ValueError('Grounded answer must cite KB sources')
    if any(type(i) is not int or not 1 <= i <= source_count for i in citations):
        raise ValueError('Invalid KB citation')
    return {'answer': answer.strip(), 'source_indices': sorted(set(citations))}
