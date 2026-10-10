"""Cache repeated questions and coalesce concurrent identical requests."""
from collections import OrderedDict
from copy import deepcopy
from threading import Event, Lock
import time


class QueryCache:
    def __init__(self, ttl=300, max_entries=256, clock=time.monotonic):
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self.lock = Lock()
        self.values = OrderedDict()
        self.pending = {}

    def run(self, key, generate):
        with self.lock:
            cached = self.values.get(key)
            if cached and cached[0] > self.clock():
                self.values.move_to_end(key)
                return deepcopy(cached[1])
            self.values.pop(key, None)
            work = self.pending.get(key)
            owner = work is None
            if owner:
                work = {'event': Event()}
                self.pending[key] = work
        if not owner:
            if not work['event'].wait(90):
                raise RuntimeError('Timed out waiting for a matching knowledge-base query')
            if 'error' in work:
                raise work['error']
            return deepcopy(work['value'])
        try:
            value = generate()
            with self.lock:
                work['value'] = deepcopy(value)
                # Do not cache an unverified no-answer result; provider capacity may recover.
                if value.get('source_type') in ('rag', 'internet'):
                    self.values[key] = (self.clock() + self.ttl, deepcopy(value))
                    while len(self.values) > self.max_entries:
                        self.values.popitem(last=False)
            return value
        except Exception as error:
            with self.lock:
                work['error'] = error
            raise
        finally:
            with self.lock:
                self.pending.pop(key, None)
                work['event'].set()
