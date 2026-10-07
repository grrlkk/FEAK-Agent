"""Constrain wire syntax; semantic action permission/budget errors stay observable."""
import json

from ..observation.graph import object_schema, array_schema, enum
from .environment import ALLOWED


def action_schema(role):
    string = {'type': 'string'}
    def args(name):
        if name in {'SCORE', 'AUDIT', 'UNDO'}:
            return object_schema({})
        if name == 'QUERY':
            return object_schema({'target': string})
        if name in {'PLAN', 'PROGRESS'}:
            return object_schema({'text': string})
        if name == 'DELEGATE':
            return object_schema({'agent': enum(['composition', 'cohesion']), 'task': string, 'scope': string})
        if name == 'FINISH':
            return object_schema({'summary': string, 'needs_explanation': array_schema(object_schema({'sid': string, 'reason': string}))})
        if name == 'REPORT':
            return object_schema({'status': enum(['done', 'blocked']), 'summary': string})
        if name == 'PREVIEW':
            return object_schema({'action': call(['MOVE', 'INSERT', 'DELETE'])})
        if name == 'MOVE':
            return object_schema({'target': string, 'position': string})
        if name == 'DELETE':
            return object_schema({'target': string})
        return object_schema({'target': string, 'new_text': string})
    def call(names):
        variants = {json.dumps(args(name), sort_keys=True): args(name) for name in names}
        return object_schema({'action': enum(names), 'args': {'anyOf': list(variants.values())}})
    return call(sorted(ALLOWED[role]))
