import sys
import warnings
import json


def flatten_semilist(x):
    """Convert a semi-nested list - a list of (lists and scalars) - to a flat list."""
    # convert to a list of lists
    lists = [n if isinstance(n, list) else [n] for n in x]
    # flatten nested list
    return [e for el in lists for e in el]


class FromDictMixin():
    def __init__(self, *args, **kwargs):
        # apply json properties to existing attributes
        attributes = self.__dict__.keys()
        if args:
            if len(args) > 1:
                warnings.warn("Positional arguments after the first are ignored.")
            struct = args[0]
            for key in struct:
                if key in attributes:
                    setattr(self, key, self.load_attribute(key, struct[key]))
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, self.load_attribute(key, kwargs[key]))
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

    def load_attribute(self, key, value):
        return value

    def dump(self):
        prop_dict = vars(self)
        return recursive_dump(prop_dict)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            all(getattr(other, attr, None) == getattr(self, attr, None) for attr in self.__dict__.keys()))

    def __hash__(self):
        return hash(json.dumps(self.dump()))


def recursive_dump(value):
    # recursively call dump() for nested objects to generate a json-serializable dict
    # this is not entirely reversible because there is no distinction between sets and lists in dict form
    if isinstance(value, dict):
        return {key:recursive_dump(value[key]) for key in value}
    elif isinstance(value, list):
        return [recursive_dump(v) for v in value]
    elif isinstance(value, set):
        return [recursive_dump(v) for v in value]
    else:
        try:
            return value.dump()
        except AttributeError:
            return value


class Text:
    """Utilities for processing text."""

    @staticmethod
    def get_curie(text):
        return text.upper().split(':')[0] if ':' in text else None

    @staticmethod
    def un_curie(text):
        return text.split(':')[1] if ':' in text else text

    @staticmethod
    def short(obj, limit=80):
        text = str(obj) if obj else None
        return (text[:min(len(text), limit)] + ('...' if len(text) > limit else '')) if text else None

    @staticmethod
    def path_last(text):
        return text.split('/')[-1:][0] if '/' in text else text

    @staticmethod
    def obo_to_curie(text):
        return ':'.join(text.split('/')[-1].split('_'))

    @staticmethod
    def snakify(text):
        decomma = '_'.join(text.split(','))
        dedash = '_'.join(decomma.split('-'))
        resu = '_'.join(dedash.split())
        return resu
