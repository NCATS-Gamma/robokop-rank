import sys


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
