import warnings
import networkx as nx

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
                    setattr(self, key, self.preprocess(key, struct[key]))
                else:
                    warnings.warn("JSON field {} ignored.".format(key))

        # override any json properties with the named ones
        for key in kwargs:
            if key in attributes:
                setattr(self, key, self.preprocess(key, kwargs[key]))
            else:
                warnings.warn("Keyword argument {} ignored.".format(key))

    def preprocess(self, key, value):
        return value