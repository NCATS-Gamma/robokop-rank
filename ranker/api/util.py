import warnings


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


class FromDictMixin2():
    """Mixin for transformation to/from json/dict."""

    constructors = {}

    def __init__(self, *args, **kwargs):
        """Initialize FromDictMixin."""
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
        """Convert json attributes to objects on input."""
        if key in self.constructors:
            if isinstance(value, list):
                value = [self.constructors[key](x) for x in value]
            else:
                value = self.constructors[key](value)
        return value

    def dump(self):
        """Dump object to json."""
        json = self.__dict__
        for key in json:
            if isinstance(json[key], FromDictMixin2):
                json[key] = json[key].dump()
            if isinstance(json[key], list) and json[key] and isinstance(json[key][0], FromDictMixin2):
                json[key] = [v.dump() for v in json[key]]
        return json
