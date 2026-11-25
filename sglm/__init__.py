"""Package exports are lazy so the independent evaluator does not import models/Torch."""
__all__=['Matcher']


def __getattr__(name):
    if name=='Matcher':
        from .model import Matcher
        return Matcher
    raise AttributeError(name)
