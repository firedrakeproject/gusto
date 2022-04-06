from abc import ABCMeta


class Tracer(object, metaclass=ABCMeta):

    def __init__(self, name, space):
        

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def space(self):
        pass


class PassiveTracer(Tracer):


class ActiveTracer(Tracer):
