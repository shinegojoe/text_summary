import abc

class IHyperOptimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self):
        return NotImplemented

class Random_search(IHyperOptimizer):
    def __init__(self, component, hp_generator=None):
        self.component = component
        self.random_hp_generator = hp_generator



    def run(self):
        for i in range(1):
            self.component.run()
