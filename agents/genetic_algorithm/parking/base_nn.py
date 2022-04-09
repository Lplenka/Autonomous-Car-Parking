from abc import ABC, abstractmethod

import numpy as np


class neural_network(ABC):
    @abstractmethod
    def retrieve_weight_and_bias(self) -> np.array:
        pass

    @abstractmethod
    def weight_bias_reform(self, weights_biases: np.array) -> None:
        pass

    def load(self, file):
        self.weight_bias_reform(np.load(file))
