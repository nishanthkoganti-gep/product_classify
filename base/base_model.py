import numpy as np
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
