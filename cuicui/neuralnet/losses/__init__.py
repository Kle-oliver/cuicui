"""
Loss functions are used to measure how well a model is performing.
They quantify the difference between the predicted output and the
actual output. We use these functions to guide the model is learning process
by indicating how much the model is predictions need to improve.
"""

from .base import Loss
from .mse import MSE

__all__ = ['Loss', 'MSE']
