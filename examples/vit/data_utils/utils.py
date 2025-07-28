import logging
from enum import Enum
from typing import Any, List

import numpy
import torch

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def normalize(weights: List[float]) -> List[float]:
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w
