"""Neural network model definitions."""

from lpcanet.models.coupling import CouplingOperator, GNNBoundaryCorrection, GNNInterfaceCorrection
from lpcanet.models.fno import FNO2d
from lpcanet.models.mlp import MLP, count_parameters

__all__ = [
    "CouplingOperator",
    "GNNBoundaryCorrection",
    "GNNInterfaceCorrection",
    "FNO2d",
    "MLP",
    "count_parameters",
]
