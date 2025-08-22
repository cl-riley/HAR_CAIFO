"""
CAIFO (Class-Aware Iterative Feature Optimisation) package.

This module provides the core components for implementing the CAIFO algorithm 
for improved Human Activity Recognition (HAR) performance.

The CAIFO algorithm works by iteratively identifying underperforming classes
and optimising both feature representation and class weights to improve their
recognition while maintaining overall model performance.
"""

__version__ = "1.0.0"

from .bayesian_opt import BayesianHyperOptimizer
from .feature_opt import FeatureOptimizer
from .class_opt import ClassWeightOptimizer
from .exit_criteria import ExitCriteria

__all__ = [
    'BayesianHyperOptimizer',
    'FeatureOptimizer',
    'ClassWeightOptimizer',
    'ExitCriteria',
]