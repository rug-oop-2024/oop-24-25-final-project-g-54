"""
This package contains the regression models.
"""

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression
)
from autoop.core.ml.model.regression.lasso_regression import Lasso
from autoop.core.ml.model.regression.gradient_boosting_regressor import (
    GradientBoostingR)

__all__ = [
    "MultipleLinearRegression",
    "Lasso",
    "GradientBoostingR"
]
