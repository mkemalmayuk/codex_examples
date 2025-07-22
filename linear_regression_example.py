"""Simple linear regression example in pure Python.

This module implements a tiny linear regression routine without using any
external dependencies. It is meant purely for educational purposes.
"""

from typing import List, Tuple


def mean(values: List[float]) -> float:
    """Return the arithmetic mean of a sequence of numbers."""
    return sum(values) / len(values)

def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Return the slope and intercept for a set of paired observations."""
    if len(xs) != len(ys):
        raise ValueError("Input lists must have the same length")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Cannot compute a slope when all x values are equal")
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict(x: float, slope: float, intercept: float) -> float:
    """Return the predicted y value for x using the given model."""
    return slope * x + intercept


def r_squared(xs: List[float], ys: List[float], slope: float, intercept: float) -> float:
    """Return the coefficient of determination for the given data."""
    mean_y = mean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - predict(x, slope, intercept)) ** 2 for x, y in zip(xs, ys))
    return 1 - ss_res / ss_tot

def main() -> None:
    """Demonstrate linear regression with a tiny example dataset."""
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 5, 4, 5]
    slope, intercept = linear_regression(xs, ys)
    r2 = r_squared(xs, ys, slope, intercept)
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R^2: {r2:.3f}")
    for x in xs:
        print(f"Prediction for {x}: {predict(x, slope, intercept):.3f}")

if __name__ == "__main__":
    main()
