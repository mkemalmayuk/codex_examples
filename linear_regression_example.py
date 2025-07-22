"""Simple multiple linear regression example in pure Python.

This module implements a tiny linear regression routine without using any
external dependencies. It fits a plane to points (x, y, z) so that ``z`` can
be predicted from ``x`` and ``y``. It is meant purely for educational
purposes.
"""

from typing import List, Tuple


def mean(values: List[float]) -> float:
    """Return the arithmetic mean of a sequence of numbers."""
    return sum(values) / len(values)

def multiple_linear_regression(
    xs: List[float], ys: List[float], zs: List[float]
) -> Tuple[float, float, float]:
    """Return slopes for ``x`` and ``y`` and the intercept fitting ``z``."""

    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("Input lists must have the same length")

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_z = sum(zs)
    sum_xx = sum(x * x for x in xs)
    sum_yy = sum(y * y for y in ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xz = sum(x * z for x, z in zip(xs, zs))
    sum_yz = sum(y * z for y, z in zip(ys, zs))

    a = [
        [n, sum_x, sum_y],
        [sum_x, sum_xx, sum_xy],
        [sum_y, sum_xy, sum_yy],
    ]
    b = [sum_z, sum_xz, sum_yz]

    coeffs = _solve_3x3(a, b)
    intercept, slope_x, slope_y = coeffs
    return slope_x, slope_y, intercept


def _solve_3x3(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a 3x3 linear system using Gaussian elimination."""

    m = [row[:] for row in a]
    v = b[:]
    n = 3

    for i in range(n):
        pivot = i
        for j in range(i + 1, n):
            if abs(m[j][i]) > abs(m[pivot][i]):
                pivot = j
        if m[pivot][i] == 0:
            raise ValueError("Singular matrix")
        if pivot != i:
            m[i], m[pivot] = m[pivot], m[i]
            v[i], v[pivot] = v[pivot], v[i]
        pivot_val = m[i][i]
        m[i] = [x / pivot_val for x in m[i]]
        v[i] /= pivot_val
        for j in range(i + 1, n):
            factor = m[j][i]
            m[j] = [m[j][k] - factor * m[i][k] for k in range(n)]
            v[j] -= factor * v[i]

    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = v[i] - sum(m[i][k] * x[k] for k in range(i + 1, n))
    return x


def predict(x: float, y: float, slope_x: float, slope_y: float, intercept: float) -> float:
    """Return the predicted z value for ``x`` and ``y`` using the model."""
    return slope_x * x + slope_y * y + intercept


def r_squared(
    xs: List[float],
    ys: List[float],
    zs: List[float],
    slope_x: float,
    slope_y: float,
    intercept: float,
) -> float:
    """Return the coefficient of determination for the given data."""

    mean_z = mean(zs)
    ss_tot = sum((z - mean_z) ** 2 for z in zs)
    if ss_tot == 0:
        raise ValueError("Cannot compute R^2 when all z values are equal")
    ss_res = sum(
        (z - predict(x, y, slope_x, slope_y, intercept)) ** 2
        for x, y, z in zip(xs, ys, zs)
    )
    return 1 - ss_res / ss_tot

def main() -> None:
    """Demonstrate multiple linear regression with a tiny dataset."""

    xs = [1, 2, 3, 4, 5]
    ys = [2, 1, 3, 5, 4]
    zs = [6, 5, 10, 15, 14]

    slope_x, slope_y, intercept = multiple_linear_regression(xs, ys, zs)
    r2 = r_squared(xs, ys, zs, slope_x, slope_y, intercept)

    print(f"Slope for x: {slope_x:.3f}")
    print(f"Slope for y: {slope_y:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R^2: {r2:.3f}")

    for x, y in zip(xs, ys):
        pred = predict(x, y, slope_x, slope_y, intercept)
        print(f"Prediction for ({x}, {y}): {pred:.3f}")

if __name__ == "__main__":
    main()
