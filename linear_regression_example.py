"""Simple linear regression example in pure Python"""

from typing import List, Tuple

def mean(values: List[float]) -> float:
    return sum(values) / len(values)

def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) != len(ys):
        raise ValueError("Input lists must have the same length")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

def predict(x: float, slope: float, intercept: float) -> float:
    return slope * x + intercept

def main() -> None:
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 5, 4, 5]
    slope, intercept = linear_regression(xs, ys)
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    for x in xs:
        print(f"Prediction for {x}: {predict(x, slope, intercept):.3f}")

if __name__ == "__main__":
    main()
