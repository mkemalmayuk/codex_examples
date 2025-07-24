import csv
import os
import sys


def main() -> None:
    """Read fund_data.csv and print its rows."""
    csv_path = os.path.join(os.path.dirname(__file__), "fund_data.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                print(", ".join(row))
        except BrokenPipeError:
            # Allow piping to utilities like ``head`` without traceback
            pass


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
