#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from stock_pricerque.crew import StockPickerque

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Ejecuta la crew para seleccionar la mejor empresa para inversión.
    """
    inputs = {
        'sector': 'Tecnología',
        "current_date": str(datetime.now())
    }

    # Create and run the crew
    result = StockPickerque().crew().kickoff(inputs=inputs)

    # Print the result
    print("\n\n=== DECISION FINAL ===\n\n")
    print(result.raw)


if __name__ == "__main__":
    run()
