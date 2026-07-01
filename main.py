from src.solver import run_simulation
from src.utils import configure_logging

# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    configure_logging()
    results = run_simulation()
