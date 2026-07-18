"""Main entry point for running the PLGA Nanoparticle Drug Delivery computational simulation.

This script configures logging, initializes boundary physics, and executes Phase 1 
mass-transport simulations and results 1-6 parameter studies.
"""

from src.solver import run_simulation
from src.utils import configure_logging

# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    configure_logging()
    results = run_simulation()
