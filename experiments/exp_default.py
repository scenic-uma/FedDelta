"""
Run one configuration (config_experiment.yaml, or FEDDELTA_CONFIG=<file>).

    python experiments/exp_default.py
"""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
from src.simulation.runner import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by the user.")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise