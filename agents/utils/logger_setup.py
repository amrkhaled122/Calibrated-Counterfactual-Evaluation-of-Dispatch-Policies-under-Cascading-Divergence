"""
Logger setup shared by preprocessing, agents, and simulation helpers.
"""
import logging
import sys


def get_logger(name: str = "calibrated_counterfactual_dispatch") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
