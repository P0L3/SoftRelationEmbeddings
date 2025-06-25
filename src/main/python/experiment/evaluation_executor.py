# -*- coding: utf-8 -*-

import expbase as xb
import experiment.config as config


class EvaluationExecutor(xb.EvaluationExecutor):
    """
    This is a placeholder for the evaluation logic.
    The original file was missing, but is required for the program to run.
    This class will simply do nothing when called.
    """
    def _init(self) -> None:
        """Initializes the evaluation executor."""
        pass

    def _run_evaluation(self) -> None:
        """This is a placeholder and will not run any evaluation."""
        print("="*50)
        print("--- SKIPPING EVALUATION (Placeholder Executor) ---")
        print("="*50)
        # We don't need to do anything here for now.
        pass