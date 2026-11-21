"""MIPROv2 subclass supporting checkpointing and cancellation."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dspy.teleprompt import MIPROv2

logger = logging.getLogger(__name__)


class GracefulExit(Exception):
    """Exception raised to interrupt Optuna study optimization gracefully."""
    pass


def _import_optuna():
    """Helper to dynamically import optuna."""
    import optuna
    return optuna


class CheckpointingMIPROv2(MIPROv2):
    """Subclass of MIPROv2 that supports checkpointing and graceful exit on SIGTERM/cancel event."""

    def __init__(self, *args, task_name: str = "unknown", cancel_event: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.cancel_event = cancel_event

    def _save_checkpoint(self, program: Any, score: float, params: Optional[Dict[str, Any]] = None) -> None:
        checkpoint_path = Path(f"data/processed/agents/{self.task_name}_checkpoint.json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json
            checkpoint_data = {
                "score": score,
                "program": program.dump_state(),
                "params": params,
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved checkpoint to {checkpoint_path} with score {score:.4f}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        evaluate: Any,
        valset: list,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Any | None:
        import json
        import os
        import time
        from collections import defaultdict
        
        from dspy.evaluate.evaluate import Evaluate
        from dspy.teleprompt.utils import (
            eval_candidate_program,
            save_candidate_program,
            get_program_with_highest_avg_score,
            print_full_program,
        )

        optuna = _import_optuna()

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )

        # Compute the adjusted total trials that we will run (including full evals)
        run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_num_trials = int(
            (num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end)
            if minibatch
            else num_trials
        )
        logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

        default_score = eval_candidate_program(len(valset), valset, program, evaluate, self.rng).score
        logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = save_candidate_program(program, self.log_dir, -1)
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deepcopy()

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        
        # Resume from checkpoint if it exists
        checkpoint_path = Path(f"data/processed/agents/{self.task_name}_checkpoint.json")
        resumed_params = None
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                resumed_score = checkpoint_data.get("score")
                if resumed_score is not None and resumed_score > best_score:
                    best_score = resumed_score
                    best_program = program.deepcopy()
                    best_program.load_state(checkpoint_data.get("program"))
                    resumed_params = checkpoint_data.get("params")
                    logger.info(f"Resumed from checkpoint: best score is now {best_score:.4f}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": best_program.deepcopy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}
        
        GREEN = "\033[92m"
        ENDC = "\033[0m"

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, score_data

            if self.cancel_event and self.cancel_event.is_set():
                logger.info("Cancellation event set. Gracefully exiting MIPROv2 trial loop.")
                raise GracefulExit()

            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
            else:
                logger.info(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            chosen_params, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            if self.verbose:
                logger.info("Evaluating the following candidate program...\n")
                print_full_program(candidate_program)

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(batch_size, valset, candidate_program, evaluate, self.rng).score
            total_eval_calls += batch_size

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
                logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")
                self._save_checkpoint(best_program, best_score, trial.params)

            # Log evaluation results
            score_data.append(
                {"score": score, "program": candidate_program, "full_eval": batch_size >= len(valset)}
            )  # score, prog, full_eval
            if minibatch:
                self._log_minibatch_eval(
                    score,
                    best_score,
                    batch_size,
                    chosen_params,
                    score_data,
                    trial,
                    adjusted_num_trials,
                    trial_logs,
                    trial_num,
                    candidate_program,
                    total_eval_calls,
                )
            else:
                self._log_normal_eval(
                    score,
                    best_score,
                    chosen_params,
                    score_data,
                    trial,
                    num_trials,
                    trial_logs,
                    trial_num,
                    valset,
                    batch_size,
                    candidate_program,
                    total_eval_calls,
                )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate_program, raw_chosen_params),
            )

            # If minibatch, perform full evaluation at intervals (and at the very end)
            if minibatch and (
                (trial_num % (minibatch_full_eval_steps + 1) == 0) or (trial_num == (adjusted_num_trials - 1))
            ):
                old_best = best_score
                best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                    trial_num,
                    adjusted_num_trials,
                    param_score_dict,
                    fully_evaled_param_combos,
                    evaluate,
                    valset,
                    trial_logs,
                    total_eval_calls,
                    score_data,
                    best_score,
                    best_program,
                    study,
                    instruction_candidates,
                    demo_candidates,
                )
                if best_score > old_best:
                    self._save_checkpoint(best_program, best_score, trial.params)

            return score

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})

        # Add default run as a baseline in optuna (TODO: figure out how to weight this by # of samples evaluated on)
        trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
            value=default_score,
        )
        study.add_trial(trial)
        
        # Add resumed trial if available
        if resumed_params:
            try:
                resumed_trial = optuna.trial.create_trial(
                    params=resumed_params,
                    distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
                    value=best_score,
                )
                study.add_trial(resumed_trial)
                logger.info(f"Added resumed baseline to Optuna study with score {best_score}")
            except Exception as e:
                logger.warning(f"Could not add resumed trial to Optuna study: {e}")

        try:
            study.optimize(objective, n_trials=num_trials)
        except GracefulExit:
            logger.info("Gracefully interrupted MIPROv2 optimization. Returning best program found so far.")

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls
            sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
            # Attach all minibatch programs
            best_program.mb_candidate_programs = [
                score_data for score_data in sorted_candidate_programs if not score_data["full_eval"]
            ]
            # Attach all programs that were evaluated on the full trainset, in descending order of score
            best_program.candidate_programs = [
                score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
            ]

        logger.info(f"Returning best identified program with score {best_score}!")

        return best_program
