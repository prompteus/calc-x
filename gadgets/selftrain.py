import abc
import math
import random
import collections
import functools
import itertools
import pathlib
import json
import uuid
from typing import NamedTuple, Iterator, Iterable, Generic, TypeVar

import numpy as np
import torch
import transformers
import more_itertools
import sklearn.pipeline

import gadgets.markup
import gadgets.metrics


class Experience(NamedTuple):
    problem_id: str
    prediction_id: str
    is_correct: bool
    style_score: float | None = None
    prompt: str | None = None
    prediction: str | None = None
    

class ExperiencePreferencePair(NamedTuple):
    accepted: Experience
    rejected: Experience

    @property
    def problem_id(self):
        if self.accepted.problem_id != self.rejected.problem_id:
            raise ValueError("accepted and rejected experiences must have the same id")
        return self.accepted.problem_id


class ExperienceCollector:
    def __init__(
        self,
        problem_ids: list[str],
        prompts: list[str],
        results: list[str],
        num_preds_per_example: int,
        sample_least_successful_with_prob: float,
        batch_size: int,
        generation_config: transformers.GenerationConfig,
        seed: int = 0,
        prefill: list[Experience] | None = None,
        prefill_buffer_do_yield: bool | None = None,
        style_classifier: sklearn.pipeline.Pipeline | None = None,
    ):
        if len(problem_ids) != len(prompts) or len(prompts) != len(results):
            raise ValueError("ids, prompts, and results must have the same length")
        self.trainer: transformers.Trainer = None
        self.problem_ids = np.array(problem_ids)
        self.prompts = np.array(prompts)
        self.results = np.array(results)
        self.trials = np.zeros(len(prompts), dtype=np.int64)
        self.successes = np.zeros(len(prompts), dtype=np.int64)
        self.num_preds_per_example = num_preds_per_example
        self.sample_least_successful_with_prob = sample_least_successful_with_prob
        self.seed = seed
        self.batch_size = batch_size
        self.random_gen = np.random.default_rng(seed)
        self.generation_config = generation_config
        self.prefill = prefill
        if prefill is not None and prefill_buffer_do_yield is None:
            raise ValueError("prefill_buffer_do_yield must be set when prefill is not None")
        self.prefill_buffer_do_yield = prefill_buffer_do_yield
        self.style_classifier = style_classifier


    def set_trainer(self, trainer: transformers.Trainer) -> None:
        self.trainer = trainer

    def _pick_example(self) -> int:
        x = self.random_gen.random()
        if x < self.sample_least_successful_with_prob:
            choose_from = self.successes
        else:
            choose_from = self.trials
        minimum = choose_from.min()
        candidates = np.nonzero(choose_from == minimum)[0]
        idx = self.random_gen.choice(candidates)
        return idx.item()

    def _example_sampler(self) -> Iterator[int]:
        while True:
            idx = self._pick_example()
            for _ in range(self.num_preds_per_example):
                yield idx

    def __iter__(self) -> Iterator[list[Experience]]:
        if self.prefill is not None:
            for experience_batch in more_itertools.chunked(self.prefill, self.num_preds_per_example, strict=True):
                if not all(exp.problem_id == experience_batch[0].problem_id for exp in experience_batch):
                    raise ValueError("prefill must contain experiences in batches of the same problem of size `num_preds_per_example`")
                for exp in experience_batch:
                    self.trials[self.problem_ids == exp.problem_id] += 1
                    self.successes[self.problem_ids == exp.problem_id] += exp.is_correct
                if self.prefill_buffer_do_yield:
                    yield experience_batch

        example_gen = self._example_sampler()
        batch_gen = more_itertools.batched(example_gen, self.batch_size)
        # queue ensures that we yield all predictions for an example together
        # regardless of the actual model batch size
        queue = collections.deque()

        while True:
            batch_idxs = np.array(next(batch_gen))
            problem_ids = self.problem_ids[batch_idxs]
            results = self.results[batch_idxs].tolist()
            prompts = self.prompts[batch_idxs].tolist()
            if self.trainer is None:
                raise ValueError("trainer must be set with .set_trainer() before collecting experience")
            # !important!
            # model is always accessed through the trainer and inside this loop
            # trainer can move the model to a different device which creates a copy
            # so we must ensure we are actually using the same model as the trained and not some "forgotten" copy
            inputs = self.trainer.tokenizer(prompts, return_tensors="pt", padding=True).to(self.trainer.model.device)
            pred_tokens = self.trainer.model.generate(**inputs, generation_config=self.generation_config)
            predictions = self.trainer.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True, spaces_between_special_tokens=False)
            pred_results = [gadgets.markup.get_result_from_output(pred) for pred in predictions]
            is_correct = gadgets.metrics.are_results_same(pred_results, results)
            prediction_ids = [str(uuid.uuid4()) for _ in range(len(is_correct))]
            self.trials[batch_idxs] += 1
            self.successes[batch_idxs] += is_correct
            style_scores = self.style_classifier.predict_proba(predictions)[:, 1]
            for experience in zip(problem_ids, prediction_ids, is_correct, style_scores.round(3).tolist(), prompts, predictions):
                queue.append(Experience(*experience))
                while len(queue) >= self.num_preds_per_example:
                    yield [queue.popleft() for _ in range(self.num_preds_per_example)]


def cycle(iterable: Iterable, n: int | None = None) -> Iterator:
    """
    repeat(10, 3) --> 10 10 10
    repeat(10) --> 10 10 10 ...
    """
    if n is None:
        return itertools.cycle(iterable)
    return itertools.chain.from_iterable(itertools.repeat(iterable, n))


class MakeSFTExamples:
    def __init__(
        self,
        random_gen: random.Random,
        target_min_examples_per_problem: int | None = None,
        max_examples_per_problem: int | None = None,
        max_oversample: int | None = None,
        prefer_good_style: bool = False,
        style_score_threshold: float | None = None,
    ):
        self.random_gen = random_gen
        self.target_min_examples = target_min_examples_per_problem
        self.max_examples = max_examples_per_problem
        self.max_oversample = max_oversample
        self.prefer_good_style = prefer_good_style
        self.style_score_threshold = style_score_threshold
        if self.target_min_examples is not None and self.max_examples is not None and self.target_min_examples > self.max_examples:
            raise ValueError("target_min_examples must be less than or equal to max_examples")

    def __call__(self, experience: list[Experience]) -> list[Experience]:
        assert all(exp.problem_id == experience[0].problem_id for exp in experience)
        experience = [exp for exp in experience if exp.is_correct]
        if self.prefer_good_style and all(exp.style_score is not None for exp in experience):
            experience = [exp for exp in experience if exp.style_score > self.style_score_threshold]

        if self._is_too_many(experience):
            return self._undersample(experience)

        if self._is_too_few(experience):
            return self._oversample(experience)
        
        return experience

    def _is_too_many(self, experience: list[Experience]) -> bool:
        if self.max_examples is None:
            return False
        return len(experience) >= self.max_examples
    
    def _is_too_few(self, experience: list[Experience]) -> bool:
        if self.target_min_examples is None:
            return False
        return len(experience) < self.target_min_examples

    def _oversample(self, experience: list[Experience]) -> list[Experience]:
        self.random_gen.shuffle(experience)
        experience = cycle(experience, self.max_oversample)
        experience = itertools.islice(experience, self.target_min_examples)
        return list(experience)

    def _undersample(self, experience: list[Experience]) -> list[Experience]:
        return self.random_gen.sample(experience, self.max_examples)



class MakePreferencePairs:
    def __init__(
        self,
        random_gen: random.Random,
        max_pairs: int | None = None,
        target_min_pairs: int | None = None,
        max_oversample_accepted: int | None = None,
        prefer_good_style: bool = False,
        style_score_margin: float | None = None,
    ):
        self.random_gen = random_gen
        self.max_pairs = max_pairs
        self.min_target_pairs = target_min_pairs
        self.max_oversample_accepted = max_oversample_accepted
        self.prefer_good_style = prefer_good_style
        self.style_score_margin = style_score_margin
        if prefer_good_style and style_score_margin is None:
            raise ValueError("`style_score_margin` must be set when `prefer_good_style` is True")

    def __call__(self, experience: list[Experience]) -> list[ExperiencePreferencePair]:
        assert all(exp.problem_id == experience[0].problem_id for exp in experience)
        if self.prefer_good_style and all(exp.style_score is not None for exp in experience):
            corrects = {exp for exp in experience if exp.is_correct}
            incorrects = {exp for exp in experience if not exp.is_correct}
            gap = self.style_score_margin
            all_prefs = {exp: set() for exp in experience}
            for correct in corrects:
                # correct nice > correct ugly
                all_prefs[correct].update({other_correct for other_correct in corrects if correct.style_score - other_correct.style_score > gap})
                # correct nice > incorrect nice
                # correct nice > incorrect ugly
                # correct ugly > incorrect ugly
                all_prefs[correct].update({incorrect for incorrect in incorrects if not incorrect.style_score - correct.style_score > gap})
            for incorrect in incorrects:
                # incorrect nice > incorrect ugly
                all_prefs[incorrect].update({other_incorrect for other_incorrect in incorrects if incorrect.style_score - other_incorrect.style_score > gap})
        else:
            if self.prefer_good_style:
                raise ValueError("`prefer_good_style` is set but some experiences do not have `style_score` property")
            accepteds = [exp for exp in experience if exp.is_correct]
            rejecteds = [exp for exp in experience if not exp.is_correct]
            all_prefs = {acc: rejecteds for acc in accepteds}
        return self._sample_pairs(all_prefs)

    def _sample_pairs(self, all_prefs: dict[Experience, set[Experience]]) -> list[ExperiencePreferencePair]:
        """
        Samples preference pairs of accepted and rejected experiences.
        Ensures that:
          (1) the number of pairs is at most `max_pairs`
          (2) all accepteds (and all rejecteds) are used almost the same number of times
          (3) each accepted is used at most `max_oversample_accepted` times
          (4) at least `min_target_pairs` are created if possible without violating (3)
        """
        all_prefs = {acc: list(rejecteds) for acc, rejecteds in all_prefs.items() if len(rejecteds) > 0}

        for rejecteds in all_prefs.values():
            self.random_gen.shuffle(rejecteds)

        prefs = all_prefs.copy()

        # undersample accepted predictions that appear too many times
        if self.max_oversample_accepted is not None:
            for accepted, rejecteds in prefs.items():
                prefs[accepted] = rejecteds[:self.max_oversample_accepted]
        
        if len(prefs) == 0:
            return []

        # oversample accepted predictions that appear too few times
        # without violating the oversampling limit
        if self.min_target_pairs is not None:
            for _ in range(self.min_target_pairs):
                least_represented_accepted = min(prefs, key=lambda acc: len(prefs[acc]))
                its_rejecteds = prefs[least_represented_accepted]
                its_rejecteds_orig = all_prefs[least_represented_accepted]
                if self._num_pairs(prefs) >= self.min_target_pairs:
                    # stop when enough pairs are created
                    break
                if self.max_pairs is not None and self._num_pairs(prefs) >= self.max_pairs:
                    # stop before too many pairs
                    break
                if len(its_rejecteds) >= self.max_oversample_accepted:
                    # stop before too much oversampling
                    break
                # oversample the least represented accepted by one
                # each rejected will be used aproximately the same number of times for this accepted
                new_len = len(its_rejecteds) + 1
                prefs[least_represented_accepted] = list(itertools.islice(cycle(its_rejecteds_orig), new_len))
                assert len(prefs[least_represented_accepted]) == new_len
        
        pairs = [
            ExperiencePreferencePair(accepted, rejected)
            for accepted, rejecteds in prefs.items()
            for rejected in rejecteds
        ]

        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            self.random_gen.shuffle(pairs)
            pairs = pairs[:self.max_pairs]

        return pairs

    @staticmethod
    def _num_pairs(prefs: dict[Experience, list[Experience]]) -> int:
        return sum(len(rejecteds) for rejecteds in prefs.values())

    # def _sample_pairs(self, accepteds: list[Experience], rejecteds: list[Experience]) -> list[ExperiencePreferencePair]:
        
    #     if self.max_pairs is None and self.max_oversample_accepted is None:
    #         pairs = itertools.product(accepteds, rejecteds)
    #     else:
    #         len_upper_bound = math.inf
    #         if self.max_pairs is not None:
    #             len_upper_bound = self.max_pairs
    #         if self.max_oversample_accepted is not None:
    #             len_upper_bound = min(len_upper_bound, self.max_oversample_accepted * len(accepteds))
    #         # this makes sure that rejecteds iterator will be always finite
    #         assert not math.isinf(len_upper_bound)
    #         len_upper_bound = int(len_upper_bound)

    #         self.random_gen.shuffle(accepteds)
    #         accepteds = cycle(accepteds, self.max_oversample_accepted)
    #         rejecteds = cycle(rejecteds, self.min_target_pairs)
    #         rejecteds = itertools.islice(rejecteds, len_upper_bound)
    #         rejecteds = list(rejecteds)
    #         self.random_gen.shuffle(rejecteds)
    #         pairs = zip(accepteds, rejecteds)
    #     return [ExperiencePreferencePair(acc, rej) for acc, rej in pairs]


class BalancerByLabel:
    def __init__(self, random_gen: random.Random):
        self.random_gen = random_gen

    def __call__(self, experience: list[Experience]) -> list[Experience]:
        accepteds = [exp for exp in experience if exp.is_correct]
        rejecteds = [exp for exp in experience if not exp.is_correct]
        min_len = min(len(accepteds), len(rejecteds)) 
        accepteds = self._drop_overrepresented(accepteds, min_len)
        rejecteds = self._drop_overrepresented(rejecteds, min_len)
        return more_itertools.interleave(accepteds, rejecteds)
               
    def _drop_overrepresented(self, experiences: list[Experience], keep_n: int) -> list[Experience]:
        """
        returns `keep_n` experiences by dropping the overrepresented ones
        """
        if len(experiences) <= keep_n:
            return experiences
        output = experiences.copy()
        self.random_gen.shuffle(output)
        output = sorted(output, key=lambda x: x["id"])
        output = itertools.groupby(output, key=lambda x: x["id"])
        output = (list(exps) for group_id, exps in output)
        output = more_itertools.interleave_longest(*output)
        output = itertools.islice(output, keep_n)
        return list(output)
            

class ExperienceLogger:
    def __init__(
        self,
        log_file: str | None,
        print_to_stdout: bool,
    ):
        if log_file is not None:
            log_file = pathlib.Path(log_file)
            if log_file.exists():
                raise ValueError(f"{log_file} already exists")
        self.log_file: pathlib.Path | None = log_file
        self.print_to_stdout = print_to_stdout

    def __call__(self, experience: list[Experience]):
        if self.print_to_stdout:
            for exp in experience:
                print(exp)
        if self.log_file is not None:
            with self.log_file.open("a") as f:
                for exp in experience:
                    f.write(json.dumps(exp._asdict()) + "\n")
                f.flush()



TrackedItem = TypeVar("TrackedItem")
class RollingWindowTracker(Generic[TrackedItem], abc.ABC):
    def __init__(
        self,
        rolling_window_size: int,
        report_after_every_n_problems: int,
        use_wandb: bool,
        use_stdout: bool,
        metric_prefix: str | None = None,
    ):
        self.rolling_window: collections.deque[TrackedItem]
        self.rolling_window = collections.deque(maxlen=rolling_window_size)
        self.report_after_every_n_problems = report_after_every_n_problems
        self.counter = 0
        self.use_wandb = use_wandb
        self.use_stdout = use_stdout
        if metric_prefix is None:
            self.metric_prefix = self.default_metric_prefix
        else:
            self.metric_prefix = metric_prefix

    def __call__(self, experiences: list[Experience]):
        self.rolling_window.append(experiences)
        self.counter += 1
        if self.counter % self.report_after_every_n_problems == 0:
            self.report()

    def report(self):
        metrics = {f"{self.metric_prefix}{k}": v for k, v in self.get_metrics().items()}
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
        if self.use_stdout:
            print(metrics)

    @abc.abstractproperty
    def default_metric_prefix(self) -> str:
        pass

    @abc.abstractmethod
    def get_metrics(self) -> dict:
        pass


class ExperienceTracker(RollingWindowTracker[list[Experience]]):

    def __init__(self, num_preds_per_problem: int, style_score_printing_threshold: float, style_score_margin: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_preds_per_problem = num_preds_per_problem
        self.style_score_printing_threshold = style_score_printing_threshold
        self.style_score_margin = style_score_margin

    @property
    def default_metric_prefix(self) -> str:
        return "experience_tracker/"

    def get_metrics(self) -> dict:
        if len(self.rolling_window) == 0:
            return {"rolling_window_size": 0}
        
        # sanity check
        if not all(len(exps) == self.num_preds_per_problem for exps in self.rolling_window):
            raise ValueError("all problems must have `num_preds_per_problem` predictions. This looks like a bug.")
        
        flattened_window = list(itertools.chain.from_iterable(self.rolling_window))
        total_success_rate = sum(exp.is_correct for exp in flattened_window) / len(flattened_window)
        per_problem_success_rates = [sum(exp.is_correct for exp in exps) for exps in self.rolling_window]
        exactly_n_corrects_per_problem = [0] * (self.num_preds_per_problem + 1)
        for per_problem_success_rate in per_problem_success_rates:
            exactly_n_corrects_per_problem[per_problem_success_rate] += 1
        digits = len(str(self.num_preds_per_problem))
        exactly_n_correct_per_problem_str = {
            f"exactly_{str(n).zfill(digits)}_num_corrects_per_problem": count / len(self.rolling_window)
            for n, count in enumerate(exactly_n_corrects_per_problem)
        }

        metrics = {
            "experience_counter": self.counter, 
            "num_total_generated_predictions": self.counter * self.num_preds_per_problem,
            "avg_num_corrects_per_problem": total_success_rate,
            "num_none_corrects": exactly_n_corrects_per_problem[0] / len(self.rolling_window),
            "num_all_corrects": exactly_n_corrects_per_problem[-1] / len(self.rolling_window),
            "num_some_corrects": sum(exactly_n_corrects_per_problem[1:-1]) / len(self.rolling_window),    
            "rolling_window_size": len(self.rolling_window),
            **exactly_n_correct_per_problem_str
        }

        if self.use_wandb:
            import wandb
            metrics["distr_num_corrects_per_problem"] = wandb.Histogram(per_problem_success_rates)

        if all(exp.style_score is not None for exp in flattened_window):
            style_scores = [exp.style_score for exp in flattened_window if exp.style_score is not None]
            total_good_style_rate = sum(exp.style_score > self.style_score_printing_threshold for exp in flattened_window) / len(flattened_window)
            per_problem_good_style_rates = [sum(exp.style_score > self.style_score_printing_threshold for exp in exps) for exps in self.rolling_window]
            exactly_n_good_style_per_problem = [0] * (self.num_preds_per_problem + 1)
            for per_problem_good_style_rate in per_problem_good_style_rates:
                exactly_n_good_style_per_problem[per_problem_good_style_rate] += 1
            digits = len(str(self.num_preds_per_problem))
            exactly_n_good_style_per_problem_str = {
                f"exactly_{str(n).zfill(digits)}_good_style_per_problem": count / len(self.rolling_window)
                for n, count in enumerate(exactly_n_good_style_per_problem)
            }

            metrics.update({
                "avg_style_score": sum(style_scores) / len(style_scores),
                "max_style_score": max(style_scores),
                "min_style_score": min(style_scores),
                "avg_good_style_rate": total_good_style_rate,
                "num_all_with_good_style": exactly_n_good_style_per_problem[-1] / len(self.rolling_window),
                "num_none_with_good_style": exactly_n_good_style_per_problem[0] / len(self.rolling_window),
                "num_some_with_good_style": sum(exactly_n_good_style_per_problem[1:-1]) / len(self.rolling_window),
                **exactly_n_good_style_per_problem_str
            })
            if self.use_wandb:
                import wandb
                metrics["distr_good_style_per_problem"] = wandb.Histogram(per_problem_good_style_rates)
                metrics["distr_style_scores"] = wandb.Histogram(style_scores)

            if self.style_score_margin is not None:
                num_style_pairs = []
                for exps in self.rolling_window:
                    num_style_pairs.append(0)
                    for exp1 in exps:
                        for exp2 in exps:
                            if exp1.style_score - exp2.style_score > self.style_score_margin:
                                num_style_pairs[-1] += 1
                metrics["avg_num_style_pairs"] = sum(num_style_pairs) / len(num_style_pairs),

                if self.use_wandb:
                    import wandb
                    metrics["distr_num_style_pairs"] = wandb.Histogram(num_style_pairs)

        return metrics
    

class NumPairsTracker(RollingWindowTracker[list[ExperiencePreferencePair]]):
    @property
    def default_metric_prefix(self) -> str:
        return "num_pairs_tracker/"
    
    def get_metrics(self) -> dict:
        if len(self.rolling_window) == 0:
            return {"rolling_window_size": 0}
        
        num_pairs = [len(pairs) for pairs in self.rolling_window]
        metrics = {
            "avg_num_pairs_per_problem": sum(num_pairs) / len(num_pairs),
            "max_num_pairs_per_problem": max(num_pairs),
            "min_num_pairs_per_problem": min(num_pairs),
            "rolling_window_size": len(self.rolling_window),
        }

        if self.use_wandb:
            import wandb
            metrics["distr_num_pairs_per_problem"] = wandb.Histogram(num_pairs)

        return metrics    



class SFTPreprocessor:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, experience: Experience) -> dict:
        return self.tokenizer(
            text=experience.prompt,
            text_target=experience.prediction
        )

class DPOPreprocessor:
    def __call__(self, experience: ExperiencePreferencePair) -> dict:
        return {
            "prompt": experience.accepted.prompt,
            "chosen": experience.accepted.prediction,
            "rejected": experience.rejected.prediction,
        }

class KTOPreprocessor:
    def __call__(self, experience: Experience) -> dict:
        return {
            "prompt": experience.prompt,
            "completion": experience.prediction,
            "label": experience.is_correct,
        }