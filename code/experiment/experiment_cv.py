## Copyright 2018 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import random
import os
import numpy as np
import time

from datetime import datetime
from itertools import product
from collections import OrderedDict

from experiment.result import Result
from experiment.result import ExperimentResult
from experiment.result import CVExperimentResult
from cornac.metrics.ranking import RankingMetric
from cornac.metrics.rating import RatingMetric
from recommender import Recommender
from eval_methods.base_method import ranking_eval
from cornac.utils import get_rng

import gc

class ExperimentCV:
    """ Experiment Class

    Parameters
    ----------
    # eval_method: :obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., RatioSplit).
    
    # aco_eval_repeat: : int, optional, default: 3
        The evaluation repeat

    models: array of :obj:`<cornac.models.Recommender>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].

    metrics: array of :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].

    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.

    show_validation: bool, optional, default: True
        Whether to show the results on validation set (if exists).

    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None,
        models will NOT be stored and logs will be saved in the current working directory.

    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the test set, initially it is set to None.

    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.

    """

    def __init__(
        self,
        eval_method,
        models,
        metrics,
        user_based=True,
        show_validation=True,
        verbose=False,
        save_dir=None,
        recommendations=False,
        aco_eval_repeat=3,
    ):
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.show_validation = show_validation
        self.verbose = verbose
        self.save_dir = save_dir
        self.result = None
        self.val_result = None
        self.recommendations = recommendations
        self.aco_eval_repeat = aco_eval_repeat

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_metrics(input_metrics):
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from cornac.eval_methods.cross_validation import CrossValidation
        from cornac.eval_methods.propensity_stratified_evaluation import (
            PropensityStratifiedEvaluation,
        )

        if isinstance(self.eval_method, CrossValidation) or isinstance(
            self.eval_method, PropensityStratifiedEvaluation
        ):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()
            # if self.show_validation:
            # if self.show_validation and self.eval_method.val_set is not None:
                # self.val_result = ExperimentResult()

    def run(self):
        """Run the Cornac Experiment"""
        self._create_result()

        for model in self.models:
            print("Training model : {}".format(model.name))
            eval_all = None
            for fold_id, evals in enumerate(self.eval_method):
                model.fold_id=fold_id
                test_result, val_result = evals.evaluate(
                    model=model,
                    metrics=self.metrics,
                    user_based=self.user_based,
                    show_validation=self.show_validation,
                    aco_eval_repeat=self.aco_eval_repeat
                )

                test_result.metric_avg_results["Cov. %"] = (len(np.unique(model.recommended_item_set))/evals.train_set.num_items) * 100

                if eval_all is None:
                    eval_all = test_result
                else:
                    for key in test_result.metric_avg_results.keys():
                        eval_all.metric_avg_results[key] = (test_result.metric_avg_results.get(key, 0)
                                                            + eval_all.metric_avg_results.get(key, 0))

                if self.val_result is not None:
                    self.val_result.append(val_result)

                if not isinstance(self.result, CVExperimentResult):
                    model.save(self.save_dir)

            if len(self.eval_method)>1:
                for key in eval_all.metric_avg_results.keys():
                    eval_all.metric_avg_results[key] = float(eval_all.metric_avg_results.get(key, 0) / len(self.eval_method))
                test_result = eval_all
            self.result.append(test_result)

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        output += "\n ALL FOLD AVERAGE TEST:\n...\n{}".format(self.result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)

    def runcv(self):
        """Run the Cornac cross-validation experiment"""
        self._create_result()

        for model in self.models:

            """Generate searching points"""
            param_set = []

            if hasattr(model, "n_trails"):
                keys = [d.name for d in model.space]
                hyp_seed = (int(random.choice(range(0,10000000))))
                rng = get_rng(hyp_seed)
                while len(param_set) < model.n_trails:
                    params = [d._sample(rng) for d in model.space]
                    param_set.append(dict(zip(keys, params)))
            else:
                keys = [d.name for d in model.space]
                for params in product(*[sorted(d.values) for d in model.space]):
                    param_set.append(dict(zip(keys, params)))

            compare_op = np.greater if model.metric.higher_better else np.less
            model.best_score = -np.inf if model.metric.higher_better else np.inf
            model.best_model = None
            model.best_params = None

            for params in param_set:
                # if hasattr(model, "n_trails"):
                #     print("Evaluating: {}".format(params))

                eval_model = model.model.clone(params)

                # this part is updated for evaluations of AcoRec
                eval_repeat = 1
                if eval_model.name.startswith("AcoRec"):
                    eval_repeat = self.aco_eval_repeat

                sum_score = 0

                start = time.time()
                for i in range(eval_repeat):
                    ev_score = 0
                    for fold_id, fold in enumerate(model.eval_method):
                        tr_set = fold.train_set
                        vl_set = fold.val_set
                        eval_model.fold_id = fold_id
                        train_model = eval_model.fit(tr_set, vl_set)
                        # gc collect
                        gc.collect()
                        score = ranking_eval(
                            train_model,
                            [model.metric],
                            tr_set,
                            vl_set,
                            rating_threshold=1,
                            exclude_unknowns=True,
                            verbose=False,
                        )[0][0]
                        sum_score += score
                        ev_score += score

                mean_score = sum_score / (eval_repeat * len(model.eval_method))
                t_time = time.time() - start

                print("{} param. set.: {} ".format(str(eval_model.name).lower(), params))
                print("Mean {} = {:.4f} Time = {}".format(model.metric.name, mean_score, t_time))

                if compare_op(mean_score, model.best_score):
                    model.best_score = mean_score
                    model.best_model = eval_model
                    model.best_params = params

                del eval_model

            print("Best parameter settings: {}".format(model.best_params))
            print("{} = {:.4f}".format(model.metric.name, model.best_score))

            # test with best validation params
            test_model = model.model.clone(model.best_params)
            test_eval = model.eval_method
            model = None
            train_model = None

            # this part is updated for evaluations of AcoRec
            eval_repeat = 1
            if test_model.name.startswith("AcoRec"):
                eval_repeat = self.aco_eval_repeat

            eval_test_result = None
            eval_val_result = None
            for eval_id in range(eval_repeat):
                fold_counter = 1
                for fold_id, fold in enumerate(test_eval):
                    tr_set = fold.train_set
                    vl_set = None
                    ts_set = fold.test_set
                    # print("\nBest:[{}] [{}] Fold {} Train started!".format(eval_id, model.name, fold_counter))
                    start = time.time()
                    test_model.fold_id = fold_id
                    # gc collect
                    gc.collect()
                    fold_model = test_model.fit(tr_set, ts_set)
                    train_time = time.time() - start
                    # print("\nBest:[{}] [{}] Fold {} Validation started!".format(eval_id, model.name, fold_counter))

                    metric_avg_results = OrderedDict()
                    metric_user_results = OrderedDict()
                    start = time.time()
                    avg_results, user_results = ranking_eval(
                            fold_model,
                            self.metrics,
                            tr_set,
                            ts_set,
                            vl_set,
                            rating_threshold=fold.rating_threshold,
                            exclude_unknowns=fold.exclude_unknowns,
                            verbose=False,
                        )
                    test_time = time.time() - start
                    for i, mt in enumerate(self.metrics):
                        metric_avg_results[mt.name] = avg_results[i]
                        metric_user_results[mt.name] = user_results[i]

                    test_result = Result(fold_model.name, metric_avg_results, metric_user_results)

                    test_result.metric_avg_results["Cov. %"] =( len(np.unique(fold_model.recommended_item_set))/tr_set.num_items) * 100
                    test_result.metric_avg_results["Train (s)"] = train_time
                    test_result.metric_avg_results["Test (s)"] = test_time

                    if eval_test_result is None:
                        eval_test_result = test_result
                    else:
                        for key in test_result.metric_avg_results.keys():
                            eval_test_result.metric_avg_results[key] = (test_result.metric_avg_results.get(key, 0)
                                                                            + eval_test_result.metric_avg_results.get(key, 0))
                    fold_counter += 1

            for key in eval_test_result.metric_avg_results.keys():
                eval_test_result.metric_avg_results[key] = float(eval_test_result.metric_avg_results.get(key, 0)
                                                                 / (eval_repeat * len(test_eval)))

            test_result = eval_test_result
            val_result = None

            self.result.append(test_result)

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        output += "\n ALL FOLD AVERAGE TEST:\n...\n{}".format(self.result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)

    def _eval(fold, model, metrics, train_set, test_set, val_set):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=metrics,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=fold.rating_threshold,
            exclude_unknowns=fold.exclude_unknowns,
            verbose=fold.verbose,
        )
        for i, mt in enumerate(metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)