import os
from typing import Tuple
from cornac.data.reader import Reader
from eval_methods.base_method import BaseMethod
from cornac.metrics import Recall,NDCG

current_directory = os.path.dirname(__file__)

DS_FOLDER = os.path.join(current_directory, "../datasets/")

def eval_datasets(dss, ds) -> Tuple:
    SEED = 12345
    mfold = 5
    long_tail = False
    if ds=="lt":
        long_tail = True

    # Instantiate evaluation measures
    rec_010 = Recall(k=10)
    rec_020 = Recall(k=20)
    rec_050 = Recall(k=50)
    ndcg_010 = NDCG(k=10)
    ndcg_020 = NDCG(k=20)
    ndcg_100 = NDCG(k=100)
    
    reader = Reader()

    eval_metric = ndcg_010
    result_metrics = [rec_010, rec_020, rec_050, ndcg_010, ndcg_020, ndcg_100]

    eval_methods = []
    for fold in range(1,mfold+1):
        # Prepare datasets for eval.
        train_data = reader.read(DS_FOLDER+"{}_{}_{}.tr".format(ds,dss,fold))
        test_data = reader.read(DS_FOLDER+"{}_{}_{}.ts".format(ds,dss,fold))
        print("------------------------")
        print("Fold : {}".format(fold))
        # Instantiate a Base evaluation method using the provided train and test sets
        eval_methods.append ( BaseMethod.from_splits(
            train_data=train_data, test_data=test_data, val_data=test_data, exclude_unknowns=True, verbose=True
        ) )

    return eval_methods, SEED, dss, ds, long_tail, result_metrics, eval_metric

def recommendation_dataset() -> Tuple:
    SEED = 12345
    # dss = "netflix"
    dss = "ml1m45"
    # dss = "pinterest"
    ds = "cs"
    mfold = 1
    long_tail = False
    if ds=="lt":
        long_tail = True

    # Instantiate evaluation measures
    rec_010 = Recall_d(k=10)
    rec_020 = Recall_d(k=20)
    ndcg_010 = NDCG_d(k=10)
    ndcg_020 = NDCG_d(k=20)

    reader = Reader()

    eval_metric = ndcg_010
    result_metrics = [rec_010, rec_020, ndcg_010, ndcg_020]

    eval_methods = []
    for fold in range(1,mfold+1):
        # Prepare datasets for eval.
        train_data = reader.read(DS_FOLDER+"{}_{}_{}.tr".format(ds,dss,fold))
        test_data = reader.read(DS_FOLDER+"{}_{}_{}.ts".format(ds,dss,fold))
        print("---")
        print("Fold : {}".format(fold))
        # Instantiate a Base evaluation method using the provided train and test sets
        eval_methods.append ( BaseMethod.from_splits(
            train_data=train_data, test_data=test_data, val_data=test_data, exclude_unknowns=True, verbose=True
        ) )

    return eval_methods, SEED, dss, ds, long_tail, result_metrics, eval_metric
