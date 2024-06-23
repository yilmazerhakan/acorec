import torch
import numpy as np

from recommender import Recommender
from cornac.exception import ScoreException

class BasePopular(Recommender):
    """

    Parameters
    ----------
    name: string, optional, default: 'BasePopular'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    **
    """

    def __init__(
            self,
            name="BasePopular",
            trainable=True,
            verbose=True,
            seed=None,
            P=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.verbose = verbose
        self.seed = seed
        self.P = P
        self.use_gpu = use_gpu        

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)
        
        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )        

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

        # U rating matrix
        U = self.train_set.matrix
        U.data = np.ones(len(U.data))  # Binarize data
        U = U.A
        U = torch.tensor(U, dtype=torch.float32, device=self.device)
        nU, nM = U.shape

        self.item_counts = torch.norm(U,"fro", dim=0).data.cpu().numpy().flatten()
        
        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            return self.item_counts
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            return self.item_counts(item_idx)