import torch
import numpy as np

from recommender import Recommender
from cornac.exception import ScoreException

class UserKNN(Recommender):
    """

    Parameters
    ----------
    name: string, optional, default: 'UserKNN'
        The name of the recommender model.

    k: float, optional, default: 100
        knn parameter k ∈ R

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * **
    """

    def __init__(
            self,
            name="UserKNN",
            k=100,
            trainable=True,
            verbose=True,
            seed=None,
            P=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.verbose = verbose
        self.seed = seed
        self.k = k
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
        
        # Compute mean and std for each user (row)
        mean = U.mean(dim=1, keepdim=True)
        min_std = 1e-8  # Minimum value for std to avoid division by zero
        std = U.std(dim=1, keepdim=True).clamp(min=min_std)

        # Normalize the user ratings
        U_normalized = (U - mean) / std

        # Pearson Sim
        Sim = torch.matmul(U_normalized, U_normalized.t()) / U.shape[1]

        # keep top K user sim. for every row
        # n, m = Sim.shape
        # Evaluation of validation items
        _, idy = torch.topk(Sim, k=self.k, dim=1)

        # reset Sim 
        Sim_zeroed = torch.zeros_like(Sim)
        Sim_zeroed.scatter_(1, idy, Sim.gather(1, idy))

        # User Priors
        self.P = torch.matmul(Sim_zeroed, U)
      
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
            
            known_item_scores = self.P[user_idx,:]
            known_item_scores = known_item_scores.data.cpu().numpy().flatten()

            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_pred = self.P[user_idx,item_idx]
            return user_pred