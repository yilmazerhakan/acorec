import torch
import numpy as np

from recommender import Recommender
from cornac.exception import ScoreException

class RP3Beta(Recommender):
    """

    Parameters
    ----------
    name: string, optional, default: 'RP3ᵝ'
        The name of the recommender model.

    bet: float, optional, default: .2
        parameter beta ∈ R
    
    alp: float, optional, default: .6
        parameter alpha ∈ R

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Paudel, B., Christoffel, F., Newell, C., & Bernstein, A. (2016). Updatable, accurate, diverse, and scalable recommendations for interactive applications. ACM Transactions on Interactive Intelligent Systems (TiiS), 7(1), 1-34.
    """

    def __init__(
            self,
            name="RP3Beta",
            bet=.2,
            alp=.6,
            normalize=True,
            trainable=True,
            verbose=True,
            seed=None,
            P=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.bet = bet
        self.alp = alp
        self.normalize = normalize
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
        
        # Pui row-normalized urm of A
        Pui = torch.nn.functional.normalize(U, p=1, dim=1)

        # Piu row-based normalize of A'
        # 'Piu is the column-normalized, "boolean" urm transposed'
        Piu = torch.transpose(U, 0, 1)

        # Taking the degree of each item to penalize top popular
        DPiu = torch.sum(Piu, 1)
        # Some rows might be zero, make sure their degree remains zero
        DPiu[DPiu==0] = 1
        degrees = torch.pow(DPiu, self.bet)
        S = torch.diag(degrees)

        # Piu row-based normalize
        Piu = torch.nn.functional.normalize(Piu, p=1, dim=1)

        # Alpha parameter
        if self.alp != 1:
            Pui = torch.pow(Pui, self.alp)
            Piu = torch.pow(Piu, self.alp)

        # Final predictions are computed as Pui * W     
        # W = (Piu * Pui)*S
        W = torch.matmul(Piu, Pui)

        if self.normalize:
            W = torch.nn.functional.normalize(W, p=1, dim=0)

        # penalize top popular items with parameters
        W = torch.matmul(W, S)

        # User Predictions
        self.P = torch.matmul(Pui, W)
      
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