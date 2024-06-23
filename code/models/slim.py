import numpy as np
from recommender import Recommender
from cornac.exception import ScoreException
from SLIM import SLIM, SLIMatrix

class SLIMFast(Recommender):
    """Reference:

    Parameters
    ----------
    name: string, optional, default: 'EASEᴿ'
        The name of the recommender model.

    l1_ratio: float, optional, default: 1
        l1-norm regularization-parameter l1_ratio ∈ R+.

    l2_ratio: float, optional, default: 1
        l2-norm regularization-parameter l2_ratio ∈ R+.        
    
    n_epochs: int, optional, default: 100
        The number of epochs for CD.

    simtype: str, optional, default: "cos"
        Specifies the similarity function for determining the neighbors. Available options are:
        - cos     -  cosine similarity [default].
        - jac     -  extended Jacquard similarity
        - dotp    -  dot-product similarity.

    nnbrs: int, optional, default: 0
        Selects fSLIM model and specifies the number of item nearest neighbors to be used. 
        Specifying few neighbors will speed-up the model learning but it may lower the accuracy. 
        The parameter *simtype* sets the measurement of similarity. 
        This package supports three similarity measurements, Jaccard similarity (\"jac\"), Cosine similarity (\"cos\"), and inner product (\"dotp\"). 
        The default value for *simtype* is \"cos\".
        A fSLIM model can be used in the same way with a SLIM model. 
        Note that, a fSLIM model can only be trained using coordinate descent.

    posW: boolean, optional, default: False
        Remove Negative Weights

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Xia Ning et al. "SLIM: Sparse Linear Methods for Top-N Recommender Systems." in ICDM 2011.

    @online{slim,
        title = {{SLIM Library for Recommender Systems}},
        author = {Ning, Xia and Nikolakopoulos, Athanasios N. and Shui, Zeren and Sharma, Mohit and Karypis, George},
        url = {https://github.com/KarypisLab/SLIM},
        year = {2019},
    }

    Reference code:
    https://github.com/KarypisLab/SLIM    
    """

    def __init__(
            self,
            name="SLIM",
            l1_ratio=1,
            l2_ratio=1,
            n_epochs=100,
            simtype="cos",
            nnbrs=100,
            from_cache=False,
            posW=True,
            trainable=True,
            verbose=False,
            seed=None,
            W=None,
            U=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.n_epochs = n_epochs
        self.simtype = simtype
        self.nnbrs = nnbrs
        self.from_cache = from_cache
        self.posW = posW
        self.verbose = verbose
        self.seed = seed
        self.W = W
        self.U = U
        self.slim_model = None

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

        # A rating matrix
        self.U = self.train_set.matrix

        # trainmat
        R = SLIMatrix(self.U)

        # trainmat
        params = {
            'nnbrs':self.nnbrs,
            'simtype':self.simtype,
            'algo':'cd',
            'nthreads':32,
            'niters':self.n_epochs,
            'nrcmds':100,
            'l1r':self.l1_ratio,
            'l2r':self.l2_ratio
            }
        model = SLIM()
        model.train(params, R)

        W = model.to_csr()
        W = W.tocsc()
        W = W.A

        diag_indices = np.diag_indices(W.shape[0])
        W[diag_indices] = 0.0

        # if self.posW remove negative values
        if self.posW:
            W[W<0]=0

        # save W for predictions
        self.W = self.U.dot(W)

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

            known_item_scores = self.W[user_idx, :]
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.W[user_idx, item_idx]
            return user_pred