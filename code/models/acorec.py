import torch
from recommender import Recommender
from cornac.exception import ScoreException

class AcoRec(Recommender):
    """AcoRec: Personalized multi-objective top-N recommendations using Continuous Ant Colony Optimization

    Parameters
    ----------
    imodel: str, default: 'gram'
        Input Similarity Model.
        Supported models: ['gram', 'cosine', 'jaccard']
    
    likelihood: str, default: 'tanh'
        name of the likelihood function by leveraging the probability distributions
        Supported torch functions:
            tanh: Tanh
            sigmoid: Sigmoid
            softmax: Softmax

    prob_norm: str, default: 'tanh'
        Name of the normalization function for the predicted probability values of ant solutions
        Supported torch functions:
            tanh: Tanh
            sigmoid: Sigmoid
            softmax: Softmax

    n_epochs: int, optional, default: 250
        The number of epochs for the β parameter search.

    archive_size: int, optional, default: 50
        The number of solutions that are stored in an archive during the optimization process

    ant_size: int, optional, default: 50
        The number of ant size is the number of β variant in each epoch

    drop_out: float, optional, default: .5
        Proportion at which observed ratings are included or selected during the validation process in each epoch

    top_n_size: int, optional, default: 100
        the number of top-ranked solutions or recommendations that are considered for validation

    long_tail: boolean, optional, default: False
        When True, long tail items will be selected for validation

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    use_gpu: boolean, optional, default: False
        If True and your system supports CUDA then training is performed on GPUs.

    References
    ----------
    * AcoRec: Personalized multi-objective top-N recommendations using Continuous Ant Colony Optimization
    """

    def __init__(
        self,
        name="AcoRec",
        imodel="gram",
        likelihood="tanh",
        prob_norm="softmax",
        n_epochs=250,
        archive_size = 50,
        ant_size = 50,
        drop_out = .5,
        top_n_size = 100,
        long_tail = False,
        rho = .05,
        lr = 1e-2,
        sig = 1,
        verbose=False,
        seed=None,
        use_gpu=True,
        device_name="cuda:0",
        trainable=True,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.imodel = imodel
        self.likelihood = likelihood
        self.prob_norm = prob_norm
        self.n_epochs = n_epochs
        self.archive_size = archive_size
        self.ant_size = ant_size
        self.drop_out = drop_out
        self.top_n_size = top_n_size
        self.long_tail = long_tail
        self.rho = rho
        self.lr = lr
        self.sig = sig
        self.seed = seed
        self.use_gpu = use_gpu
        self.device_name = device_name
        self.model_name = name + " _ " + imodel

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

        from models.acorec_learn import AcoRec, learn

        torch.set_printoptions(precision=5)

        self.device = (
            torch.device(self.device_name)
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        self.name =  self.model_name

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "recaco") or hasattr(self, "batch_cv"):
                self.acorec = AcoRec(
                    self.train_set,
                    self.imodel,
                    self.likelihood,
                    self.long_tail,
                    self.drop_out,
                    self.prob_norm,
                    self.top_n_size,
                    self.ant_size,
                    self.device_name
                ).to(self.device)

            learn(
                self.acorec,
                self.val_set,
                batch_size = 1,
                n_epochs = self.n_epochs,
                archive_size = self.archive_size,
                rho = self.rho,
                lr = self.lr,
                sig = self.sig,
                verbose=self.verbose
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

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
            return self.acorec.P[user_idx,:].data.cpu().numpy().flatten()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            return self.acorec.P[user_idx,item_idx].data.cpu().numpy().flatten()