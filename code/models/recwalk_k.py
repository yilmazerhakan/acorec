import torch
import numpy as np
from recommender import Recommender
from cornac.exception import ScoreException
from SLIM import SLIM, SLIMatrix
import torch.nn.functional as F

class Recwalk_K(Recommender):
    """

    Parameters
    ----------
    name: string, optional, default: 'Recwalkᴷ⁻ˢᵗᵉᵖ'
        The name of the recommender model.

    imodel: str, default: 'cosine'
        Input Similarity Model.
        Supported models: ['cosine', 'slim']
        
    alpha: float, optional, default: .005
        alpha parameter α ∈ R
    
    k: int, optional, default: 7
        parameter k ∈ R
    
    l1_ratio: float, optional, default: .01
        parameter l1_ratio ∈ R+
        It will be used when input model selected as SLIM
    
    l2_ratio: float, optional, default: .1
        parameter l2_ratio ∈ R+
        It will be used when input model selected as SLIM

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Athanasios N. Nikolakopoulos and George Karypis. 2019. RecWalk: Nearly Uncoupled Random Walks for Top-N Recommendation. 
    In The Twelfth ACM International Conference on Web Search and Data Mining (WSDM ’19), February 11–15, 2019, Melbourne, VIC, Australia. 
    ACM, New York, NY, USA, 9 pages
    """

    def __init__(
            self,
            name="Recwalkᴷ⁻ˢᵗᵉᵖ",
            imodel="slim",
            alpha=.005,
            k=7,
            l1_ratio=.01,
            l2_ratio=.1,
            trainable=True,
            verbose=True,
            seed=None,
            P=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.imodel = imodel
        self.alpha = alpha
        self.k = k
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.verbose = verbose
        self.seed = seed
        self.P = P
        self.use_gpu = use_gpu
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
        
        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if torch.cuda.is_available():
            self.P = None
            torch.cuda.empty_cache()
        else:
            self.P = []

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
        
        # U rating matrix
        U = self.train_set.matrix
        U.data = np.ones(len(U.data))  # Binarize data
        U = U.A
        U = torch.tensor(U, dtype=torch.bool, device=self.device)
        self.nU = U.shape[0]
        self.nM = U.shape[1]
        chunk_size = round(self.nU/10)

        # cosine similarity
        if self.imodel=="cosine":
            U_norm = F.normalize(U.float(), dim=0)
            G = torch.mm(U_norm.t(), U_norm)
        # slim similarity
        elif self.imodel=="slim":

            trainmat = SLIMatrix(self.train_set.matrix)

            params = {
                'simtype':'cos',
                'nnbrs':int(self.nM * 0.1),
                'l1r':self.l1_ratio,
                'l2r':self.l2_ratio,
                }
            
            model = SLIM()
            model.train(params, trainmat)
            W = model.to_csr()
            W = W.tocsc()
            W = W.A

            diag_indices = np.diag_indices(W.shape[0])
            W[diag_indices] = 0.0
            W[W<0]=0
            G = torch.tensor(W, dtype=torch.float32, device=self.device)

        else:
            chunks_U = torch.chunk(self.U, chunks=self.nM // chunk_size, dim=0)
            G = sum([torch.matmul(chunk.t().float(), chunk.float()) for chunk in chunks_U])
            del chunks_U
        
        # # Build RecWalk Model
        self.nU = U.shape[0]
        self.nM = U.shape[1]
        
        alpha = torch.tensor(self.alpha)
        alpha_ = torch.tensor(1-self.alpha)

        # Build RecWalk Model
        row_sums = torch.sum(G, 1)
        dmax = torch.max(row_sums)
        A_temp = ((1 / dmax) * G)
        tdiag = torch.diag(torch.sum(A_temp, 1))
        Mii = (torch.eye(self.nM).to(self.device) - tdiag) + A_temp

        # Hui = RowStochastic(TrainSet)
        row_sums = torch.sum(U.float(), 1)
        row_sums[row_sums==0]=1
        Hui = torch.matmul(torch.diag(1 / row_sums) , U.float()) * alpha

        # Hiu = RowStochastic(TrainSet')
        row_sums = torch.sum(U.t().float(), 1)
        row_sums[row_sums==0]=1
        Hiu = torch.matmul(torch.diag(1 / row_sums), U.t().float()) * alpha

        # M = vcat(hcat(Muu,spzeros(n,m)), hcat(spzeros(m,n),Mii))
        m1 = torch.cat([torch.eye(self.nU).to(self.device), torch.zeros(self.nU,self.nM).to(self.device)], 1) * alpha_
        m2 = torch.cat([torch.zeros(self.nM,self.nU).to(self.device), Mii], 1) * alpha_
        M = torch.cat([m1, m2],0)
        Mii=None

        # H = vcat(hcat(spzeros(n,n),Hui), hcat(Hiu,spzeros(m,m)))
        devi = torch.device("cpu")
        h1 = torch.cat([torch.zeros(self.nU,self.nU).to(devi), Hui.to(devi)], 1).to(devi)
        h2 = torch.cat([Hiu.to(devi), torch.zeros(self.nM,self.nM).to(devi)], 1).to(devi)
        H = torch.cat([h1, h2],0).to(devi)
        M = M.to(devi)
        torch.cuda.empty_cache()

        # Rec-walk
        self.P = H + M

        del M,H,m1,m2,Hui,Hiu,Mii

        # K-Step
        self.P = torch.matrix_power(self.P, self.k - 1)
      
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

            ru = torch.reshape(self.P[user_idx,:], (1, self.nU + self.nM))
            known_item_scores = ru[:,self.nU:]
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
            user_pred = user_pred.data.cpu().numpy().flatten()
            return user_pred