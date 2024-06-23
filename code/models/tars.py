import torch
import numpy as np

from recommender import Recommender
from cornac.exception import ScoreException

class TARS(Recommender):
    """ Trust-based recommender system using ant colony for trust computation.

    Parameters
    ----------
    name: string, optional, default: 'TARS'
        The name of the recommender model.
        
    conf_value: float, optional, default: .1
        confidence value

    ant_size: int, optional, default: 100
        ant_size

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Bedi, P., & Sharma, R. (2012). Trust-based recommender system using ant colony for trust computa-tion. Expert Systems with Applications, 39(1), 1183-1190.
    """

    def __init__(
            self,
            name="TARS",
            conf_value=.2,
            ant_size=100,
            modal_data=None,
            trainable=True,
            verbose=True,
            seed=None,
            U=None,
            ITG=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.conf_value = conf_value
        self.ant_size = ant_size
        self.modal_data = modal_data
        self.verbose = verbose
        self.seed = seed
        self.U = U
        self.ITG = ITG
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

            #if not hasattr(self, "vae"):
            #    data_dim = train_set.matrix.shape[1]
        
        # U rating matrix
        U = self.train_set.matrix
        U.data = np.ones(len(U.data))  # Binarize data
        U = U.A
        U = torch.tensor(U, dtype=torch.float32, device=self.device)

        # Confdence Model
        Iu = U.sum(dim=1)
        Iu[Iu <= 0] = 1
        Confidence = torch.matmul(U, U.t()) / Iu.unsqueeze(1)
        Confidence[torch.isnan(Confidence)] = 0

        # Pearson Similarity
        Sim = torch.corrcoef(U)

        # create initial ITG
        all_both = (Confidence != 0) & (Sim != 0)
        ITG = ((2 * Confidence * Sim) / (Sim + Confidence)) * all_both.float()
        ITG[torch.isnan(ITG)] = 0
        Confidence_only = (Confidence != 0).float() * self.conf_value * (Sim == 0).float()
        ITG += Confidence_only
        ITG[torch.isnan(ITG)] = 0

        ## Trust-Based Ant Recommender System (TARS)
        self.U = U
        self.ITG = ITG

        del Confidence_only,ITG
      
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
            
            observations = self.U[user_idx,:]
            level = 1
            active_user = user_idx

            nU,nM = self.U.shape

            # traced path from one level to another
            p = []
            p.append(active_user)
            notAllowed = [user_idx]  # in order to avoid visiting same nodes again

            time_to_live=10
            trusted_user_size=self.ant_size

            ITG = self.ITG

            while time_to_live > 0:
                # find unknown items at current time which user has rated later
                unknowns = ~observations.bool()
                empty_items = torch.sum(unknowns).item()

                if empty_items > 0:
                    probabilities = ITG[active_user, :]

                    probabilities[active_user] = 0
                    probabilities[notAllowed] = 0

                    sortedValues, indices = torch.sort(probabilities, descending=True)

                    if indices.numel() > trusted_user_size:
                        TF_S = indices[:trusted_user_size]
                    else:
                        TF_S = indices

                    if TF_S.numel() == 0:
                        # if this user doesn't have any trustworthy friend
                        # then there is no need for the next level
                        break

                    neigh_ratings = torch.zeros(len(TF_S), nM)
                    for neigh in range(len(TF_S)):
                        neigh_ratings[neigh, :] = self.U[TF_S[neigh], :] * ITG[active_user, TF_S[neigh]]

                    mean_ratings = torch.floor_divide(torch.sum(neigh_ratings, dim=0), torch.sum(neigh_ratings.bool(), dim=0))
                    observations += mean_ratings * (1 / level) * unknowns.float()
                    observations[torch.isnan(observations)] = 0

                    # Update pheromone in this level
                    # Trust deposition for neighbors who involved in recommendations
                    for neigh in range(len(TF_S)):
                        T_traced = unknowns.float() * self.U[neigh, :]
                        pie = 1
                        path = p.copy()
                        path.append(TF_S[neigh])
                        for j in range(len(path) - 1):
                            pie = pie * ITG[path[j], path[j + 1]]
                        deltaQ = (pie / level) * (torch.sum(T_traced.bool()) / empty_items)
                        ITG[user_idx, TF_S[neigh]] = (1 - 0.01) * ITG[user_idx, TF_S[neigh]] + deltaQ

                active_user = TF_S[0]
                level += 1
                time_to_live -= 1
                p.append(active_user)
                notAllowed = list(set(notAllowed).union(TF_S))

            probabilities = ITG[user_idx, :]
            _, indices = torch.sort(probabilities, descending=True)
            if indices.numel() > trusted_user_size:
                TF_S = indices[:trusted_user_size]
            else:
                TF_S = indices
            
            known_item_scores = ITG[user_idx, TF_S] @ self.U[TF_S, :]
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