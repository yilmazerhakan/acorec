import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import random
from torch.optim.adam import Adam
import gc

torch.set_default_dtype(torch.float32)

LL = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(dim=1),
    "relu": nn.ReLU(),
}

EPS = 1e-8

GaussianNLLLoss = torch.nn.GaussianNLLLoss(full=True,eps=EPS,reduction='none')

class AcoRec(torch.nn.Module):
    def __init__(self, train_set, imodel, likelihood, long_tail, drop_out, prob_norm, top_n_size, ant_size, device_name):
        super(AcoRec, self).__init__()

        self.device = device_name if torch.cuda.is_available() else "cpu"

        self.imodel = imodel
        self.likelihood = likelihood
        self.prob_norm = prob_norm
        self.long_tail = long_tail
        self.drop_out = drop_out
        self.short_items = None
        self.long_tail_items = None
        self.top_n_size = top_n_size
        self.ant_size = ant_size

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # nDCG (normalized Discounted Cumulative Gain) Template
        eval_temp = 1. / np.log2(np.arange(2, self.top_n_size + 2)) 
        
        self.metric_temp = torch.from_numpy(eval_temp).to(self.device)

        # Binarize ratings
        self.U = torch.tensor(train_set.matrix.A, dtype=torch.bool, device=self.device)

        # Posterior Predictions
        self.P = torch.empty_like(self.U, dtype=torch.float32)

        # Find Item Frequencies
        if self.long_tail:
            self.short_long_tail_items()

        # Selected Input Model
        self.item_input_model()

        # Prior Predictions
        self.ll_fn = LL.get(self.likelihood, None)
        if self.ll_fn is None:
            raise ValueError("Supported ll_fn: {}".format(LL.keys()))

    def clicked_items(self, x):
        # validation items
        if self.long_tail:
            val_items = x.clone().flatten()
            val_items.index_fill_(0, self.short_items, 0)
            clicked_items = torch.nonzero(val_items).flatten()
        else:
            clicked_items = torch.nonzero(x)[:,1].flatten()
        return clicked_items

    def forward(self, x):
        clicked_items = self.clicked_items(x)
        prior = torch.mm(x, self.G)
        prior = self.normalize_tensor(prior)
        ll = F.binary_cross_entropy(self.ll_fn(prior), x, x, reduction='none')
        x = torch.mm(ll, self.G)
        return x, clicked_items

    def item_input_model(self):
        nU, nM = self.U.shape
        # Cosine Similarity
        if self.imodel=="cosine":
            # Normalization
            U_norm = F.normalize(self.U.float(), dim=0)
            G = torch.mm(U_norm.t(), U_norm)
        # Jaccard Similarity
        elif self.imodel=="jaccard":
            G = torch.matmul(self.U.t().float(), self.U.float())
            G = torch.div(G , torch.sum(self.U.t().float(), dim=1).repeat(nM, 1).t() + torch.sum(self.U.t().float(), dim=1).repeat(nM, 1) - G)
            G[G<0] = 0
        # Gram Matrix
        else:
            G = torch.matmul(self.U.t().float(), self.U.float())

        # L-2 norm of G
        self.l2_scale = torch.norm(G, p="fro", dim=1)
        self.log_l2_scale = torch.log(self.l2_scale)

        self.G = G
        del G

        return self

    def short_long_tail_items(self):
        # Find Item Frequencies
        chunk_size = 2000
        num_columns = self.U.shape[1]

        chunks = [self.U[:, i:i + chunk_size] for i in range(0, num_columns, chunk_size)]
        idf_all_chunks = [torch.norm(chunk.float(), p=1, dim=0) for chunk in chunks]

        idf_all = torch.cat(idf_all_chunks, dim=0)

        idf_vy, idf_iy = idf_all.sort(descending=True, stable=True)

        pop_items_count = sum(torch.sum(chunk > 0).item() for chunk in torch.chunk(self.U, chunks=chunk_size)) * (1/3)

        total = 0
        for i in range(idf_vy.shape[0]):
            total += idf_vy[i]
            if total >= pop_items_count:
                break

        self.short_items =  idf_iy[:i]
        self.long_tail_items = idf_iy[:-i]

        return self

    def ant_probs(self, vdy, hit_items):
        vdy = self.normalize_tensor(vdy)
        prob_norm = LL.get(self.prob_norm, None)
        vdy = prob_norm(vdy)
        if self.drop_out > 0:
            dropout_mask = F.dropout(torch.ones_like(vdy), self.drop_out, self.training, False)
            # num_items = hit_items.size(0)
            # num_select = int(num_items * self.drop_out)
            # selected_indices = hit_items[torch.randperm(num_items)[:num_select]]
            # dropout_mask.index_fill_(1, selected_indices, torch.max(dropout_mask))
            vdy = vdy * dropout_mask
            # Normalize the vdy after dropout if needed
            vdy /= torch.sum(vdy, dim=1, keepdim=True)

        return vdy

    def normalize_tensor(self, vdy):
        min_outs, _ = torch.min(vdy, dim=1)
        max_outs, _ = torch.max(vdy, dim=1)
        vdy_normalized = (vdy - min_outs.unsqueeze(-1)) / (max_outs.unsqueeze(-1) - min_outs.unsqueeze(-1) + EPS)

        return vdy_normalized
    
    def run_epoch(self, mu, sigma, likelihoods, clicked_items):

        # Change seed in each epoch
        torch.manual_seed(random.choice(range(10000)))

        # Create Gaussian distributions with μ , σ²
        sampled_betas = torch.normal(mean=mu.item(), std=sigma.item(), size=(self.ant_size, 1)).to(self.device)

        # Probabilities for each ant with sampled beta values
        probabilities = likelihoods * torch.exp(sampled_betas * self.log_l2_scale)
        probabilities[~torch.isfinite(probabilities)] = 0
        # Map probabilities into the same range using activation functions
        probabilities = self.ant_probs(probabilities, clicked_items)

        # Trim top_k probs for each ant find hit items in these lists
        top_k_probs, idy = torch.topk(probabilities, k=self.top_n_size, dim=1)
        hits = torch.isin(idy, clicked_items).float()

        # Calculate the nDCG for ant predictions
        dcg = hits * self.metric_temp
        len_clicked = len(clicked_items)
        idcg = torch.sum(self.metric_temp[:len_clicked])
        costs = dcg / idcg

        ant_fitness = torch.sum(F.binary_cross_entropy(top_k_probs, hits, costs, reduction='none'), dim=1)

        # Select best ants with fitness score
        best_ant_scores, best_ant_ids = torch.topk(ant_fitness, k=ant_fitness.shape[0], dim=0)

        # Keep best solutions in solution archive
        selected_solutions = torch.column_stack((torch.flatten(sampled_betas[best_ant_ids]), best_ant_scores))

        return selected_solutions

    def loss(self, solution_archive, mu, sigma):
        norm_s = nn.ReLU()
        q1 = norm_s(solution_archive[:, 1].unsqueeze(1) + EPS ).squeeze(1)
        # Gaussian Negative Log Likelihood Loss
        losses = GaussianNLLLoss(solution_archive[:, 0], mu, sigma)
        ll = torch.exp(-losses) * q1
        # Take the negative log to give the GMM negative log-likelihood loss
        nll_loss = -torch.log(torch.mean(ll))

        return  nll_loss

def learn(
    acorec,
    val_set,
    n_epochs,
    batch_size,
    archive_size,
    rho,
    lr,
    sig,
    verbose,
):

    for batch_id, u_ids in tqdm(enumerate(
            val_set.user_iter(batch_size, shuffle=False)
        ), disable=not verbose):

        # construct test user vectors
        likelihoods, clicked_items = acorec(acorec.U[u_ids,:].float())
        
        # mu and sigma with initial values to be used in the subsequent optimization process
        mu = torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=acorec.device)
        sigma = torch.tensor([sig], dtype=torch.float32, requires_grad=True, device=acorec.device)

        # mu and sigma will be updated during the optimization process.
        optimizer = Adam([
            {'params': mu, 'lr': lr},
            {'params': sigma, 'lr': lr},
        ], amsgrad=False) 

        for n_epoch in range(1,n_epochs+1):

            if torch.isnan(sigma) or torch.any(sigma < 0):
                mu = torch.mean(solution_archive[:,0])
                break

            # best solutions in solution archive
            selected_solutions = acorec.run_epoch(mu, sigma, likelihoods, clicked_items)

            if n_epoch==1:
                solution_archive = selected_solutions
            else:
                solution_archive = torch.cat((solution_archive, selected_solutions), 0)

            # Take the negative log to give the GMM negative log-likelihood loss
            loss = acorec.loss(solution_archive, mu, sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sigma.data = torch.clamp(sigma, min=EPS)
            if sigma <= EPS:
                break
               
            # Evaporation
            solution_archive = solution_archive[torch.topk(solution_archive[:,1], k=archive_size)[1], :]
            solution_archive[:, 1] *= (1 - rho)

        # predictions for user
        acorec.P[u_ids,:] = likelihoods * torch.pow(acorec.l2_scale, mu.item())

    return acorec