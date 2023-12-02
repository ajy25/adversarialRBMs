import torch
import torch.nn as nn
import numpy as np
import json
from .util import (
    partition_into_batches, k_nearest_neighbors, inverse_distance_sum
)
torch.set_default_dtype(torch.float32)

class ConditionalRBM(nn.Module):
    """Conditional Restricted Boltzmann Machine"""

    def __init__(self, n_vis: int, n_hid: int, n_cond: int, 
                 var: float = None):
        """
        Constructs a Gaussian-Bernoulli Conditional Restricted Boltzmann 
        Machine.

        @args
        - n_vis: int << number of visible nodes
        - n_hid: int << number of hidden nodes
        - n_cond: int << number of conditional nodes
        - var: float | None << set variance for each visible node;
            if None, we learn the variance on each visible node
        """
        super().__init__()
        self.rng = torch.Generator()
        self.reset_seed(42)
        self.adversary_memory = None
        self.reset_hyperparameters(n_vis=n_vis, n_hid=n_hid, n_cond=n_cond,
                                   var=var)

    def metadata(self):
        """
        Returns the metadata of the RBM object.

        @returns
        - dict
        """
        metadata = {
            "n_vis": self.n_vis,
            "n_hid": self.n_hid,
            "n_cond": self.n_cond,
            "var": self.var
        }
        return metadata
    
    def reset_hyperparameters(self, n_vis: int = None, n_hid: int = None, 
                              n_cond: int = None, var: float = None):
        if n_vis is not None:
            self.n_vis = n_vis
        if n_hid is not None:
            self.n_hid = n_hid
        if n_cond is not None:
            self.n_cond = n_cond
        self.var = var
        self.W = nn.Parameter(torch.Tensor(self.n_vis, self.n_hid))
        self.mu = nn.Linear(self.n_cond, self.n_vis, bias=False)
        self.b = nn.Linear(self.n_cond, self.n_hid, bias=False)
        if self.var is None:
            self.log_var = nn.Parameter(torch.Tensor(self.n_vis))
        else:
            self.log_var = torch.ones((self.n_vis)) * np.log(var)
        self.reset_parameters()

    def reset_parameters(self, seed: int = 42):
        """
        Resets trainable parameters of the Gaussian-Bernoulli RBM.
        """
        torch.manual_seed(seed)
        nn.init.xavier_normal_(self.W)
        self.mu.reset_parameters()
        self.b.reset_parameters()
        if self.var is None:
            nn.init.constant_(self.log_var, 0)

    def reset_seed(self, seed: int):
        """
        Resets the rng seed to enforce reproducibility after training.

        @args
        - seed: int
        """
        self.rng.manual_seed(seed)

    def _variance(self):
        """
        Returns the variance; we only attempt to train the log variance.

        @args
        - c: torch.Tensor ~ (batch_size, n_cond)

        @returns
        - torch.Tensor | float
        """
        return torch.exp(self.log_var)

    def _energy(self, v: torch.Tensor, c: torch.Tensor, h: torch.Tensor):
        """
        Equation (1, 2) in https://arxiv.org/pdf/2210.10318.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - c: torch.Tensor ~ (batch_size, n_cond)
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - float
        """
        var = self._variance()
        pos = torch.sum(0.5 * (v - self.mu.forward(c)) ** 2 / var, dim=1)
        neg = torch.sum(((v / var) @ (self.W)) * h, dim=1) + \
            torch.sum(h * self.b.forward(c), dim=1)
        return (pos - neg) / v.shape[0]
    
    @torch.no_grad()
    def reconstruct(self, v: np.ndarray, c: np.ndarray, 
                    clamp: np.ndarray = None, random_init = True, 
                    n_gibbs: int = 1, add_noise=True):
        """
        Reconstructs the visible units.

        @args
        - v: np.array ~ (batch_size, n_vis)
        - c: np.array ~ (batch_size, n_cond)
        - clamp: boolean np.array ~ (batch_size, n_vis) << will only
            reconstruct elements marked False in the boolean mask
        - random_init: bool << if True, reset v via torch.randn
        - n_gibbs: int

        @returns
        - np.array ~ (batch_size, n_vis)
        """
        c = torch.Tensor(c)
        if random_init:
            v = torch.randn(v.shape, generator=self.rng).requires_grad_(False)
        else:
            v = torch.Tensor(v).requires_grad_(False)
        if clamp is not None:
            v_sample, _ = self._block_gibbs_sample(c, v=v, n_gibbs=n_gibbs,
                                                    clamp=torch.Tensor(clamp),
                                                    add_noise=add_noise)
        else:
            v_sample, _ = self._block_gibbs_sample(c, v=v, n_gibbs=n_gibbs,
                                                    add_noise=add_noise)
        return v_sample.numpy()
    
    @torch.no_grad()
    def _block_gibbs_sample(self, c: torch.Tensor, v: torch.Tensor = None, 
                            h: torch.Tensor = None, clamp: torch.Tensor = None,
                            n_gibbs = 1, add_noise = False):
        """
        Familiar block Gibbs sampling method of visible and hidden units.

        @args
        - c: torch.Tensor ~ (batch_size, n_cond)
        - v: torch.Tensor ~ (batch_size, n_vis) | None << if None and h is
            not None, we begin the sampling process with the hidden units
        - h: torch.Tensor ~ (batch_size, n_hid) | None << if None and v is
            not None, we begin the sampling process with the visible units
        - clamp: torch.Tensor ~ (batch_size, n_vis) | None << will only
            reconstruct elements marked False in the boolean mask
        - n_gibbs: int << number of Gibbs sampling steps
        - add_noise: bool << adds noise to the visible units

        @returns
        - torch.Tensor ~ (batch_size, n_vis)
        - torch.Tensor ~ (batch_size, n_hid)
        """
        std = self._variance().sqrt()
        if clamp is not None:
            clamp = clamp.bool()
        if v is None and h is None:
            v_sample = torch.randn(size=(1, self.n_vis), generator=self.rng)
        elif v is None:
            v_sample = self._prob_v_given_h(h, c)
        else:
            v_sample = v.clone()
        m = v_sample.shape[0]
        h_sample = torch.bernoulli(self._prob_h_given_v(v_sample, c),
                                    generator=self.rng)
        for _ in range(n_gibbs):
            if clamp is not None:
                old_v_sample = v_sample.clone()
            v_sample = self._prob_v_given_h(h_sample, c)
            if add_noise:
                v_sample += torch.randn(size=(m, self.n_vis),
                                generator=self.rng) * std
            if clamp is not None:
                v_sample[clamp] = old_v_sample[clamp]
            h_sample = torch.bernoulli(self._prob_h_given_v(v_sample, c),
                                        generator=self.rng)
        return v_sample, h_sample
    
    @torch.no_grad()
    def _prob_h_given_v(self, v: torch.Tensor, c: torch.Tensor):
        """
        Computes sigmoid activation for p(h=1|v) according to equation (3) in
        https://arxiv.org/pdf/2210.10318; in other words, computes the
        parameters for hidden Bernoulli random variables given visible units.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - c: torch.Tensor ~ (batch_size, n_cond)

        @returns
        - torch.Tensor ~ (batch_size, n_hid)
        """
        return torch.sigmoid((v / self._variance()) @ self.W + self.b.forward(c))

    @torch.no_grad()
    def _prob_v_given_h(self, h: torch.Tensor, c: torch.Tensor):
        """
        Computes mu for p(v|h) according to equation (4) in
        https://arxiv.org/pdf/2210.10318.

        @args
        - h: torch.Tensor ~ (batch_size, n_hid)
        - c: torch.Tensor ~ (batch_size, n_cond)

        @returns
        - torch.Tensor ~ (batch_size, n_vis)
        """
        return h @ self.W.t() + self.mu.forward(c)
    
    @torch.no_grad()
    def _reconstruction_MSE(self, v: torch.Tensor, c: torch.Tensor):
        """
        Computes the MSE loss of a reconstructed visible unit and an input
        visible unit.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - c: torch.Tensor ~ (batch_size, n_cond)

        @ returns
        - torch.Tensor
        """
        prob_h = self._prob_h_given_v(v, c)
        v_bar = self._prob_v_given_h(prob_h, c)
        return torch.mean((v_bar - v) ** 2)

    def cd_loss(self, v: np.ndarray, c: np.ndarray, 
                n_gibbs: int = 1, gamma: float = 1):
        """
        Computes the contrastive divergence loss with which parameters may be
        updated via an optimizer. Follows Algorithm 3 of
        https://arxiv.org/pdf/2210.10318. Contrastive divergence loss 
        may be combined with an adversarial loss as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - c: np.array ~ (batch_size, n_cond)
        - n_gibbs: int
        - gamma: float << weight of the non-adversarial loss component

        @returns
        - torch.Tensor ~ (1) << contrastive divergence loss
        """
        v_data: torch.Tensor = torch.Tensor(v)
        c: torch.Tensor = torch.Tensor(c)
        _, h_data = self._block_gibbs_sample(c, v_data, n_gibbs=0)
        v_model, h_model = self._block_gibbs_sample(c, v_data, n_gibbs=n_gibbs)
        L = self._energy(v_data, c, h_data) - self._energy(v_model, c, h_model)
        A = self._weighted_neighbors_critic(h_model)
        if A is None:
            A = 0
        else:
            A = torch.mean(A)
        return gamma * L.mean() + (1 - gamma) * A
    
    @torch.no_grad()
    def _weighted_neighbors_critic(self, h_sample: torch.Tensor, k=2):
        """
        Weightest nearest neighbors linear critic, as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - h_sample: torch.Tensor ~ (batch_size, n_hid)
        - k: int << the k in k-nearest neighbors

        @returns
        - torch.Tensor ~ (batch_size)
        """
        if self.adversary_memory is None:
            return None
        batch_size = h_sample.shape[0]
        if batch_size < k:
            return None
        ind, distances = k_nearest_neighbors(self.adversary_memory,
                                             h_sample.numpy(), k, 
                                             'hamming')
        ind_from_data = ind.copy()
        ind_from_data[:, :batch_size] = False
        return torch.Tensor(2 * inverse_distance_sum(distances, ind_from_data) \
                            / inverse_distance_sum(distances, ind) - 1)
    
    @torch.no_grad()
    def _update_adversary_memory(self, v: np.ndarray, c: np.ndarray, 
                                 n_gibbs = 10):
        """
        Re-caches a minibatch of visible units drawn from the data and visible
        units drawn from the model, as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - c: np.array ~ (batch_size, n_cond)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v_data = torch.Tensor(v)
        c = torch.Tensor(c)
        _, h_data = self._block_gibbs_sample(c, v_data, n_gibbs=0)
        _, h_model = self._block_gibbs_sample(c, v_data, n_gibbs=n_gibbs)
        self.adversary_memory = np.vstack((h_model.numpy(), h_data.numpy()))
    
    def fit_autograd(self, X: np.ndarray, y: np.ndarray, n_gibbs: int = 1,
            lr: float = 0.1, n_epochs: int = 1, batch_size: int = 1,
            gamma: float = 1.0, gamma_delay: int = 10, 
            fail_tol: int = None, rng_seed: int = 0, 
            verbose_interval: int = None, reduce_lr_on_plateau = False, 
            checkpoint_path = None):
        """
        Built-in, simple train method that relies on Torch autograd 
        for computation of gradients. Robust to NaNs in X. Conditional values
        in y, so model learns P(X | y). 
        """
        self.reset_seed(rng_seed)
        contains_missing = np.any(np.isnan(X))
        if fail_tol is None:
            fail_tol = n_epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        recon_loss_history = 1000
        fail_count = 0
        for epoch in range(1, n_epochs + 1):
            if fail_count >= fail_tol:
                break
            reconstruction_loss = 0
            self.train()
            batched_train_data, _ = partition_into_batches([X, y], \
                batch_size, rng_seed + epoch)
            for batch in batched_train_data:
                if contains_missing:
                    missing_mask = np.isnan(batch[0])
                    clamp = np.logical_not(missing_mask)
                    batch[0][missing_mask] = 0
                    batch[0] = self.reconstruct(v=batch[0], c=batch[1], 
                                                clamp=clamp, random_init=False, 
                                                n_gibbs=n_gibbs)
                if epoch >= gamma_delay and gamma < 1:
                    self._update_adversary_memory(batch[0], batch[1],
                                                    n_gibbs=10)
                batch_train_recon = \
                    self._reconstruction_MSE(torch.Tensor(batch[0]), 
                                             torch.Tensor(batch[1]))
                optimizer.zero_grad()
                if epoch >= gamma_delay and gamma < 1:
                    loss = self.cd_loss(batch[0], batch[1], n_gibbs, gamma)
                else:
                    loss = self.cd_loss(batch[0], batch[1], n_gibbs)
                loss.backward()
                optimizer.step()
                reconstruction_loss += batch_train_recon
            reconstruction_loss /= len(batched_train_data)
            if reduce_lr_on_plateau:
                scheduler.step(reconstruction_loss)
            reconstruction_loss = np.round(reconstruction_loss.numpy(), 3)
            if reconstruction_loss > recon_loss_history:
                fail_count += 1
            else:
                fail_count = 0
            recon_loss_history = reconstruction_loss
            if verbose_interval is not None:
                if epoch % verbose_interval == 0:
                    message = f"\repoch: {str(epoch).zfill(len(str(n_epochs)))}"
                    message += f" of {n_epochs} |"
                    message += f" recon_loss: {reconstruction_loss}"
                    print(message, end="\n")
        if checkpoint_path is not None:
            metadata_path = ".".join(checkpoint_path.split(".")[:-1]) + \
                ".json"
            torch.save(self.state_dict(), checkpoint_path)
            with open(metadata_path, "w") as json_file:
                json.dump(self.metadata(), json_file)

def load(checkpoint_path: str, metadata_path: str = None) -> ConditionalRBM:
    """
    Given a checkpoint path and optionally a metadata path, construct a
    ConditionalRBM

    @args
    - checkpoint_path: str
    - metadata_path: str | None << if None, infers path from checkpoint_path

    @returns
    - ConditionalRBM
    """
    if metadata_path is None:
        metadata_path = ".".join(checkpoint_path.split(".")[:-1]) + ".json"
    with open(metadata_path, "r") as json_file:
        metadata = json.load(json_file)
    model = ConditionalRBM(
        metadata["n_vis"], metadata["n_hid"], metadata["n_cond"], 
        metadata["var"]
    )
    model.load_state_dict(torch.load(checkpoint_path))
    return model

