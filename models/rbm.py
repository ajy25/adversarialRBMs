import torch
import torch.nn as nn
import numpy as np
import json
import os
from .util import (
    partition_into_batches, k_nearest_neighbors, inverse_distance_sum, 
    approx_kl_div
)
torch.set_default_dtype(torch.float32)

class RBM(nn.Module):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine as described in
    https://arxiv.org/pdf/2210.10318. Added adversarial component as 
    described in https://arxiv.org/abs/1804.08682. 
    """

    def __init__(self, n_vis: int, n_hid: int, var: float = None):
        """
        Constructs a Gaussian-Bernoulli Restricted Boltzmann Machine with 
        adversarial training on hidden units.

        @args
        - n_vis: int << number of visible nodes
        - n_hid: int << number of hidden nodes
        - var: float | None << set variance for each visible node;
            if None, we learn the variance on each visible node
        """
        super().__init__()
        self.rng = torch.Generator()
        self.reset_seed(42)
        self.adversary_memory = None
        self.reset_hyperparameters(n_vis=n_vis, n_hid=n_hid, var=var)

    def metadata(self):
        """
        Returns the metadata of the RBM object.

        @returns
        - dict
        """
        metadata = {
            "n_vis": self.n_vis,
            "n_hid": self.n_hid,
            "var": self.var
        }
        return metadata
    
    def reset_hyperparameters(self, n_vis: int = None, n_hid: int = None, 
                              var: float = None):
        if n_vis is not None:
            self.n_vis = n_vis
        if n_hid is not None:
            self.n_hid = n_hid
        self.var = var
        self.W = nn.Parameter(torch.Tensor(self.n_vis, self.n_hid))
        self.mu = nn.Parameter(torch.Tensor(self.n_vis))
        self.b = nn.Parameter(torch.Tensor(self.n_hid))
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
        nn.init.constant_(self.mu, 0)
        nn.init.constant_(self.b, 0)
        if self.var is None:
            nn.init.constant_(self.log_var, 0)

    def reset_seed(self, seed: int):
        """
        Resets the rng seed to enforce reproducibility after training.

        @args
        - seed: int
        """
        self.rng.manual_seed(seed)

    def _energy(self, v: torch.Tensor, h: torch.Tensor):
        """
        Equation (1, 2) in https://arxiv.org/pdf/2210.10318.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - float
        """
        var = self._variance()
        pos = torch.sum(0.5 * (v - self.mu) ** 2 / var, dim=1)
        neg = torch.sum(((v / var) @ (self.W)) * h, dim=1) + \
            torch.sum(h * self.b, dim=1)
        return (pos - neg) / v.shape[0]

    def _marginal_energy(self, v: torch.Tensor):
        """
        Equation (5) in https://arxiv.org/pdf/2210.10318.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)

        @returns
        - torch.Tensor
        """
        var = self._variance()
        pos = torch.sum(torch.square(0.5 * (v - self.mu) / var.sqrt()), dim=1)
        softmax_logit = ((v / var) @ self.W + self.b).clip(max=80)
        neg = torch.sum(torch.log(1 + torch.exp(softmax_logit)), dim=1)
        return pos - neg
  
    @torch.no_grad()
    def _prob_h_given_v(self, v: torch.Tensor):
        """
        Computes sigmoid activation for p(h=1|v) according to equation (3) in
        https://arxiv.org/pdf/2210.10318; in other words, computes the
        parameters for hidden Bernoulli random variables given visible units.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)

        @returns
        - torch.Tensor ~ (batch_size, n_hid)
        """
        return torch.sigmoid((v / self._variance()) @ self.W + self.b)

    @torch.no_grad()
    def _prob_v_given_h(self, h: torch.Tensor):
        """
        Computes mu for p(v|h) according to equation (4) in
        https://arxiv.org/pdf/2210.10318.

        @args
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - torch.Tensor ~ (batch_size, n_vis)
        """
        return h @ self.W.t() + self.mu
  
    @torch.no_grad()
    def _block_gibbs_sample(self, v: torch.Tensor = None,
                            h: torch.Tensor = None, clamp: torch.Tensor = None,
                            n_gibbs = 1, add_noise = True):
        """
        Familiar block Gibbs sampling method of visible and hidden units.

        @args
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
            v_sample = self._prob_v_given_h(h)
        else:
            v_sample = v.clone()
        m = v_sample.shape[0]
        h_sample = torch.bernoulli(self._prob_h_given_v(v_sample),
                                    generator=self.rng)
        for _ in range(n_gibbs):
            if clamp is not None:
                old_v_sample = v_sample.clone()
            v_sample = self._prob_v_given_h(h_sample)
            if add_noise:
                v_sample += torch.randn(size=(m, self.n_vis),
                                generator=self.rng) * std
            if clamp is not None:
                v_sample[clamp] = old_v_sample[clamp]
            h_sample = torch.bernoulli(self._prob_h_given_v(v_sample),
                                        generator=self.rng)
        return v_sample, h_sample

    def _variance(self):
        """
        Returns the variance; we only attempt to train the log variance.

        @returns
        - torch.Tensor | float
        """
        return torch.exp(self.log_var)
  
    @torch.no_grad()
    def _reconstruction_MSE(self, v: torch.Tensor):
        """
        Computes the MSE loss of a reconstructed visible unit and an input
        visible unit.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)

        @ returns
        - torch.Tensor
        """
        prob_h = self._prob_h_given_v(v)
        v_bar = self._prob_v_given_h(prob_h)
        return torch.mean((v_bar - v) ** 2)
  
    @torch.no_grad()
    def reconstruct(self, v: np.ndarray, clamp: np.ndarray = None,
                    random_init = True, n_gibbs: int = 1, add_noise=True):
        """
        Reconstructs the visible units.

        @args
        - v: np.array ~ (batch_size, n_vis)
        - clamp: boolean np.array ~ (batch_size, n_vis) << will only
            reconstruct elements marked False in the boolean mask
        - random_init: bool << if True, reset v via torch.randn
        - n_gibbs: int

        @returns
        - np.array ~ (batch_size, n_vis)
        """
        if random_init:
            v = torch.randn(v.shape, generator=self.rng).requires_grad_(False)
        else:
            v = torch.Tensor(v).requires_grad_(False)
        if clamp is not None:
            v_sample, _ = self._block_gibbs_sample(v=v, n_gibbs=n_gibbs,
                                                    clamp=torch.Tensor(clamp),
                                                    add_noise=add_noise)
        else:
            v_sample, _ = self._block_gibbs_sample(v=v, n_gibbs=n_gibbs,
                                                    add_noise=add_noise)
        return v_sample.numpy()
        
    @torch.no_grad()
    def metrics(self, v: np.ndarray, n_gibbs: int = 10):
        """
        Returns the reconstruction MSE, KL(h_data || h_model), 
        and KL(h_model || h_data), all as floats

        @args
        - v: np.array ~ (batch_size, n_vis)

        @returns 
        - float << recon_mse
        - float << kl_vdata_vmodel
        - float << kl_vmodel_vdata
        """
        v_model, _ = self._block_gibbs_sample(torch.Tensor(v), 
                                              n_gibbs=n_gibbs)
        v_model = v_model.numpy()
        kl_vdata_vmodel = approx_kl_div(v, v_model)
        kl_vmodel_vdata = approx_kl_div(v_model, v)
        recon_mse = self._reconstruction_MSE(torch.Tensor(v)).item()
        return recon_mse, kl_vdata_vmodel, kl_vmodel_vdata

    def cd_loss(self, v: np.ndarray, n_gibbs: int = 1, gamma: float = 1):
        """
        Computes the contrastive divergence loss with which parameters may be
        updated via an optimizer. Follows Algorithm 3 of
        https://arxiv.org/pdf/2210.10318. Contrastive divergence loss 
        may be combined with an adversarial loss as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - n_gibbs: int
        - gamma: float << weight of the non-adversarial loss component

        @returns
        - torch.Tensor ~ (1) << contrastive divergence loss
        """
        v_data: torch.Tensor = torch.Tensor(v)
        _, h_data = self._block_gibbs_sample(v_data, n_gibbs=0)
        v_model, h_model = self._block_gibbs_sample(v_data, 
                                                    n_gibbs=n_gibbs)
        L = self._energy(v_data, h_data) - self._energy(v_model, h_model)
        if gamma == 1:
            return L.mean()
        A = self._weighted_neighbors_critic(h_model)
        if A is None:
            return L.mean()
        else:
            A = torch.mean(A)
        return gamma * L.mean() + (1 - gamma) * A

    @torch.no_grad()
    def _energy_grad_param(self, v: torch.Tensor, h: torch.Tensor):
        """
        Computes the gradient of energy with respect to parameter averaged 
        over the batch size. See the repository associated with the paper 
        https://arxiv.org/pdf/2210.10318:
        https://github.com/DSL-Lab/GRBM/blob/main/grbm.py.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - dict
        """
        var = self._variance()
        grad = {}
        grad["W"] = -torch.einsum("bi,bj->ij", v / var, h) / v.shape[0]
        grad["b"] = -h.mean(dim=0)
        grad["mu"] = ((self.mu - v) / var).mean(dim=0)
        grad["log_var"] = (-0.5 * (v - self.mu)**2 / var +
                            ((v / var) * h.mm(self.W.T))).mean(dim=0)
        return grad     

    @torch.no_grad()
    def _energy_grad_param_no_avg(self, v: torch.Tensor, h: torch.Tensor):
        """
        Computes the gradient of energy with respect to parameter. 
        See the repository associated with the paper 
        https://arxiv.org/pdf/2210.10318:
        https://github.com/DSL-Lab/GRBM/blob/main/grbm.py.

        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - h: torch.Tensor ~ (batch_size, n_hid)

        @returns
        - dict
        """
        var = self._variance()
        grad = {}
        grad["W"] = -((v / var)[:, :, np.newaxis] @ h[:, np.newaxis, :])
        grad["b"] = -h
        grad["mu"] = ((self.mu - v) / var)
        grad["log_var"] = (-0.5 * (v - self.mu)**2 / var +
                            ((v / var) * h.mm(self.W.T)))
        return grad
  
    @torch.no_grad()
    def _update_adversary_memory(self, v: np.ndarray, n_gibbs = 10):
        """
        Re-caches a minibatch of visible units drawn from the data and visible
        units drawn from the model, as described in 
        https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v_data = torch.Tensor(v)
        _, h_data = self._block_gibbs_sample(v_data, n_gibbs=0)
        _, h_model = self._block_gibbs_sample(v_data, n_gibbs=n_gibbs)
        self.adversary_memory = np.vstack((h_model.numpy(), h_data.numpy()))
    
    @torch.no_grad()
    def _total_grad(self, v: torch.Tensor, n_gibbs = 1, gamma = 1.0):
        """
        Combines the CD loss gradient with the adversarial
        gradient as described in https://arxiv.org/abs/1804.08682. 

        @args
        - v: np.array ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        - gamma: float << proportion of loss from non-adversarial training 
        """
        v_data = v.clone()
        _, h_data = self._block_gibbs_sample(v_data, n_gibbs=0)
        v_model, h_model = self._block_gibbs_sample(v_data, n_gibbs=n_gibbs)
        pos_grad = self._energy_grad_param_no_avg(v_data, h_data)
        neg_grad = self._energy_grad_param_no_avg(v_model, h_model)
        grad = {}
        for key in pos_grad.keys():
            grad[key] = torch.mean(pos_grad[key] - neg_grad[key], dim=0)
            critic = self._nearest_neighbors_critic(h_model)
            if critic is not None:
                grad[key] = gamma * grad[key] + (1 - gamma) * \
                    self._adversarial_grad(critic, neg_grad[key])
        return grad

    @torch.no_grad()
    def _positive_grad(self, v: torch.Tensor):
        """
        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        """
        _, h = self._block_gibbs_sample(v=v, n_gibbs=0)
        grad = self._energy_grad_param(v, h)
        return grad
    
    @torch.no_grad()
    def _negative_grad(self, v: torch.Tensor, n_gibbs = 1):
        """
        @args
        - v: torch.Tensor ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v, h = self._block_gibbs_sample(v, n_gibbs=n_gibbs)
        grad = self._energy_grad_param(v, h)
        return grad
  
    @torch.no_grad()
    def _adversarial_grad(self, critic: torch.Tensor, param: torch.Tensor):
        """
        @args
        - critic: torch.Tensor ~ (batch_size)
        - param: torch.Tensor ~ (batch_size, n) or (batch_size, m, n)
        """
        batch_size = critic.shape[0]
        param_subtracted = param - torch.mean(param, axis=0)
        critic_subtracted = critic - torch.mean(critic, axis=0)
        param_reshaped = param_subtracted.permute(*range(1, param.dim()), 0)
        cov = torch.tensordot(param_reshaped, critic_subtracted,
                                dims=([-1], [0]))
        return cov / batch_size

    @torch.no_grad()
    def cd_grad(self, v: np.ndarray, n_gibbs = 1):
        """
        Updates gradients of the parameters. 

        @args
        - v: np.ndarray ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v_tensor = torch.Tensor(v)
        pos = self._positive_grad(v_tensor)
        neg = self._negative_grad(v_tensor, n_gibbs)
        for name, param in self.named_parameters():
            param.grad = pos[name] - neg[name]
  
    @torch.no_grad()
    def cd_grad_adversarial(self, v: np.ndarray, n_gibbs = 1, gamma = 1):
        """
        Updates gradients of the parameters, but with adversarial training. 

        @args
        - v: np.ndarray ~ (batch_size, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps to sample from model
        """
        v_tensor = torch.Tensor(v)
        grad = self._total_grad(v_tensor, n_gibbs, gamma)
        for name, param in self.named_parameters():
            param.grad = grad[name]
    
    @torch.no_grad()
    def _nearest_neighbors_critic(self, h_sample: torch.Tensor, k=5):
        """
        Nearest neighbors linear critic. 

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
        ind, _ = k_nearest_neighbors(self.adversary_memory, h_sample, k)
        j = np.sum(ind >= batch_size, axis=1)
        return torch.Tensor(2 * j / k - 1)
    
    @torch.no_grad()
    def _weighted_neighbors_critic(self, h_sample: torch.Tensor, k=5):
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
                                             h_sample.numpy(), k)
        mask_data = ind >= batch_size
        return torch.Tensor(2 * inverse_distance_sum(distances, mask_data) \
                            / inverse_distance_sum(distances) - 1)
        
    def fit(self, X: np.ndarray, n_gibbs: int = 1,
            lr: float = 0.1, n_epochs: int = 100, batch_size: int = 10,
            gamma: float = 1.0, gamma_delay: int = 10, fail_tol: int = None,
            rng_seed: int = 0, verbose_interval: int = None, 
            reduce_lr_on_plateau = False, checkpoint_path = None):
        """
        Built-in, simple train method. Gradients are computed analytically. 
        Robust to NaNs in X. 

        @args
        - X: np.ndarray ~ (n_examples, n_vis)
        - n_gibbs: int << number of Gibbs sampling steps (k in the CD-k loss), 
            literature recommends to keep at 1
        - lr: float << learning rate
        - n_epochs: int << number of epochs
        - batch_size: int
        - gamma: float << proportion of loss coming from the 
        """
        stats = {
            'epoch_num': [],
            'recon_mse': [],
            'kl_data_model': [],
            'kl_model_data': []
        }
        self.reset_seed(rng_seed)
        contains_missing = np.any(np.isnan(X))
        if fail_tol is None:
            fail_tol = n_epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        recon_loss_history = 1000
        fail_count = 0
        for epoch in range(0, n_epochs + 1):
            if fail_count >= fail_tol:
                break
            reconstruction_loss = 0
            self.train()
            batched_train_data, _ = partition_into_batches([X], \
                batch_size, rng_seed + epoch)
            for batch in batched_train_data:
                if contains_missing:
                    missing_mask = np.isnan(batch[0])
                    clamp = np.logical_not(missing_mask)
                    batch[0][missing_mask] = 0
                    batch[0] = self.reconstruct(v=batch[0], clamp=clamp,
                                             random_init=False, n_gibbs=n_gibbs)
                if epoch >= gamma_delay and gamma < 1:
                    self._update_adversary_memory(batch[0])
                optimizer.zero_grad()
                batch_train_recon = \
                    self._reconstruction_MSE(torch.Tensor(batch[0]))
                if epoch >= gamma_delay and gamma < 1:
                    self.cd_grad_adversarial(batch[0], n_gibbs, gamma)
                else:
                    self.cd_grad(batch[0], n_gibbs)
                optimizer.step()
                reconstruction_loss += batch_train_recon
            reconstruction_loss /= len(batched_train_data)
            if reduce_lr_on_plateau:
                scheduler.step(reconstruction_loss)
            if reconstruction_loss > recon_loss_history:
                fail_count += 1
            recon_loss_history = reconstruction_loss
            if verbose_interval is not None:
                if epoch % verbose_interval == 0:
                    msg = f"\repoch: {str(epoch).zfill(len(str(n_epochs)))}"
                    msg += f" of {n_epochs}"
                    metrics_idx = np.random.choice(len(X), min(100, len(X)), 
                                                   replace=False)
                    recon_mse, kl_data_model, kl_model_data \
                        = self.metrics(X[metrics_idx], n_gibbs=n_gibbs)
                    msg += f" | recon_mse: {round(recon_mse, 3)}"
                    msg += f" | kl_data_model: {round(kl_data_model, 3)}"
                    msg += f" | kl_model_data: {round(kl_model_data, 3)}"
                    print(msg, end="\n")
                    stats['epoch_num'].append(epoch)
                    stats['recon_mse'].append(recon_mse)
                    stats['kl_data_model'].append(kl_data_model)
                    stats['kl_model_data'].append(kl_model_data)
        if checkpoint_path is not None:
            metadata_path = ".".join(checkpoint_path.split(".")[:-1]) + \
                ".json"
            torch.save(self.state_dict(), checkpoint_path)
            with open(metadata_path, "w") as json_file:
                json.dump(self.metadata(), json_file)
        return stats

    def fit_autograd(self, X: np.ndarray, n_gibbs: int = 1,
            lr: float = 0.1, n_epochs: int = 1, batch_size: int = 1,
            gamma: float = 1.0, gamma_delay: int = 10, 
            fail_tol: int = None, rng_seed: int = 0, 
            verbose_interval: int = None, reduce_lr_on_plateau = False, 
            checkpoint_path = None):
        """
        Built-in, simple train method that relies on Torch autograd 
        for computation of gradients. Robust to NaNs in X. 
        """
        stats = {
            'epoch_num': [],
            'recon_mse': [],
            'kl_data_model': [],
            'kl_model_data': []
        }
        self.reset_seed(rng_seed)
        contains_missing = np.any(np.isnan(X))
        if fail_tol is None:
            fail_tol = n_epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        recon_loss_history = 1000
        fail_count = 0
        for epoch in range(0, n_epochs + 1):
            if fail_count >= fail_tol:
                break
            reconstruction_loss = 0
            self.train()
            batched_train_data, _ = partition_into_batches([X], \
                batch_size, rng_seed + epoch)
            for batch in batched_train_data:
                if contains_missing:
                    missing_mask = np.isnan(batch[0])
                    clamp = np.logical_not(missing_mask)
                    batch[0][missing_mask] = 0
                    batch[0] = self.reconstruct(v=batch[0], clamp=clamp,
                                             random_init=False, n_gibbs=n_gibbs)
                if epoch >= gamma_delay and gamma < 1:
                    self._update_adversary_memory(batch[0],
                                                    n_gibbs=10)
                batch_train_recon = \
                    self._reconstruction_MSE(torch.Tensor(batch[0]))
                optimizer.zero_grad()
                if epoch >= gamma_delay and gamma < 1:
                    loss = self.cd_loss(batch[0], n_gibbs, gamma)
                else:
                    loss = self.cd_loss(batch[0], n_gibbs)
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
                    msg = f"\repoch: {str(epoch).zfill(len(str(n_epochs)))}"
                    msg += f" of {n_epochs}"
                    metrics_idx = np.random.choice(len(X), min(100, len(X)), 
                                                   replace=False)
                    recon_mse, kl_data_model, kl_model_data \
                        = self.metrics(X[metrics_idx], n_gibbs=n_gibbs)
                    msg += f" | loss: {round(loss.item(), 3)}"
                    msg += f" | recon_mse: {round(recon_mse, 3)}"
                    msg += f" | kl_data_model: {round(kl_data_model, 3)}"
                    msg += f" | kl_model_data: {round(kl_model_data, 3)}"
                    print(msg, end="\n")
                    stats['epoch_num'].append(epoch)
                    stats['recon_mse'].append(recon_mse)
                    stats['kl_data_model'].append(kl_data_model)
                    stats['kl_model_data'].append(kl_model_data)
            # if checkpoint_path is not None:
            #     metadata_path = os.path.splitext(checkpoint_path)[0] + f"-{epoch}" + ".json"
            #     newpath = os.path.splitext(checkpoint_path)[0] + f"-{epoch}" + ".pth"
            #     torch.save(self.state_dict(), newpath)
            #     with open(metadata_path, "w") as json_file:
            #         json.dump(self.metadata(), json_file)
        if checkpoint_path is not None:
            metadata_path = os.path.splitext(checkpoint_path)[0] + ".json"
            torch.save(self.state_dict(), checkpoint_path)
            with open(metadata_path, "w") as json_file:
                json.dump(self.metadata(), json_file)
        return stats

def load(checkpoint_path: str, metadata_path: str = None) -> RBM:
    """
    Given a checkpoint path and optionally a metadata path, construct a RBM

    @args
    - checkpoint_path: str
    - metadata_path: str | None << if None, infers path from checkpoint_path

    @returns
    - RBM
    """
    if metadata_path is None:
        metadata_path = ".".join(checkpoint_path.split(".")[:-1]) + ".json"
    with open(metadata_path, "r") as json_file:
        metadata = json.load(json_file)
    model = RBM(
        metadata["n_vis"], metadata["n_hid"], metadata["var"]
    )
    model.load_state_dict(torch.load(checkpoint_path))
    return model




