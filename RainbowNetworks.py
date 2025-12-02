import torch
import torch.nn as nn
from PopArtLayer import PopArtLayer
from PopArtDuelingLayer import PopArtDuelingHead
from PopArtIQNLayer import PopArtIQNLayer
from PopArtDuelingIQNLayer import PopArtDuelingIQNLayer


class EV_Q_Network(nn.Module):
    """Single-valued (non-distributional) multi-dimensional Q network with optional dueling.

    Outputs Q-values per action dimension and per bin: [B, D, Bins].
    Dueling implemented by producing a single value head plus an advantage head of size D*Bins,
    which is reshaped and mean-subtracted across all (D*Bins) entries.
    """

    def __init__(
        self,
        input_dim: int,
        n_action_dims: int,
        n_action_bins: int,
        hidden_layer_sizes=[128, 128],
        dueling: bool = False,
        popart: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.popart = popart
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions_total = n_action_dims * n_action_bins  # legacy convenience
        layers = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        for li in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        last_hidden = hidden_layer_sizes[-1]
        if self.dueling:
            if self.popart:
                self.output_layer = PopArtDuelingHead(last_hidden, self.n_actions_total)
            else:
                self.value_layer = nn.Linear(last_hidden, 1)
                self.advantage_layer = nn.Linear(last_hidden, self.n_actions_total)
        else:
            if self.popart:
                self.output_layer = PopArtLayer(last_hidden, self.n_actions_total)
            else:
                self.out_layer = nn.Linear(last_hidden, self.n_actions_total)

    def forward(self, x: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))

        if self.popart:
            out = self.output_layer(h, normalized=normalized)
            if h.ndim == 1:
                return out.view(self.n_action_dims, self.n_action_bins)
            return out.view(h.shape[0], self.n_action_dims, self.n_action_bins)

        if not self.dueling:
            out = self.out_layer(h)
            if h.ndim == 1:
                return out.view(self.n_action_dims, self.n_action_bins)
            return out.view(h.shape[0], self.n_action_dims, self.n_action_bins)
        if h.ndim == 1:
            v = self.value_layer(h).view(1)  # scalar value
            a = self.advantage_layer(h).view(self.n_action_dims, self.n_action_bins)
            a_mean = a.mean()
            q = a - a_mean + v
            return q
        bsz = h.shape[0]
        v = self.value_layer(h).view(bsz, 1, 1)
        a = self.advantage_layer(h).view(bsz, self.n_action_dims, self.n_action_bins)
        a_mean = a.mean(dim=(1, 2), keepdim=True)
        q = a - a_mean + v
        return q

    def expected_value(self, x: torch.Tensor, probs: bool = False):
        # For EV network, expected value per dimension/bin is x itself.
        return x


class IQN_Network(nn.Module):
    """Implicit Quantile Network supporting multi-dimensional discrete actions.

    Output shape: [B, Nq, D, Bins].
    Dueling implemented similarly to EV_Q_Network using a shared advantage head sized D*Bins.
    """

    def __init__(
        self,
        input_dim: int,
        n_action_dims: int,
        n_action_bins: int,
        hidden_layer_sizes=[128, 128],
        n_cosines: int = 64,
        dueling: bool = False,
        popart: bool = False,
    ):
        super().__init__()
        self.dueling = dueling
        self.popart = popart
        self.n_action_dims = n_action_dims
        self.n_action_bins = n_action_bins
        self.n_actions_total = n_action_dims * n_action_bins
        self.n_cosines = n_cosines
        base = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        for li in range(len(hidden_layer_sizes) - 1):
            base.append(nn.Linear(hidden_layer_sizes[li], hidden_layer_sizes[li + 1]))
        self.base_layers = nn.ModuleList(base)
        self.relu = nn.ReLU()
        last_hidden = hidden_layer_sizes[-1]
        self.cosine_layer = nn.Linear(n_cosines, last_hidden)
        if dueling:
            if self.popart:
                self.output_layer = PopArtDuelingIQNLayer(
                    last_hidden, n_action_dims, n_action_bins
                )
            else:
                self.value_head = nn.Linear(last_hidden, 1)
                self.adv_head = nn.Linear(last_hidden, self.n_actions_total)
        else:
            if self.popart:
                self.output_layer = PopArtIQNLayer(last_hidden, self.n_actions_total)
            else:
                self.out_head = nn.Linear(last_hidden, self.n_actions_total)

    def _phi(self, tau: torch.Tensor) -> torch.Tensor:
        i = torch.arange(self.n_cosines, device=tau.device, dtype=tau.dtype).view(
            1, 1, -1
        )
        cosines = torch.cos(i * torch.pi * tau.unsqueeze(-1))
        emb = self.relu(self.cosine_layer(cosines))  # [B,N,H]
        return emb

    def forward(
        self, x: torch.Tensor, tau: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        h = x
        for layer in self.base_layers:
            h = self.relu(layer(h))  # [B,H]
        h = h.unsqueeze(1)  # [B,1,H]
        tau_emb = self._phi(tau)  # [B,N,H]
        h = h * tau_emb  # [B,N,H]

        if self.popart:
            out = self.output_layer(h, normalized=normalized)
            if self.dueling:
                return out
            else:
                return out.view(
                    out.shape[0], out.shape[1], self.n_action_dims, self.n_action_bins
                )

        if not self.dueling:
            out = self.out_head(h)  # [B,N,D*Bins]
            return out.view(
                out.shape[0], out.shape[1], self.n_action_dims, self.n_action_bins
            )
        v = self.value_head(h)  # [B,N,1]
        a = self.adv_head(h)  # [B,N,D*Bins]
        a = a.view(a.shape[0], a.shape[1], self.n_action_dims, self.n_action_bins)
        a_mean = a.mean(dim=(2, 3), keepdim=True)
        q = a - a_mean + v.unsqueeze(-1)  # broadcast v
        return q  # [B,N,D,Bins]

    def expected_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        # Mean over quantile dimension -> [B,D,Bins]
        return quantiles.mean(dim=1)
