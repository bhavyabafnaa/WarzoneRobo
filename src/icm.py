import torch
import torch.nn.functional as F
from torch import nn


class ICMModule(nn.Module):
    """Intrinsic Curiosity Module."""

    def __init__(self, input_dim, action_dim, feature_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ):
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)

        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(pred_action_logits, action)

        forward_input = torch.cat([phi_s, action_onehot], dim=-1)
        pred_phi_next = self.forward_model(forward_input)

        # The target feature is detached so the forward model cannot reduce
        # its error by moving both sides of the objective together.  The
        # per-transition prediction error is the intrinsic reward.  Do not
        # min-max normalize it here: training currently evaluates one
        # transition at a time, for which min == max and the old
        # normalization collapsed every ICM reward to exactly zero.
        target_phi_next = phi_s_next.detach()
        forward_error = F.mse_loss(
            pred_phi_next,
            target_phi_next,
            reduction="none",
        ).mean(dim=1)
        forward_loss = forward_error.mean()
        curiosity = forward_error.detach()

        return curiosity, forward_loss, inverse_loss
