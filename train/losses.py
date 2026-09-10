import torch
import math

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

def rel_l1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    epsilon_tensor = torch.tensor(epsilon, device=y_true.device, dtype=y_true.dtype)
    return torch.mean(torch.abs((y_true - y_pred) / torch.maximum(torch.abs(y_true), epsilon_tensor))) 

class RelL1Loss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        return rel_l1_loss(y_true, y_pred, self.epsilon)

def heteroscedastic_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    #print('sigma before clamping:', sigma)
    sigma_clamped = torch.clamp(sigma, min=1e-15, max=1e6)
    #print('sigma_clamped:', sigma_clamped)
    #print('y_true:', y_true)
    #print('y_pred:', y_pred)
    loss = ((y_true - y_pred) ** 2) / (2 * sigma_clamped ** 2) + torch.log(sigma_clamped)
    #print('heteroscedastic loss per sample:', loss)
    return torch.mean(loss)

class HeteroscedasticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return heteroscedastic_loss(y_pred, y_true, sigma)

class MAEToHetLoss(torch.nn.Module):
    """Linearly interpolates from MAE (alpha=0) to heteroscedastic loss (alpha=1).
    Focuses on large residuals early in training via MAE, then transitions to the
    full uncertainty-aware het loss as alpha → 1."""
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                sigma: torch.Tensor, alpha: float) -> torch.Tensor:
        mae = torch.mean(torch.abs(y_true - y_pred))
        het = heteroscedastic_loss(y_pred, y_true, sigma)
        return (1.0 - alpha) * mae + alpha * het