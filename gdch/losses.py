from typing import Dict, Optional, Tuple

import torch


def add_regularization(
    nll: torch.Tensor,
    reg_terms: Optional[Dict[str, torch.Tensor]],
    model,
    reg_cfg: Optional[Dict],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    reg_cfg = reg_cfg or {}
    lambda_z = float(reg_cfg.get("lambda_z", 0.0))
    lambda_jump = float(reg_cfg.get("lambda_jump", 0.0))
    lambda_beta = float(reg_cfg.get("lambda_beta", 0.0))

    zero = torch.zeros((), device=nll.device, dtype=nll.dtype)
    z_term = reg_terms.get("z", zero) if reg_terms is not None else zero
    jump_term = reg_terms.get("jump", zero) if reg_terms is not None else zero
    beta_term = model.beta.pow(2)

    reg_loss = lambda_z * z_term + lambda_jump * jump_term + lambda_beta * beta_term
    total = nll + reg_loss

    components = {
        "z": z_term.detach(),
        "jump": jump_term.detach(),
        "beta": beta_term.detach(),
        "reg_loss": reg_loss.detach(),
    }
    return total, components
