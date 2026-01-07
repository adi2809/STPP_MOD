import math
import unittest

import numpy as np
import torch
from torchdiffeq import odeint

from gdch.model import GDCH


def zero_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()


def generate_constant_rate_events(num_nodes: int, rates: np.ndarray, num_events: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    total_rate = float(rates.sum())
    times = []
    nodes = []
    t = 0.0
    for _ in range(num_events):
        t += rng.exponential(1.0 / total_rate)
        s = rng.choice(num_nodes, p=rates / total_rate)
        times.append(t)
        nodes.append(s)
    return np.array(times, dtype=np.float32), np.array(nodes, dtype=np.int64)


class TestGDCH(unittest.TestCase):
    def test_constant_intensity_no_dynamics(self):
        torch.manual_seed(0)
        model = GDCH(
            num_nodes=3,
            latent_dim=4,
            time_embed_dim=4,
            node_embed_dim=2,
            alpha_init=1e-6,
            beta_init=1e-6,
            jump_eta=0.0,
        )
        zero_module(model.mlp_u)
        zero_module(model.mlp_r)
        zero_module(model.intensity_net)
        zero_module(model.jump_net)

        t0 = torch.tensor(0.0)
        t1 = torch.tensor(0.5)
        z0 = model.Z0
        a0 = torch.tensor(0.0)
        lam0, _ = model.intensity(t0, z0)
        z1, _a1 = model.integrate(t0, t1, z0, a0, solver_config={})
        lam1, _ = model.intensity(t1, z1)

        self.assertTrue(torch.allclose(lam0, lam1, atol=1e-4, rtol=1e-4))

    def test_nll_decreases_on_toy_data(self):
        torch.manual_seed(1)
        np.random.seed(1)
        num_nodes = 3
        rates = np.array([0.2, 0.1, 0.15], dtype=np.float64)
        times, nodes = generate_constant_rate_events(num_nodes, rates, num_events=50, seed=1)

        model = GDCH(
            num_nodes=num_nodes,
            latent_dim=4,
            time_embed_dim=4,
            node_embed_dim=2,
            alpha_init=1e-6,
            beta_init=1e-6,
            jump_eta=0.0,
        )
        zero_module(model.mlp_u)
        zero_module(model.mlp_r)
        zero_module(model.intensity_net)
        zero_module(model.jump_net)
        with torch.no_grad():
            model.w.zero_()
            model.Z0.zero_()

        times_t = torch.tensor(times)
        nodes_t = torch.tensor(nodes)

        nll_before, _state, _ = model.nll_chunk(times_t, nodes_t, 0, len(times_t) - 1, solver_config={})

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(10):
            nll, _state, _ = model.nll_chunk(times_t, nodes_t, 0, len(times_t) - 1, solver_config={})
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            optimizer.step()

        nll_after, _state, _ = model.nll_chunk(times_t, nodes_t, 0, len(times_t) - 1, solver_config={})
        self.assertLess(nll_after.item(), nll_before.item())

    def test_lambda_positive_and_log_finite(self):
        torch.manual_seed(2)
        model = GDCH(
            num_nodes=4,
            latent_dim=4,
            time_embed_dim=4,
            node_embed_dim=2,
        )
        z0 = model.Z0
        t = torch.tensor(1.23)
        lam, _ = model.intensity(t, z0)
        self.assertTrue(torch.all(lam > 0))
        self.assertTrue(torch.isfinite(torch.log(lam)).all())

    def test_integral_matches_trapezoid(self):
        torch.manual_seed(3)
        model = GDCH(
            num_nodes=3,
            latent_dim=4,
            time_embed_dim=4,
            node_embed_dim=2,
        )
        zero_module(model.mlp_u)
        zero_module(model.mlp_r)
        zero_module(model.jump_net)

        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)
        t_grid = torch.linspace(t0, t1, steps=30)
        y0 = model._flatten_state(model.Z0, torch.tensor(0.0))
        sol = odeint(model.ode_func, y0, t_grid, rtol=1e-5, atol=1e-6, method="dopri5")
        a_end = sol[-1, -1]

        lam_sum = []
        for i in range(sol.shape[0]):
            z_i, _a = model._unflatten_state(sol[i])
            lam, lam_total = model.intensity(t_grid[i], z_i)
            lam_sum.append(lam_total)
        lam_sum = torch.stack(lam_sum)

        trap = torch.trapz(lam_sum, t_grid)
        self.assertTrue(torch.allclose(a_end, trap, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
