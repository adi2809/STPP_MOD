import math
import weakref
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

from gdch.features import TimeEmbedding, calendar_features


def inv_softplus(x: float) -> float:
    x = max(x, 1e-6)
    return math.log(math.expm1(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims,
        out_dim: int,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    def __init__(self, model: "GDCH"):
        super().__init__()
        self._model_ref = weakref.ref(model)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("GDCH model reference has been cleared")
        return model._ode_rhs(t, y)


class GDCH(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        latent_dim: int,
        time_embed_dim: int,
        node_embed_dim: int,
        mlp_hidden_dim: Optional[int] = None,
        time_hidden_dim: Optional[int] = None,
        mlp_layers: Optional[Tuple[int, ...]] = None,
        time_mlp_layers: Optional[Tuple[int, ...]] = None,
        mlp_dropout: float = 0.0,
        mlp_layer_norm: bool = False,
        time_mlp_dropout: float = 0.0,
        time_mlp_layer_norm: bool = False,
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
        jump_eta: float = 0.0,
        jump_tanh: bool = False,
        jump_scale: float = 1.0,
        gate_use_z: bool = False,
        intensity_use_z: bool = False,
        eps: float = 1e-8,
        t0_days: float = 0.0,
        total_time: float = 1.0,
        baseline_init: float = 0.0,
        per_node_w: bool = False,
        laplacian: Optional[torch.Tensor] = None,
        jump_kernel: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.N = num_nodes
        self.d = latent_dim
        self.d_g = time_embed_dim
        self.d_e = node_embed_dim
        self.d_t = int(calendar_features(torch.tensor(0.0)).shape[-1])

        mlp_hidden_dim = mlp_hidden_dim or max(latent_dim, 16)
        time_hidden_dim = time_hidden_dim or max(time_embed_dim, 16)
        mlp_layers = mlp_layers or (mlp_hidden_dim,)
        time_mlp_layers = time_mlp_layers or (time_hidden_dim,)

        self.Z0 = nn.Parameter(torch.randn(num_nodes, latent_dim) * 0.01)
        self.b = nn.Parameter(torch.full((num_nodes,), float(baseline_init)))
        self.per_node_w = bool(per_node_w)
        if self.per_node_w:
            self.w = nn.Parameter(torch.randn(num_nodes, latent_dim) * 0.01)
        else:
            self.w = nn.Parameter(torch.randn(latent_dim) * 0.01)

        self._alpha = nn.Parameter(torch.tensor(inv_softplus(alpha_init), dtype=torch.float32))
        self._beta = nn.Parameter(torch.tensor(inv_softplus(beta_init), dtype=torch.float32))

        self.time_embed = TimeEmbedding(
            self.d_t,
            time_mlp_layers,
            time_embed_dim,
            dropout=time_mlp_dropout,
            layer_norm=time_mlp_layer_norm,
        )
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)

        self.mlp_u = MLP(time_embed_dim, mlp_layers, latent_dim, dropout=mlp_dropout, layer_norm=mlp_layer_norm)
        self.gate_use_z = bool(gate_use_z)
        self.intensity_use_z = bool(intensity_use_z)
        r_in_dim = time_embed_dim + node_embed_dim + (latent_dim if self.gate_use_z else 0)
        self.mlp_r = MLP(r_in_dim, mlp_layers, latent_dim, dropout=mlp_dropout, layer_norm=mlp_layer_norm)
        self.jump_net = MLP(
            latent_dim + time_embed_dim + node_embed_dim,
            mlp_layers,
            latent_dim,
            dropout=mlp_dropout,
            layer_norm=mlp_layer_norm,
        )
        h_in_dim = time_embed_dim + node_embed_dim + (latent_dim if self.intensity_use_z else 0)
        self.intensity_net = MLP(h_in_dim, mlp_layers, 1, dropout=mlp_dropout, layer_norm=mlp_layer_norm)

        self.jump_eta = float(jump_eta)
        self.jump_tanh = bool(jump_tanh)
        self.jump_scale = float(jump_scale)
        self.eps = float(eps)
        self.t0_days = float(t0_days)
        self.total_time = float(total_time) if total_time > 0 else 1.0

        if laplacian is not None:
            self.register_buffer("L", laplacian)
        else:
            self.L = None

        if jump_kernel is not None:
            self.register_buffer("K", jump_kernel)
        else:
            self.K = None

        self.ode_func = ODEFunc(self)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.Z0.device, dtype=self.Z0.dtype)
        else:
            t = t.to(device=self.Z0.device, dtype=self.Z0.dtype)
        phi = calendar_features(t, self.t0_days, self.total_time)
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        g = self.time_embed(phi)
        if g.dim() == 2 and g.shape[0] == 1:
            g = g.squeeze(0)
        return g

    def _flatten_state(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.reshape(-1), a.reshape(1)])

    def _unflatten_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = y[:-1].reshape(self.N, self.d)
        a = y[-1]
        return z, a

    def _make_tspan(
        self, t0: torch.Tensor, t1: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if torch.is_tensor(t0):
            t0_t = t0.to(device=device, dtype=dtype)
        else:
            t0_t = torch.tensor(t0, device=device, dtype=dtype)
        if torch.is_tensor(t1):
            t1_t = t1.to(device=device, dtype=dtype)
        else:
            t1_t = torch.tensor(t1, device=device, dtype=dtype)
        return torch.stack([t0_t, t1_t])

    def _odeint(self, y0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, solver_config: Optional[Dict]) -> torch.Tensor:
        solver_config = solver_config or {}
        method = solver_config.get("method", "dopri5")
        rtol = solver_config.get("rtol", 1e-4)
        atol = solver_config.get("atol", 1e-6)
        max_num_steps = solver_config.get("max_num_steps", 1000)
        min_step = solver_config.get("min_step", None)
        max_step = solver_config.get("max_step", None)
        first_step = solver_config.get("first_step", None)
        step_size = solver_config.get("step_size", None)
        use_adjoint = solver_config.get("use_adjoint", False)

        time_dtype_name = str(solver_config.get("time_dtype", "float64")).lower()
        time_dtype = torch.float64 if time_dtype_name in ("float64", "double", "fp64") else y0.dtype
        options = {}
        if max_num_steps is not None:
            options["max_num_steps"] = int(max_num_steps)
        if min_step is not None:
            options["min_step"] = float(min_step)
        if max_step is not None:
            options["max_step"] = float(max_step)
        if first_step is not None:
            options["first_step"] = float(first_step)
        if step_size is not None:
            options["step_size"] = float(step_size)
        if not options:
            options = None
        t_span = self._make_tspan(t0, t1, y0.device, time_dtype)

        solver = odeint_adjoint if use_adjoint else odeint
        solver_kwargs = {"rtol": rtol, "atol": atol, "method": method}
        if options is not None:
            solver_kwargs["options"] = options
        if use_adjoint:
            solver_kwargs["adjoint_params"] = tuple(self.parameters())

        return solver(self.ode_func, y0, t_span, **solver_kwargs)

    def integrate(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        z0: torch.Tensor,
        a0: torch.Tensor,
        solver_config: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        solver_config = solver_config or {}
        min_dt = float(solver_config.get("min_dt", 1e-7))
        dt = t1 - t0
        dt_val = dt.item() if torch.is_tensor(dt) else float(dt)
        if dt_val <= min_dt:
            return z0, a0
        y0 = self._flatten_state(z0, a0)
        try:
            sol = self._odeint(y0, t0, t1, solver_config)
        except AssertionError:
            fallback_method = solver_config.get("fallback_method", "rk4")
            time_dtype_name = str(solver_config.get("time_dtype", "float64")).lower()
            time_dtype = torch.float64 if time_dtype_name in ("float64", "double", "fp64") else y0.dtype
            t_span = self._make_tspan(t0, t1, y0.device, time_dtype)
            sol = odeint(
                self.ode_func,
                y0,
                t_span,
                method=fallback_method,
                options={"step_size": float(dt_val)},
            )
        y1 = sol[-1]
        return self._unflatten_state(y1)

    def _compute_intensity(
        self, g: torch.Tensor, z: torch.Tensor, ge: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ge is None:
            e = self.node_embed.weight
            g_expand = g.unsqueeze(0).expand(self.N, -1)
            ge = torch.cat([g_expand, e], dim=-1)
        if self.intensity_use_z:
            gz = torch.cat([ge, z], dim=-1)
            h = self.intensity_net(gz).squeeze(-1)
        else:
            h = self.intensity_net(ge).squeeze(-1)
        if self.per_node_w:
            z_proj = (z * self.w).sum(dim=1)
        else:
            z_proj = z @ self.w
        lam = F.softplus(self.b + z_proj + h) + self.eps
        return lam, lam.sum()

    def intensity(self, t: torch.Tensor, z: torch.Tensor, g: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if g is None:
            g = self._time_embedding(t)
        return self._compute_intensity(g, z)

    def apply_jump(
        self, z: torch.Tensor, s: torch.Tensor, t: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if g is None:
            g = self._time_embedding(t)
        s_idx = int(s.item()) if torch.is_tensor(s) else int(s)
        z_s = z[s_idx]
        e_s = self.node_embed.weight[s_idx]
        jump_in = torch.cat([z_s, g, e_s], dim=-1).unsqueeze(0)
        delta = self.jump_net(jump_in).squeeze(0)
        if self.jump_tanh:
            delta = torch.tanh(delta)
        if self.jump_scale != 1.0:
            delta = delta * self.jump_scale

        z_post = z.clone()
        z_post[s_idx] = z_post[s_idx] + delta

        if self.K is not None and self.jump_eta != 0.0:
            spill = self.jump_eta * self.K[:, s_idx].unsqueeze(-1) * delta.unsqueeze(0)
            z_post = z_post + spill

        return z_post, delta

    def _ode_rhs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z, _a = self._unflatten_state(y)
        g = self._time_embedding(t)

        e = self.node_embed.weight
        g_expand = g.unsqueeze(0).expand(self.N, -1)
        ge = torch.cat([g_expand, e], dim=-1)

        u = self.mlp_u(g.unsqueeze(0)).squeeze(0)
        if self.gate_use_z:
            r_in = torch.cat([ge, z], dim=-1)
        else:
            r_in = ge
        r = torch.sigmoid(self.mlp_r(r_in))
        forcing = r * u.unsqueeze(0)

        if self.L is None:
            diffusion = torch.zeros_like(z)
        else:
            if self.L.is_sparse:
                lz = torch.sparse.mm(self.L, z)
            else:
                lz = self.L @ z
            diffusion = -self.beta * lz

        dz = -self.alpha * z + diffusion + forcing
        _lam, lam_sum = self._compute_intensity(g, z, ge=ge)
        da = lam_sum

        return self._flatten_state(dz, da)

    def nll_chunk(
        self,
        times: torch.Tensor,
        nodes: torch.Tensor,
        start_idx: int,
        end_idx: int,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        solver_config: Optional[Dict] = None,
        collect_reg: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        device = self.Z0.device
        times = times.to(device=device, dtype=self.Z0.dtype)
        nodes = nodes.to(device=device)

        if state is None:
            z = self.Z0
            a = torch.tensor(0.0, device=device, dtype=self.Z0.dtype)
        else:
            z, a = state

        loss = torch.zeros((), device=device, dtype=self.Z0.dtype)
        z_sum = torch.zeros((), device=device, dtype=self.Z0.dtype)
        jump_sum = torch.zeros((), device=device, dtype=self.Z0.dtype)

        num_events = end_idx - start_idx + 1
        for n in range(start_idx, end_idx + 1):
            t_n = times[n]
            s_n = nodes[n]

            g = self._time_embedding(t_n)
            lam, _ = self._compute_intensity(g, z)
            s_idx = int(s_n.item()) if torch.is_tensor(s_n) else int(s_n)
            loss = loss - torch.log(lam[s_idx])

            if collect_reg:
                z_sum = z_sum + z.pow(2).mean()

            z_post, delta = self.apply_jump(z, s_n, t_n, g=g)
            if collect_reg:
                jump_sum = jump_sum + delta.pow(2).mean()

            if n < len(times) - 1:
                t_next = times[n + 1]
                if t_next > t_n:
                    a_prev = a
                    z, a = self.integrate(t_n, t_next, z_post, a, solver_config)
                    loss = loss + (a - a_prev)
                else:
                    z = z_post
            else:
                z = z_post

        reg_terms = None
        if collect_reg and num_events > 0:
            reg_terms = {"z": z_sum / num_events, "jump": jump_sum / num_events}

        return loss, (z, a), reg_terms
