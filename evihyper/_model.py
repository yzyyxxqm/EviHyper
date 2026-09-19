import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .layers import CausalAnchor, QuaternionLinear, hamilton_product, quaternion_conjugate
from .config import EviHyperConfig


class Model(nn.Module):
    """Query-local observation-event hypergraph forecasting."""

    N_ROLES = 4

    @staticmethod
    def _parse_index_list(raw: object) -> tuple[int, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (list, tuple)):
            return tuple((int(item) for item in raw))
        text = str(raw).strip()
        if not text:
            return ()
        return tuple((int(item.strip()) for item in text.split(",") if item.strip()))

    @staticmethod
    def _parse_period_map(raw: object) -> dict[int, float]:
        if raw is None:
            return {}
        text = str(raw).strip()
        if not text:
            return {}
        periods: dict[int, float] = {}
        for item in text.split(","):
            token = item.strip()
            if not token:
                continue
            if ":" in token:
                key, value = token.split(":", 1)
            else:
                key, value = token.split("=", 1)
            period = float(value.strip())
            if period > 1e-06:
                periods[int(key.strip())] = period
        return periods

    def __init__(self, configs: EviHyperConfig):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four")
        self.part = self.d_model // 4
        self.time_mark_mode = str(configs.time_mark_mode)
        if self.time_mark_mode not in {"input_first", "prepend_relative"}:
            raise ValueError(f"Unknown time-mark mode: {self.time_mark_mode}")
        self.topk = max(0, int(configs.topk))
        self.residual_scale = max(0.0, float(configs.residual_scale))
        self.value_readout_scale = max(0.0, float(configs.value_readout_scale))
        self.extrapolate_weight = min(1.0, max(0.0, float(configs.extrapolate_weight)))
        self.residual_bound_ratio = max(0.0, float(configs.residual_bound_ratio))
        self.residual_bound_min = max(0.0, float(configs.residual_bound_min))
        self.hv_residual_bias_scale = max(0.0, float(configs.hv_residual_bias_scale))
        self.soft_memory_patches = max(1, int(configs.soft_memory_patches))
        self.soft_memory_scale = max(0.0, float(configs.soft_memory_scale))
        self.soft_memory_bound_ratio = max(0.0, float(configs.soft_memory_bound_ratio))
        self.soft_memory_phase_scale = max(0.0, float(configs.soft_memory_phase_scale))
        self.soft_memory_same_var_prior = float(configs.soft_memory_same_var_prior)
        self.obs_aux_weight = max(0.0, float(configs.obs_aux_weight))
        self.intensity_floor = min(1.0, max(0.0, float(configs.intensity_floor)))
        self.use_horizon_context = bool(configs.horizon_context)
        self.process_decay_scale = max(0.0, float(configs.process_decay_scale))
        self.process_decay_rate_scale = max(0.0, float(configs.process_decay_rate_scale))
        self.candidate_oracle_aux_weight = max(0.0, float(configs.candidate_oracle_aux_weight))
        self.candidate_oracle_temperature = max(0.001, float(configs.candidate_oracle_temperature))
        self.candidate_oracle_horizon_gamma = max(
            0.0, float(configs.candidate_oracle_horizon_gamma)
        )
        self.candidate_margin_aux_weight = max(0.0, float(configs.candidate_margin_aux_weight))
        self.candidate_margin_threshold = max(0.0, float(configs.candidate_margin_threshold))
        self.candidate_prior_scale = max(0.0, float(configs.candidate_prior_scale))
        self.sixth_candidate_logit_bias = float(configs.sixth_candidate_logit_bias)
        self.sixth_candidate_max_mass = min(1.0, max(0.0, float(configs.sixth_candidate_max_mass)))
        self.periodic_post_blend = min(1.0, max(0.0, float(configs.periodic_post_blend)))
        self.station_time_steps = max(0, int(configs.station_time_steps))
        self.station_count = max(1, int(configs.station_count))
        self.station_value_bias_enabled = bool(configs.station_value_bias)
        self.station_value_bias_scale = max(0.0, float(configs.station_value_bias_scale))
        self.station_phase_bias_enabled = bool(configs.station_phase_bias)
        self.station_phase_bias_scale = max(0.0, float(configs.station_phase_bias_scale))
        self.station_phase_components = min(3, max(1, int(configs.station_phase_components)))
        self.circular_embedding_variables = tuple(
            (
                idx
                for idx in self._parse_index_list(configs.circular_embedding_variables)
                if 0 <= idx < self.enc_in
            )
        )
        self.circular_embedding_periods = self._parse_period_map(configs.circular_embedding_periods)
        self.anchor = CausalAnchor(configs)
        self.variable_embedding = nn.Embedding(self.enc_in, self.part)
        self.horizon_embedding = nn.Embedding(self.pred_len, self.d_model)
        self.node_value = nn.Linear(4, self.part)
        self.node_circular_value = nn.Linear(4, self.part)
        self.node_phase = nn.Linear(4, self.part)
        self.node_reliability = nn.Linear(4, self.part)
        self.query_value = nn.Linear(4, self.part)
        self.query_circular_value = nn.Linear(4, self.part)
        self.query_phase = nn.Linear(4, self.part)
        self.query_reliability = nn.Linear(4, self.part)
        self.obs_intensity_net = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, max(8, self.part)),
            nn.GELU(),
            nn.Linear(max(8, self.part), 1),
        )
        self.query_intensity_net = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, max(8, self.part)),
            nn.GELU(),
            nn.Linear(max(8, self.part), 1),
        )
        self.variable_coupling = nn.Parameter(torch.zeros(self.enc_in, self.enc_in))
        self.role_code = nn.Parameter(torch.randn(self.N_ROLES, self.d_model) * 0.02)
        self.soft_memory_delta = nn.Parameter(torch.zeros(self.enc_in, self.soft_memory_patches))
        base_width = 1.0 / float(self.soft_memory_patches)
        self.soft_memory_log_width = nn.Parameter(
            torch.full((self.enc_in, self.soft_memory_patches), math.log(base_width))
        )
        self.soft_memory_tau = nn.Parameter(torch.zeros(self.enc_in))
        self.soft_memory_query = nn.Parameter(
            torch.randn(self.enc_in, self.soft_memory_patches, self.d_model) * 0.02
        )
        self.soft_memory_source_bias = nn.Parameter(torch.zeros(self.enc_in, self.enc_in))
        self.soft_memory_cross_residual_scale = nn.Parameter(torch.zeros(self.enc_in, self.enc_in))
        self.query_proj = QuaternionLinear(self.d_model, self.d_model)
        self.event_proj = QuaternionLinear(self.d_model, self.d_model)
        self.role_transport = QuaternionLinear(self.d_model, self.d_model)
        self.soft_memory_transport = QuaternionLinear(self.d_model, self.d_model)
        self.role_gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 2 + 10),
            nn.Linear(self.d_model * 2 + 10, max(16, self.part)),
            nn.GELU(),
            nn.Linear(max(16, self.part), self.N_ROLES),
        )
        self.residual_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model * 2 + 10),
            nn.Linear(self.d_model * 2 + 10, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )
        self.process_decay_gate = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, max(8, self.part)),
            nn.GELU(),
            nn.Linear(max(8, self.part), 1),
        )
        self.seasonal_correction_gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 2 + 10),
            nn.Linear(self.d_model * 2 + 10, max(16, self.part)),
            nn.GELU(),
            nn.Linear(max(16, self.part), 1),
        )
        self.soft_memory_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model * 2 + 12),
            nn.Linear(self.d_model * 2 + 12, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1),
        )
        self.candidate_mixer = nn.Sequential(
            nn.LayerNorm(22),
            nn.Linear(22, max(16, self.part)),
            nn.GELU(),
            nn.Linear(max(16, self.part), 6),
        )
        self.hv_residual_bias = nn.Parameter(torch.zeros(self.pred_len, self.enc_in))
        self.station_value_bias = nn.Parameter(torch.zeros(self.station_count, self.enc_in))
        self.station_phase_bias = nn.Parameter(torch.zeros(self.station_count, self.enc_in, 3, 2))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)
        nn.init.zeros_(self.soft_memory_decoder[-1].weight)
        nn.init.zeros_(self.soft_memory_decoder[-1].bias)
        nn.init.zeros_(self.candidate_mixer[-1].weight)
        nn.init.zeros_(self.candidate_mixer[-1].bias)
        with torch.no_grad():
            self.variable_coupling.fill_(0.0)
            self.soft_memory_cross_residual_scale.fill_(0.05)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        exp_stage = kwargs.get("exp_stage", "train")
        bsz = x.shape[0]
        x_mask = torch.ones_like(x) if x_mask is None else x_mask.to(x)
        x = torch.nan_to_num(x) * x_mask
        pred_len = self.pred_len
        y_mask = torch.ones_like(y) if y_mask is None else y_mask.to(x)
        x_mark = self._ensure_marks(x_mark, x.shape[1], bsz, x.device, x.dtype)
        y_mark = self._ensure_marks(y_mark, pred_len, bsz, x.device, x.dtype, start=1.0)
        x_mark, y_mark = self._prepare_time_marks(x_mark, y_mark, pred_len)
        anchor = self.anchor(x, x_mask, pred_len)
        station_index = self._station_index_from_sample_id(kwargs.get("sample_ID"), x.device)
        anchor = self._apply_station_value_bias(anchor, station_index)
        anchor = self._apply_station_phase_bias(
            anchor, station_index, self._absolute_phase_features(y_mark)
        )
        stats = self._history_stats(x, x_mask, x_mark, y_mark, pred_len)
        event_state, query_state, event_mask, obs_logits, query_logits = (
            self._encode_observation_events(x, x_mask, x_mark, y_mark, stats, pred_len)
        )
        pred, candidates, weights = self._forecast_from_support(
            x,
            x_mask,
            x_mark,
            y_mark,
            anchor,
            stats,
            event_state,
            query_state,
            event_mask,
            obs_logits,
            query_logits,
        )
        y = y[:, :, : self.c_out]
        y_mask = y_mask[:, :, : self.c_out]
        output = {"pred": pred[:, :, : self.c_out], "true": y, "mask": y_mask}
        if exp_stage == "train":
            output["aux_loss"] = self._auxiliary_loss(
                obs_logits,
                query_logits[:, :, : self.c_out],
                x_mask,
                y,
                y_mask,
                candidates[:, :, : self.c_out],
                weights[:, :, : self.c_out],
            )
        return output

    def _forecast_from_support(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        anchor: Tensor,
        stats: dict[str, Tensor],
        event_state: Tensor,
        query_state: Tensor,
        event_mask: Tensor,
        obs_logits: Tensor,
        query_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores = self._role_scores(
            x, x_mask, x_mark, y_mark, stats, event_state, query_state, event_mask
        )
        top_indices, incidence_topk = self._incidence_from_scores(scores)
        role_states, role_values, role_extrapolated_values, numeric = self._readout_roles(
            x,
            x_mask,
            x_mark,
            y_mark,
            anchor,
            stats,
            event_state,
            query_state,
            top_indices,
            incidence_topk,
            query_logits,
        )
        horizon_context = self._horizon_context(
            anchor.shape[1], anchor.shape[2], anchor.shape[0], anchor.device, anchor.dtype
        )
        gate_features = torch.cat(
            [role_states.mean(dim=3), query_state + horizon_context, numeric], dim=-1
        )
        role_weights = torch.softmax(self.role_gate(gate_features), dim=-1)
        role_context = (role_weights.unsqueeze(-1) * role_states).sum(dim=3)
        local_value = (
            1.0 - self.extrapolate_weight
        ) * role_values + self.extrapolate_weight * role_extrapolated_values
        role_value = (role_weights * local_value).sum(dim=3)
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        bound = torch.maximum(
            scale * self.residual_bound_ratio, torch.full_like(scale, self.residual_bound_min)
        )
        role_value = anchor + self.value_readout_scale * (role_value - anchor).clamp(-bound, bound)
        periodic = self._periodic_candidate(
            role_value, role_values, role_extrapolated_values, anchor, stats
        )
        seasonal = self._seasonal_candidate(
            x,
            x_mask,
            x_mark,
            y_mark,
            role_value,
            role_values,
            role_extrapolated_values,
            anchor,
            stats,
        )
        process, _ = self._process_decay_candidate(
            role_value, role_values, role_extrapolated_values, role_weights, anchor, stats, numeric
        )
        seasonal_correction = self._seasonal_correction(
            role_value, seasonal, anchor, role_context, query_state, horizon_context, numeric, stats
        )
        memory = self._soft_memory_candidate(
            x,
            x_mask,
            x_mark,
            y_mark,
            anchor,
            role_value,
            stats,
            event_state,
            query_state,
            obs_logits,
            numeric,
        )
        mixed, weights, candidates = self._mix_prediction_candidates(
            anchor, role_value, memory, periodic, process, stats, numeric, role_weights
        )
        mixed = mixed + seasonal_correction
        features = torch.cat([role_context, query_state + horizon_context, numeric], dim=-1)
        raw = self.residual_decoder(features).squeeze(-1)
        support = numeric[..., 4].clamp(0.0, 1.0)
        residual = torch.tanh(raw) * bound * self.residual_scale * support
        residual = residual + self._horizon_variable_residual_bias(anchor, stats)
        pred = self._periodic_post_blend(mixed + residual, periodic)
        return (pred, candidates, weights)

    def _soft_memory_candidate(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        anchor: Tensor,
        role_value_readout: Tensor,
        stats: dict[str, Tensor],
        event_state: Tensor,
        query_state: Tensor,
        obs_logits: Tensor,
        numeric: Tensor,
    ) -> Tensor:
        bsz, seq_len, n_vars = x.shape
        pred_len = anchor.shape[1]
        obs_t = x_mark[:, :, :1].to(dtype=x.dtype).expand(-1, -1, n_vars)
        obs_t_var = obs_t.permute(0, 2, 1).reshape(bsz * n_vars, 1, seq_len)
        values = x.permute(0, 2, 1).reshape(bsz * n_vars, seq_len, 1)
        mask = x_mask.to(dtype=x.dtype).permute(0, 2, 1).reshape(bsz * n_vars, 1, seq_len)
        gate = torch.sigmoid(obs_logits).permute(0, 2, 1).reshape(bsz * n_vars, 1, seq_len)
        mask = mask * gate
        patch_ids = torch.arange(self.soft_memory_patches, device=x.device, dtype=x.dtype)
        base_width = 1.0 / float(self.soft_memory_patches)
        centers = (patch_ids + 0.5) * base_width
        left = centers.view(1, -1) - 0.5 * base_width + self.soft_memory_delta[:n_vars]
        width = torch.exp(self.soft_memory_log_width[:n_vars]).clamp_min(0.0001)
        right = left + width
        tau = F.softplus(self.soft_memory_tau[:n_vars]).view(n_vars, 1) + 0.0001
        left = (
            left.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * n_vars, self.soft_memory_patches, 1)
        )
        right = (
            right.unsqueeze(0)
            .expand(bsz, -1, -1)
            .reshape(bsz * n_vars, self.soft_memory_patches, 1)
        )
        tau = tau.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * n_vars, 1, 1)
        soft_window = torch.sigmoid((right - obs_t_var) / tau) * torch.sigmoid(
            (obs_t_var - left) / tau
        )
        patch_weights = soft_window * mask
        denom = patch_weights.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        patch_values = torch.bmm(patch_weights, values).squeeze(-1) / denom.squeeze(-1)
        patch_support = patch_weights.sum(dim=-1).view(bsz, n_vars, self.soft_memory_patches)
        patch_time = torch.bmm(patch_weights / denom, obs_t_var.transpose(1, 2)).squeeze(-1)
        patch_time = patch_time.view(bsz, n_vars, self.soft_memory_patches)
        phase_dim = min(3, x_mark.shape[-1] - 1, y_mark.shape[-1] - 1)
        if phase_dim > 0:
            obs_phase = x_mark[:, :, 1 : 1 + phase_dim].to(dtype=x.dtype)
            obs_phase = (
                obs_phase.unsqueeze(1)
                .expand(-1, n_vars, -1, -1)
                .reshape(bsz * n_vars, seq_len, phase_dim)
            )
            patch_phase = torch.bmm(patch_weights / denom, obs_phase)
            patch_phase = patch_phase.view(bsz, n_vars, self.soft_memory_patches, phase_dim)
            query_phase = y_mark[:, :pred_len, 1 : 1 + phase_dim].to(dtype=x.dtype)
        else:
            patch_phase = x.new_zeros(bsz, n_vars, self.soft_memory_patches, 0)
            query_phase = x.new_zeros(bsz, pred_len, 0)
        flat_event_state = event_state.view(bsz, seq_len, n_vars, self.d_model).permute(0, 2, 1, 3)
        flat_event_state = flat_event_state.reshape(bsz * n_vars, seq_len, self.d_model)
        patch_states = torch.bmm(patch_weights / denom, flat_event_state)
        patch_states = patch_states.view(bsz, n_vars, self.soft_memory_patches, self.d_model)
        patch_values = patch_values.view(bsz, n_vars, self.soft_memory_patches)
        (
            memory_context,
            memory_value,
            memory_cross_residual,
            _,
            memory_support,
            memory_same_support,
        ) = self._read_soft_memory(
            patch_states=patch_states,
            patch_values=patch_values,
            patch_support=patch_support,
            patch_time=patch_time,
            patch_phase=patch_phase,
            query_phase=query_phase,
            query_state=query_state,
            stats=stats,
        )
        query_gap = (stats["query_t"] - stats["last_t"].unsqueeze(1)).abs().clamp(0.0, 1.0)
        density = stats["density"].unsqueeze(1).expand_as(anchor).clamp(0.0, 1.0)
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        memory_features = torch.stack(
            [
                (memory_value - anchor) / scale,
                (memory_value - role_value_readout) / scale,
                query_gap,
                density,
                1.0 - density,
                stats["gap"][:, -1:, :].expand_as(anchor).clamp(0.0, 1.0),
                numeric[..., 4].clamp(0.0, 1.0),
                numeric[..., 5].clamp(0.0, 1.0),
                memory_support.clamp(0.0, 1.0),
                memory_same_support.clamp(0.0, 1.0),
                torch.sin(2.0 * math.pi * stats["query_t"]),
                torch.cos(2.0 * math.pi * stats["query_t"]),
            ],
            dim=-1,
        )
        raw = self.soft_memory_decoder(
            torch.cat([memory_context, query_state, memory_features], dim=-1)
        ).squeeze(-1)
        bound = torch.maximum(
            scale * self.soft_memory_bound_ratio, torch.full_like(scale, self.residual_bound_min)
        )
        support = (
            0.4 * numeric[..., 4].clamp(0.0, 1.0)
            + 0.4 * memory_support.clamp(0.0, 1.0)
            + 0.2 * memory_same_support.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        memory_shift = (memory_value - role_value_readout).clamp(min=-bound, max=bound)
        cross_residual_signal = memory_cross_residual.clamp(-3.0, 3.0)
        learned_shift = torch.tanh(raw + cross_residual_signal) * bound
        direct_ratio = (0.25 + 0.75 * memory_same_support.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        correction = (
            self.soft_memory_scale
            * support
            * (direct_ratio * memory_shift + (1.0 - direct_ratio) * learned_shift)
        )
        candidate = role_value_readout + correction
        return candidate

    def _read_soft_memory(
        self,
        patch_states: Tensor,
        patch_values: Tensor,
        patch_support: Tensor,
        patch_time: Tensor,
        patch_phase: Tensor,
        query_phase: Tensor,
        query_state: Tensor,
        stats: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsz, n_vars, n_patches, _ = patch_states.shape
        pred_len = query_state.shape[1]
        q_proj = self.query_proj(query_state)
        m_proj = self.event_proj(patch_states)
        q = q_proj.unsqueeze(3).unsqueeze(4)
        m = m_proj.unsqueeze(1).unsqueeze(1)
        memory_query = self.soft_memory_query[:n_vars].view(
            1, 1, 1, n_vars, n_patches, self.d_model
        )
        interaction = self.soft_memory_transport(
            hamilton_product(q + memory_query, quaternion_conjugate(m))
        )
        logits = interaction.mean(dim=-1) / math.sqrt(self.d_model)
        logits = logits + self._soft_memory_structural_logits(
            patch_support=patch_support,
            patch_time=patch_time,
            patch_phase=patch_phase,
            query_phase=query_phase,
            stats=stats,
        )
        valid = patch_support.view(bsz, 1, 1, n_vars, n_patches) > 1e-06
        logits = logits.masked_fill(~valid, -10000.0)
        weights = torch.softmax(logits.view(bsz, pred_len, n_vars, n_vars * n_patches), dim=-1)
        interaction_flat = interaction.view(bsz, pred_len, n_vars, n_vars * n_patches, self.d_model)
        values_flat = patch_values.view(bsz, 1, 1, n_vars * n_patches)
        valid_flat = valid.view(bsz, 1, 1, n_vars * n_patches).to(dtype=weights.dtype)
        memory_context = (weights.unsqueeze(-1) * interaction_flat).sum(dim=3)
        memory_value = (weights * values_flat).sum(dim=-1)
        memory_support = (weights * valid_flat).sum(dim=-1)
        source_weights = weights.view(bsz, pred_len, n_vars, n_vars, n_patches)
        memory_weights_out = source_weights.sum(dim=3)
        same_mask = torch.eye(n_vars, device=weights.device, dtype=weights.dtype).view(
            1, 1, n_vars, n_vars, 1
        )
        memory_same_support = (source_weights * same_mask).sum(dim=(3, 4))
        same_values = patch_values.view(bsz, 1, 1, n_vars, n_patches)
        same_value = (source_weights * same_mask * same_values).sum(dim=(3, 4))
        same_denom = memory_same_support.clamp_min(1e-06)
        cross_value = memory_value
        memory_value = torch.where(
            memory_same_support > 1e-06, same_value / same_denom, cross_value
        )
        source_mean = patch_values.mean(dim=2, keepdim=True)
        source_residual = patch_values - source_mean
        cross_mask = 1.0 - same_mask
        residual_scale = torch.tanh(self.soft_memory_cross_residual_scale[:n_vars, :n_vars])
        residual_scale = residual_scale.view(1, 1, n_vars, n_vars, 1)
        source_residual = source_residual.view(bsz, 1, 1, n_vars, n_patches)
        memory_cross_residual = (
            source_weights * cross_mask * residual_scale * source_residual
        ).sum(dim=(3, 4))
        return (
            memory_context,
            memory_value,
            memory_cross_residual,
            memory_weights_out,
            memory_support,
            memory_same_support,
        )

    def _soft_memory_structural_logits(
        self,
        patch_support: Tensor,
        patch_time: Tensor,
        patch_phase: Tensor,
        query_phase: Tensor,
        stats: dict[str, Tensor],
    ) -> Tensor:
        bsz, n_vars, n_patches = patch_support.shape
        density_logit = torch.log1p(patch_support).to(dtype=patch_time.dtype)
        query_t = stats["query_t"].to(dtype=patch_time.dtype)
        recency = -(
            query_t.unsqueeze(3).unsqueeze(4) - patch_time.view(bsz, 1, 1, n_vars, n_patches)
        ).abs()
        logits = 0.35 * recency + 0.2 * density_logit.view(bsz, 1, 1, n_vars, n_patches)
        query_ids = torch.arange(n_vars, device=patch_time.device)
        source_ids = torch.arange(n_vars, device=patch_time.device)
        same_var = (query_ids.view(n_vars, 1) == source_ids.view(1, n_vars)).to(
            dtype=patch_time.dtype
        )
        coupling = torch.tanh(
            self.variable_coupling[:n_vars, :n_vars]
            + self.soft_memory_source_bias[:n_vars, :n_vars]
        )
        logits = logits + self.soft_memory_same_var_prior * same_var.view(1, 1, n_vars, n_vars, 1)
        logits = logits + 0.25 * coupling.view(1, 1, n_vars, n_vars, 1)
        return logits + self.soft_memory_phase_scale * self._soft_memory_phase_affinity(
            patch_phase, query_phase, n_vars
        )

    def _soft_memory_phase_affinity(
        self,
        patch_phase: Tensor,
        query_phase: Tensor,
        n_query_vars: int,
    ) -> Tensor:
        bsz, source_vars, n_patches, phase_dim = patch_phase.shape
        pred_len = query_phase.shape[1]
        if phase_dim == 0:
            return patch_phase.new_zeros(bsz, pred_len, n_query_vars, source_vars, n_patches)
        diff = (
            query_phase.view(bsz, pred_len, 1, 1, phase_dim)
            - patch_phase.view(bsz, 1, source_vars, n_patches, phase_dim)
        ).abs()
        circular_dist = torch.minimum(diff, 1.0 - diff.clamp(0.0, 1.0))
        affinity = -circular_dist.mean(dim=-1)
        return affinity.unsqueeze(2).expand(-1, -1, n_query_vars, -1, -1)

    def _seasonal_correction(
        self,
        raw_role_value_readout: Tensor,
        seasonal_candidate: Tensor,
        anchor: Tensor,
        role_context: Tensor,
        query_state: Tensor,
        horizon_context: Tensor,
        numeric: Tensor,
        stats: dict[str, Tensor],
    ) -> Tensor:
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        features = torch.cat([role_context, query_state + horizon_context, numeric], dim=-1)
        gate = torch.sigmoid(self.seasonal_correction_gate(features)).squeeze(-1)
        target = (seasonal_candidate - raw_role_value_readout) / scale
        correction = gate * target
        correction = torch.tanh(correction) * scale
        return correction

    def _periodic_candidate(
        self,
        raw_role_value_readout: Tensor,
        role_values: Tensor,
        role_extrapolated_values: Tensor,
        anchor: Tensor,
        stats: dict[str, Tensor],
    ) -> Tensor:
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        periodic = 0.75 * role_values[..., 1] + 0.25 * role_extrapolated_values[..., 1]
        max_shift = torch.maximum(
            scale * self.residual_bound_ratio, torch.full_like(scale, self.residual_bound_min)
        )
        shift = (periodic - anchor).clamp(min=-max_shift, max=max_shift)
        return anchor + shift

    def _periodic_post_blend(self, pred: Tensor, periodic_candidate: Tensor) -> Tensor:
        if self.periodic_post_blend <= 0.0:
            return pred
        return (
            1.0 - self.periodic_post_blend
        ) * pred + self.periodic_post_blend * periodic_candidate

    def _seasonal_candidate(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        raw_role_value_readout: Tensor,
        role_values: Tensor,
        role_extrapolated_values: Tensor,
        anchor: Tensor,
        stats: dict[str, Tensor],
    ) -> Tensor:
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        out_dtype = anchor.dtype
        solve_dtype = (
            torch.float32 if anchor.dtype in (torch.float16, torch.bfloat16) else anchor.dtype
        )
        obs_basis = self._temporal_frequency_basis(x_mark, device=anchor.device, dtype=solve_dtype)
        query_basis = self._temporal_frequency_basis(
            y_mark[:, : anchor.shape[1], :], device=anchor.device, dtype=solve_dtype
        )
        obs_values = torch.nan_to_num(x.to(dtype=solve_dtype)) * x_mask.to(dtype=solve_dtype)
        obs_mask = x_mask.to(dtype=solve_dtype)
        basis_dim = obs_basis.shape[-1]
        reg = (
            torch.eye(basis_dim, device=anchor.device, dtype=solve_dtype).view(
                1, 1, basis_dim, basis_dim
            )
            * 0.001
        )
        ata = torch.einsum("blk,blv,blm->bvkm", obs_basis, obs_mask, obs_basis)
        atr = torch.einsum("blk,blv,blv->bvk", obs_basis, obs_mask, obs_values)
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            coeffs = torch.linalg.solve((ata + reg).float(), atr.unsqueeze(-1).float()).squeeze(-1)
        seasonal_memory = torch.einsum("bpk,bvk->bpv", query_basis, coeffs).to(dtype=out_dtype)
        obs_mask = obs_mask.to(dtype=out_dtype)
        density = obs_mask.mean(dim=1).clamp(0.0, 1.0)
        confidence = density.unsqueeze(1).expand_as(anchor)
        seasonal = raw_role_value_readout + confidence * (seasonal_memory - raw_role_value_readout)
        max_shift = torch.maximum(
            scale * self.residual_bound_ratio, torch.full_like(scale, self.residual_bound_min)
        )
        shift = (seasonal - anchor).clamp(min=-max_shift, max=max_shift)
        return anchor + shift

    def _process_decay_candidate(
        self,
        role_value_readout: Tensor,
        role_values: Tensor,
        role_extrapolated_values: Tensor,
        role_weights: Tensor,
        anchor: Tensor,
        stats: dict[str, Tensor],
        numeric: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.process_decay_scale <= 0.0:
            return (role_value_readout, torch.zeros_like(role_value_readout))
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        query_gap = (stats["query_t"] - stats["last_t"].unsqueeze(1)).abs().clamp(0.0, 1.0)
        density = stats["density"].unsqueeze(1).expand_as(anchor).clamp(0.0, 1.0)
        last_gap = stats["gap"][:, -1:, :].expand_as(anchor).clamp(0.0, 1.0)
        process_support = role_weights[..., 3].clamp(0.0, 1.0)
        process_delta = ((role_values[..., 3] - anchor) / scale).clamp(-6.0, 6.0)
        process_extrap_delta = ((role_extrapolated_values[..., 3] - anchor) / scale).clamp(
            -6.0, 6.0
        )
        readout_delta = ((role_value_readout - anchor) / scale).clamp(-6.0, 6.0)
        mean_target = stats["mean"].unsqueeze(1).expand_as(anchor)
        trend_target = stats["last"].unsqueeze(1).expand_as(anchor) + stats["slope"].unsqueeze(
            1
        ) * (stats["query_t"] - stats["last_t"].unsqueeze(1))
        decay_rate = self.process_decay_rate_scale * (
            0.25 + query_gap + last_gap + (1.0 - density) + process_support
        )
        decay = torch.exp(-decay_rate.clamp_min(0.0)).clamp(0.0, 1.0)
        process_target = 0.5 * mean_target + 0.5 * trend_target
        process_candidate = decay * role_value_readout + (1.0 - decay) * process_target
        gate_features = torch.stack(
            [
                query_gap,
                density,
                1.0 - density,
                last_gap,
                process_support,
                numeric[..., 4].clamp(0.0, 1.0),
                torch.sigmoid(numeric[..., 8]),
                readout_delta,
                process_delta,
                process_extrap_delta,
                numeric[..., 3].clamp(0.0, 1.0),
                (process_candidate - role_value_readout) / scale,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.process_decay_gate(torch.nan_to_num(gate_features)).squeeze(-1))
        gate = gate * self.process_decay_scale * process_support
        gate = gate.clamp(0.0, 1.0)
        process_candidate = role_value_readout + gate * (process_candidate - role_value_readout)
        return (process_candidate, gate)

    def _mix_prediction_candidates(
        self,
        anchor: Tensor,
        role_value_readout: Tensor,
        soft_memory_candidate: Tensor,
        periodic_candidate: Tensor,
        process_decay_candidate: Tensor,
        stats: dict[str, Tensor],
        numeric: Tensor,
        role_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        query_gap = (stats["query_t"] - stats["last_t"].unsqueeze(1)).abs().clamp(0.0, 1.0)
        density = stats["density"].unsqueeze(1).expand_as(anchor).clamp(0.0, 1.0)
        trend = stats["last"].unsqueeze(1).expand_as(anchor) + stats["slope"].unsqueeze(1) * (
            stats["query_t"] - stats["last_t"].unsqueeze(1)
        )
        trend = 0.7 * trend + 0.3 * stats["mean"].unsqueeze(1).expand_as(anchor)
        features = torch.stack(
            [
                query_gap,
                density,
                1.0 - density,
                stats["gap"][:, -1:, :].expand_as(anchor).clamp(0.0, 1.0),
                numeric[..., 4].clamp(0.0, 1.0),
                role_weights[..., 0],
                role_weights[..., 1],
                role_weights[..., 2],
                role_weights[..., 3],
                ((anchor - role_value_readout) / scale).clamp(-6.0, 6.0),
                ((soft_memory_candidate - role_value_readout) / scale).clamp(-6.0, 6.0),
                ((periodic_candidate - role_value_readout) / scale).clamp(-6.0, 6.0),
                ((process_decay_candidate - role_value_readout) / scale).clamp(-6.0, 6.0),
                (trend - role_value_readout).div(scale).clamp(-6.0, 6.0),
                numeric[..., 0].clamp(-6.0, 6.0),
                numeric[..., 2].clamp(0.0, 6.0),
                numeric[..., 3].clamp(0.0, 1.0),
                torch.sigmoid(numeric[..., 8]),
                torch.sin(2.0 * math.pi * stats["query_t"]),
                torch.cos(2.0 * math.pi * stats["query_t"]),
                torch.zeros_like(anchor),
                torch.zeros_like(anchor),
            ],
            dim=-1,
        )
        candidates = torch.stack(
            [
                anchor,
                role_value_readout,
                soft_memory_candidate,
                periodic_candidate,
                process_decay_candidate,
                trend,
            ],
            dim=-1,
        )
        prior = self._candidate_prior_logits(
            anchor,
            role_value_readout,
            soft_memory_candidate,
            periodic_candidate,
            process_decay_candidate,
            trend,
            stats,
            numeric,
            role_weights,
        )
        logits = self.candidate_mixer(torch.nan_to_num(features)) + prior
        if self.sixth_candidate_logit_bias != 0.0:
            logits = logits.clone()
            logits[..., 5] = logits[..., 5] + self.sixth_candidate_logit_bias
        weights = torch.softmax(logits, dim=-1)
        if self.sixth_candidate_max_mass < 1.0:
            sixth = weights[..., 5:6].clamp(max=self.sixth_candidate_max_mass)
            other = weights[..., :5]
            remaining = (1.0 - sixth).clamp_min(0.0)
            weights = torch.cat(
                [other / other.sum(-1, keepdim=True).clamp_min(1e-08) * remaining, sixth], -1
            )
        return ((weights * candidates).sum(-1), weights, candidates)

    def _candidate_prior_logits(
        self,
        anchor: Tensor,
        role_value_readout: Tensor,
        soft_memory_candidate: Tensor,
        periodic_candidate: Tensor,
        process_decay_candidate: Tensor,
        fifth_candidate: Tensor,
        stats: dict[str, Tensor],
        numeric: Tensor,
        role_weights: Tensor,
    ) -> Tensor:
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        density = stats["density"].unsqueeze(1).expand_as(anchor).clamp(0.0, 1.0)
        sparse = 1.0 - density
        support = numeric[..., 4].clamp(0.0, 1.0)
        last_gap = stats["gap"][:, -1:, :].expand_as(anchor).clamp(0.0, 1.0)
        zero_fraction = stats["zero_fraction"].unsqueeze(1).expand_as(anchor).clamp(0.0, 1.0)
        tail_delta = (
            (
                stats["tail_value"].unsqueeze(1).expand_as(anchor)
                - stats["mean"].unsqueeze(1).expand_as(anchor)
            )
            / scale
        ).clamp(0.0, 6.0)
        eventness = (zero_fraction * (tail_delta / 3.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        deltas = torch.stack(
            [
                (anchor - role_value_readout).abs() / scale,
                torch.zeros_like(anchor),
                (soft_memory_candidate - role_value_readout).abs() / scale,
                (periodic_candidate - role_value_readout).abs() / scale,
                (process_decay_candidate - role_value_readout).abs() / scale,
                (fifth_candidate - role_value_readout).abs() / scale,
            ],
            dim=-1,
        ).clamp(0.0, 6.0)
        proximity = -0.35 * deltas
        fifth_evidence = 0.25 * sparse + 0.25 * last_gap
        role_evidence = torch.stack(
            [
                0.7 * sparse + 0.2 * last_gap,
                0.45 * role_weights[..., 0] + 0.35 * support + 0.2 * density,
                0.55 * support + 0.35 * sparse + 0.25 * role_weights[..., 2],
                1.2 * role_weights[..., 1] + 0.25 * density + 0.25 * eventness,
                1.0 * role_weights[..., 3] + 0.35 * sparse + 0.2 * last_gap,
                fifth_evidence,
            ],
            dim=-1,
        )
        return self.candidate_prior_scale * (role_evidence + proximity)

    def _station_index_from_sample_id(
        self, sample_id: Tensor | None, device: torch.device
    ) -> Tensor | None:
        if self.station_time_steps <= 0 or sample_id is None:
            return None
        return (sample_id.to(device=device, dtype=torch.long) // self.station_time_steps).clamp(
            min=0, max=self.station_count - 1
        )

    def _apply_station_value_bias(self, anchor: Tensor, station_index: Tensor | None) -> Tensor:
        if (
            not self.station_value_bias_enabled
            or self.station_value_bias_scale <= 0.0
            or station_index is None
        ):
            return anchor
        bias = torch.tanh(
            self.station_value_bias[station_index, : anchor.shape[2]].to(dtype=anchor.dtype)
        )
        return anchor + self.station_value_bias_scale * bias.unsqueeze(1)

    def _absolute_phase_features(self, y_mark: Tensor) -> Tensor:
        n_vars = self.enc_in
        if y_mark.shape[-1] < 4:
            return y_mark.new_zeros(y_mark.shape[0], y_mark.shape[1], n_vars, 3, 2)
        phase_components = []
        for idx in range(min(self.station_phase_components, 3)):
            phase = y_mark[..., idx + 1].clamp(0.0, 1.0)
            phase = phase.unsqueeze(-1).expand(-1, -1, n_vars)
            phase_components.append(
                torch.stack(
                    [torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)], dim=-1
                )
            )
        while len(phase_components) < 3:
            phase_components.append(y_mark.new_zeros(y_mark.shape[0], y_mark.shape[1], n_vars, 2))
        return torch.stack(phase_components[:3], dim=-2)

    def _apply_station_phase_bias(
        self, anchor: Tensor, station_index: Tensor | None, station_phase: Tensor | None
    ) -> Tensor:
        if (
            not self.station_phase_bias_enabled
            or self.station_phase_bias_scale <= 0.0
            or station_index is None
            or (station_phase is None)
        ):
            return anchor
        phase = station_phase.to(dtype=anchor.dtype)
        bias = self.station_phase_bias[station_index, : anchor.shape[2], : phase.shape[-2], :].to(
            dtype=anchor.dtype
        )
        phase_bias = (bias.unsqueeze(1) * phase).sum(dim=(-2, -1))
        return anchor + self.station_phase_bias_scale * torch.tanh(phase_bias)

    def _horizon_variable_residual_bias(self, anchor: Tensor, stats: dict[str, Tensor]) -> Tensor:
        if self.hv_residual_bias_scale <= 0.0:
            return torch.zeros_like(anchor)
        bias = self.hv_residual_bias[: anchor.shape[1], : anchor.shape[2]].view(
            1, anchor.shape[1], anchor.shape[2]
        )
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        bound = torch.maximum(
            scale * self.residual_bound_ratio, torch.full_like(scale, self.residual_bound_min)
        )
        return torch.tanh(bias) * bound * self.hv_residual_bias_scale

    def _circular_embedding_period_vector(
        self, n_vars: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        periods = torch.zeros(n_vars, device=device, dtype=dtype)
        for idx in self.circular_embedding_variables:
            if idx < n_vars and idx in self.circular_embedding_periods:
                periods[idx] = float(self.circular_embedding_periods[idx])
        return periods

    def _circular_embedding_indicator(
        self, batch: int, length: int, n_vars: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        mask = (self._circular_embedding_period_vector(n_vars, device, dtype) > 0.0).to(dtype=dtype)
        return mask.view(1, 1, n_vars).expand(batch, length, n_vars)

    def _mixed_value_embedding(
        self, features: Tensor, linear_layer: nn.Module, circular_layer: nn.Module
    ) -> Tensor:
        base = linear_layer(features)
        if not self.circular_embedding_variables or not self.circular_embedding_periods:
            return base
        circular = self._circular_embedding_indicator(
            batch=features.shape[0],
            length=features.shape[1],
            n_vars=features.shape[2],
            device=features.device,
            dtype=features.dtype,
        ).unsqueeze(-1)
        return torch.where(circular > 0.0, circular_layer(features), base)

    def _role_scores(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        stats: dict[str, Tensor],
        event_state: Tensor,
        query_state: Tensor,
        event_mask: Tensor,
    ) -> Tensor:
        bsz, seq_len, n_vars = x.shape
        pred_len = y_mark.shape[1]
        obs_t = x_mark[:, :, :1].to(dtype=x.dtype).expand(-1, -1, n_vars)
        obs_t_view = obs_t.view(bsz, 1, 1, seq_len, n_vars)
        query_t = y_mark[:, :pred_len, :1].to(dtype=x.dtype).view(bsz, pred_len, 1, 1, 1)
        recency = 1.0 - (query_t - obs_t_view).abs().clamp(0.0, 1.0)
        obs_phase, query_phase = self._periodic_phase_pair(
            x_mark=x_mark, y_mark=y_mark, pred_len=pred_len
        )
        obs_phase = (
            obs_phase.unsqueeze(2)
            .view(bsz, 1, 1, seq_len, 1, -1)
            .expand(-1, -1, -1, -1, n_vars, -1)
        )
        query_phase = query_phase.view(bsz, pred_len, 1, 1, 1, -1)
        phase_affinity = -self._periodic_distance(query_phase=query_phase, obs_phase=obs_phase)
        obs_vars = torch.arange(n_vars, device=x.device).view(1, 1, 1, 1, n_vars)
        query_vars = torch.arange(n_vars, device=x.device).view(1, 1, n_vars, 1, 1)
        same_var = (query_vars == obs_vars).to(dtype=x.dtype)
        cross_var = 1.0 - same_var
        coupling = torch.tanh(self.variable_coupling[:n_vars, :n_vars])
        coupling = coupling.view(1, 1, n_vars, 1, n_vars)
        reliability = event_mask.view(bsz, 1, 1, seq_len, n_vars)
        gap = stats["gap"].view(bsz, 1, 1, seq_len, n_vars)
        source_scale = stats["scale"].view(bsz, 1, 1, 1, n_vars).clamp_min(0.001)
        local_mag = (
            stats["local_delta"].abs().view(bsz, 1, 1, seq_len, n_vars) / source_scale
        ).clamp_max(6.0)
        filled = stats["filled"].view(bsz, 1, 1, seq_len, n_vars)
        target_last = stats["last"].view(bsz, 1, n_vars, 1, 1)
        target_scale = stats["scale"].view(bsz, 1, n_vars, 1, 1).clamp_min(0.001)
        value_gap = ((filled - target_last).abs() / target_scale).clamp_max(6.0)
        if self.circular_embedding_variables and self.circular_embedding_periods:
            period = self._circular_embedding_period_vector(n_vars, x.device, x.dtype).view(
                1, 1, 1, 1, n_vars
            )
            circular_mask = period > 0.0
            safe_period = torch.where(circular_mask, period, torch.ones_like(period))
            source_angle = 2.0 * math.pi * torch.remainder(filled, safe_period) / safe_period
            target_angle = 2.0 * math.pi * torch.remainder(target_last, safe_period) / safe_period
            angular_gap = torch.atan2(
                torch.sin(source_angle - target_angle), torch.cos(source_angle - target_angle)
            ).abs()
            circular_gap = (angular_gap / math.pi * 3.0).clamp_max(6.0)
            value_gap = torch.where(circular_mask, circular_gap, value_gap)
        value_affinity = -value_gap
        role_scores = torch.stack(
            [
                2.0 * reliability + 1.8 * same_var + 1.5 * recency + 0.2 * value_affinity,
                2.0 * reliability + 1.4 * same_var + 1.5 * phase_affinity + 0.5 * recency,
                1.8 * reliability
                + 1.4 * cross_var
                + 0.8 * recency
                + coupling
                + 0.2 * phase_affinity,
                1.8 * reliability + 0.9 * local_mag - 0.8 * gap + 0.6 * recency + 0.2 * same_var,
            ],
            dim=3,
        )
        role_scores = role_scores + self.role_code.mean(dim=-1).view(1, 1, 1, self.N_ROLES, 1, 1)
        query_proj = self.query_proj(query_state).unsqueeze(3).unsqueeze(4)
        event_proj = self.event_proj(event_state).view(bsz, 1, 1, 1, seq_len * n_vars, self.d_model)
        role_code = self.role_code.view(1, 1, 1, self.N_ROLES, 1, self.d_model).to(
            device=x.device, dtype=x.dtype
        )
        learned = hamilton_product(query_proj + role_code, quaternion_conjugate(event_proj))[
            ..., : self.part
        ]
        learned = learned.mean(dim=-1) / math.sqrt(self.part)
        role_scores = role_scores.reshape(bsz, pred_len, n_vars, self.N_ROLES, seq_len * n_vars)
        role_scores = role_scores + learned
        role_scores = role_scores.masked_fill(
            event_mask.view(bsz, 1, 1, 1, seq_len * n_vars) <= 0.0, -10000.0
        )
        return role_scores

    def _incidence_from_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        if self.topk >= scores.shape[-1]:
            indices = torch.arange(scores.shape[-1], device=scores.device)
            indices = indices.view(1, 1, 1, 1, -1).expand_as(scores)
            return (indices, torch.softmax(scores, dim=-1))
        values, indices = torch.topk(scores, k=self.topk, dim=-1)
        return (indices, torch.softmax(values, dim=-1))

    def _readout_roles(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        anchor: Tensor,
        stats: dict[str, Tensor],
        event_state: Tensor,
        query_state: Tensor,
        top_indices: Tensor,
        incidence_topk: Tensor,
        query_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bsz, seq_len, n_vars = x.shape
        pred_len = anchor.shape[1]
        topk = top_indices.shape[-1]
        flat_index = top_indices.reshape(bsz, -1)
        selected_state = torch.gather(
            event_state, dim=1, index=flat_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        ).view(bsz, pred_len, n_vars, self.N_ROLES, topk, self.d_model)
        selected_values = torch.gather(x.reshape(bsz, -1), dim=1, index=flat_index)
        selected_values = selected_values.view(bsz, pred_len, n_vars, self.N_ROLES, topk)
        selected_mask = torch.gather(
            x_mask.to(dtype=x.dtype).reshape(bsz, -1), dim=1, index=flat_index
        )
        selected_mask = selected_mask.view(bsz, pred_len, n_vars, self.N_ROLES, topk)
        obs_t_flat = x_mark[:, :, :1].to(dtype=x.dtype).expand(-1, -1, n_vars).reshape(bsz, -1)
        selected_time = torch.gather(obs_t_flat, dim=1, index=flat_index).view(
            bsz, pred_len, n_vars, self.N_ROLES, topk
        )
        obs_vars_flat = (
            torch.arange(n_vars, device=x.device).view(1, n_vars).expand(seq_len, -1).reshape(1, -1)
        )
        selected_vars = torch.gather(obs_vars_flat.expand(bsz, -1), dim=1, index=flat_index)
        selected_vars = selected_vars.view(bsz, pred_len, n_vars, self.N_ROLES, topk)
        query_vars = torch.arange(n_vars, device=x.device).view(1, 1, n_vars, 1, 1)
        selected_same_var = (selected_vars == query_vars).to(dtype=x.dtype)
        slope_flat = (
            stats["slope"].view(bsz, 1, 1, 1, n_vars).expand(-1, pred_len, n_vars, self.N_ROLES, -1)
        )
        selected_slope = torch.gather(slope_flat, dim=-1, index=selected_vars)
        hyperedge_state = (incidence_topk.unsqueeze(-1) * selected_state).sum(dim=-2)
        query_proj = self.query_proj(query_state).unsqueeze(3).expand_as(hyperedge_state)
        role_proj = self.event_proj(hyperedge_state)
        role_states = self.role_transport(
            hamilton_product(query_proj, quaternion_conjugate(role_proj))
        )
        event_center = (incidence_topk * selected_values).sum(dim=-1)
        query_t_for_extrap = y_mark[:, :pred_len, :1].to(dtype=x.dtype).view(bsz, pred_len, 1, 1, 1)
        selected_extrapolated = selected_values + selected_slope * (
            query_t_for_extrap - selected_time
        )
        extrapolated_center = (incidence_topk * selected_extrapolated).sum(dim=-1)
        role_value = event_center.mean(dim=3)
        last = stats["last"].view(bsz, 1, n_vars, 1, 1)
        selected_delta = selected_values - last
        scale = stats["scale"].unsqueeze(1).expand_as(anchor).clamp_min(0.001)
        role_support = (incidence_topk * selected_mask).sum(dim=-1).sum(dim=3) / self.N_ROLES
        numeric = torch.stack(
            [
                (role_value - anchor) / scale,
                (incidence_topk * selected_delta).sum(dim=-1).mean(dim=3) / scale,
                (incidence_topk * selected_delta.abs()).sum(dim=-1).mean(dim=3) / scale,
                (incidence_topk * (query_t_for_extrap - selected_time).abs())
                .sum(dim=-1)
                .mean(dim=3),
                role_support,
                (incidence_topk * selected_same_var).sum(dim=-1).mean(dim=3),
                stats["density"].unsqueeze(1).expand_as(anchor),
                stats["gap"][:, -1:, :].expand_as(anchor),
                torch.sigmoid(query_logits),
                stats["slope"].unsqueeze(1) / scale,
            ],
            dim=-1,
        )
        return (
            role_states,
            event_center,
            extrapolated_center,
            torch.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _encode_observation_events(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        stats: dict[str, Tensor],
        pred_len: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        bsz, seq_len, n_vars = x.shape
        scale = stats["scale"].unsqueeze(1).clamp_min(0.001)
        filled = stats["filled"]
        last = stats["last"].unsqueeze(1)
        mean = stats["mean"].unsqueeze(1)
        density = stats["density"].unsqueeze(1).expand(-1, seq_len, -1)
        gap = stats["gap"]
        local_delta = stats["local_delta"]
        obs_t = x_mark[:, :, :1].expand(-1, -1, n_vars).to(dtype=x.dtype)
        node_value_features = torch.stack(
            [(filled - mean) / scale, (filled - last) / scale, local_delta / scale, x_mask], dim=-1
        )
        if self.circular_embedding_variables and self.circular_embedding_periods:
            period = self._circular_embedding_period_vector(n_vars, x.device, x.dtype).view(
                1, 1, n_vars
            )
            circular = period > 0.0
            safe_period = torch.where(circular, period, torch.ones_like(period))
            filled_angle = 2.0 * math.pi * torch.remainder(filled, safe_period) / safe_period
            last_angle = (
                2.0 * math.pi * torch.remainder(last.expand_as(filled), safe_period) / safe_period
            )
            circular_delta = torch.atan2(
                torch.sin(filled_angle - last_angle), torch.cos(filled_angle - last_angle)
            )
            circular_value_features = torch.stack(
                [
                    torch.sin(filled_angle),
                    torch.cos(filled_angle),
                    circular_delta / math.pi,
                    x_mask,
                ],
                dim=-1,
            )
            node_value_features = torch.where(
                circular.unsqueeze(-1), circular_value_features, node_value_features
            )
        node_phase_features = torch.stack(
            [obs_t, torch.sin(2.0 * math.pi * obs_t), torch.cos(2.0 * math.pi * obs_t), gap], dim=-1
        )
        node_rel_features = torch.stack([x_mask, density, 1.0 - density, gap], dim=-1)
        obs_intensity_features = torch.cat([node_phase_features, node_rel_features], dim=-1)
        obs_logits = self.obs_intensity_net(obs_intensity_features).squeeze(-1)
        obs_gate = self.intensity_floor + (1.0 - self.intensity_floor) * torch.sigmoid(obs_logits)
        var_ids = torch.arange(n_vars, device=x.device).clamp_max(self.enc_in - 1)
        var_embed = (
            self.variable_embedding(var_ids)
            .view(1, 1, n_vars, self.part)
            .expand(bsz, seq_len, -1, -1)
        )
        event_state = torch.cat(
            [
                self._mixed_value_embedding(
                    node_value_features, self.node_value, self.node_circular_value
                ),
                self.node_phase(node_phase_features),
                var_embed,
                self.node_reliability(node_rel_features),
            ],
            dim=-1,
        )
        event_mask = (x_mask * obs_gate).reshape(bsz, seq_len * n_vars)
        event_state = (event_state * event_mask.view(bsz, seq_len, n_vars, 1)).reshape(
            bsz, seq_len * n_vars, self.d_model
        )
        horizon = torch.linspace(0.0, 1.0, pred_len, device=x.device, dtype=x.dtype).view(
            1, pred_len, 1
        )
        horizon = horizon.expand(bsz, -1, n_vars)
        query_delta = stats["query_t"] - stats["last_t"].unsqueeze(1)
        query_value_features = torch.stack(
            [
                (last.expand(-1, pred_len, -1) - stats["mean"].unsqueeze(1))
                / stats["scale"].unsqueeze(1).clamp_min(0.001),
                stats["slope"].unsqueeze(1)
                * query_delta
                / stats["scale"].unsqueeze(1).clamp_min(0.001),
                horizon,
                stats["density"].unsqueeze(1).expand(-1, pred_len, -1),
            ],
            dim=-1,
        )
        if self.circular_embedding_variables and self.circular_embedding_periods:
            period = self._circular_embedding_period_vector(n_vars, x.device, x.dtype).view(
                1, 1, n_vars
            )
            circular = period > 0.0
            safe_period = torch.where(circular, period, torch.ones_like(period))
            last_expanded = last.expand(-1, pred_len, -1)
            query_linear = last_expanded + stats["slope"].unsqueeze(1) * query_delta
            last_angle = 2.0 * math.pi * torch.remainder(last_expanded, safe_period) / safe_period
            query_angle = 2.0 * math.pi * torch.remainder(query_linear, safe_period) / safe_period
            angular_delta = torch.atan2(
                torch.sin(query_angle - last_angle), torch.cos(query_angle - last_angle)
            )
            circular_query_features = torch.stack(
                [
                    torch.sin(last_angle),
                    torch.cos(last_angle),
                    angular_delta / math.pi,
                    stats["density"].unsqueeze(1).expand(-1, pred_len, -1),
                ],
                dim=-1,
            )
            query_value_features = torch.where(
                circular.unsqueeze(-1), circular_query_features, query_value_features
            )
        query_t = y_mark[:, :pred_len, :1].expand(-1, -1, n_vars).to(dtype=x.dtype)
        query_phase_features = torch.stack(
            [
                query_t,
                torch.sin(2.0 * math.pi * query_t),
                torch.cos(2.0 * math.pi * query_t),
                horizon,
            ],
            dim=-1,
        )
        query_rel_features = torch.stack(
            [
                stats["density"].unsqueeze(1).expand(-1, pred_len, -1),
                1.0 - stats["density"].unsqueeze(1).expand(-1, pred_len, -1),
                stats["gap"][:, -1:, :].expand(-1, pred_len, -1),
                query_delta.abs().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        query_logits = self.query_intensity_net(
            torch.cat([query_phase_features, query_rel_features], dim=-1)
        ).squeeze(-1)
        query_var = (
            self.variable_embedding(var_ids)
            .view(1, 1, n_vars, self.part)
            .expand(bsz, pred_len, -1, -1)
        )
        query_state = torch.cat(
            [
                self._mixed_value_embedding(
                    query_value_features, self.query_value, self.query_circular_value
                ),
                self.query_phase(query_phase_features),
                query_var,
                self.query_reliability(query_rel_features),
            ],
            dim=-1,
        )
        return (event_state, query_state, event_mask, obs_logits, query_logits)

    def _history_stats(
        self, x: Tensor, x_mask: Tensor, x_mark: Tensor, y_mark: Tensor, pred_len: int
    ) -> dict[str, Tensor]:
        mask = x_mask.to(dtype=x.dtype)
        filled = CausalAnchor._forward_fill(x, x_mask)
        last = CausalAnchor._last_observed(x, x_mask)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (x * mask).sum(dim=1) / count
        centered = (x - mean.unsqueeze(1)) * mask
        scale = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0.0001))
        density = count / max(x.shape[1], 1)
        masked_for_tail = torch.where(mask > 0.0, x, torch.full_like(x, -1000000.0))
        tail_k = max(1, min(x.shape[1], int(math.ceil(0.1 * x.shape[1]))))
        tail_values = torch.topk(masked_for_tail, k=tail_k, dim=1).values
        tail_valid = tail_values > -100000.0
        tail_count = tail_valid.to(dtype=x.dtype).sum(dim=1).clamp_min(1.0)
        tail_value = (
            torch.where(tail_valid, tail_values, torch.zeros_like(tail_values)).sum(dim=1)
            / tail_count
        )
        near_zero = ((x.abs() <= 0.05 * scale.unsqueeze(1).clamp_min(0.001)) & (mask > 0.0)).to(
            dtype=x.dtype
        )
        zero_fraction = near_zero.sum(dim=1) / count
        gap = self._gap_feature(x_mask)
        local_delta = torch.zeros_like(filled)
        local_delta[:, 1:] = filled[:, 1:] - filled[:, :-1]
        t = x_mark[:, :, :1].expand(-1, -1, x.shape[2]).to(dtype=x.dtype)
        t_mean = (t * mask).sum(dim=1) / count
        cov = ((t - t_mean.unsqueeze(1)) * (x - mean.unsqueeze(1)) * mask).sum(dim=1)
        var = ((t - t_mean.unsqueeze(1)).square() * mask).sum(dim=1).clamp_min(0.0001)
        slope = cov / var
        last_t = self._last_time(x_mark=x_mark, x_mask=x_mask)
        query_t = y_mark[:, :pred_len, :1].to(dtype=x.dtype).expand(-1, -1, x.shape[2])
        return {
            "filled": filled,
            "last": last,
            "mean": mean,
            "scale": scale,
            "density": density,
            "tail_value": tail_value,
            "zero_fraction": zero_fraction,
            "gap": gap,
            "local_delta": local_delta,
            "slope": slope,
            "last_t": last_t,
            "query_t": query_t,
        }

    def _auxiliary_loss(
        self,
        obs_logits: Tensor,
        query_logits: Tensor,
        x_mask: Tensor,
        y: Tensor,
        y_mask: Tensor,
        candidates: Tensor,
        weights: Tensor,
    ) -> Tensor:
        obs_loss = F.binary_cross_entropy_with_logits(
            obs_logits, x_mask.to(obs_logits), reduction="mean"
        )
        query_loss = F.binary_cross_entropy_with_logits(
            query_logits, y_mask.to(query_logits), reduction="mean"
        )
        losses = [self.obs_aux_weight * (obs_loss + 0.5 * query_loss)]
        if self.candidate_oracle_aux_weight > 0.0 or self.candidate_margin_aux_weight > 0.0:
            log_probs = torch.log(weights.clamp_min(1e-08))
            error = (candidates.detach() - y.unsqueeze(-1)).square()
            scale = (candidates[..., 0].detach() - candidates[..., 1].detach()).abs().clamp_min(1.0)
            normalized_error = error / scale.unsqueeze(-1).square()
            horizon_weight = 1.0
            if self.candidate_oracle_horizon_gamma > 0.0:
                horizon = torch.linspace(0.0, 1.0, y.shape[1], device=y.device, dtype=y.dtype)
                horizon_weight = 1.0 + self.candidate_oracle_horizon_gamma * horizon.view(1, -1, 1)
            weighted_mask = y_mask * horizon_weight
            if self.candidate_oracle_aux_weight > 0.0:
                oracle = torch.softmax(-error / self.candidate_oracle_temperature, dim=-1)
                cross_entropy = -(oracle * log_probs).sum(-1)
                losses.append(
                    self.candidate_oracle_aux_weight
                    * (cross_entropy * weighted_mask).sum()
                    / weighted_mask.sum().clamp_min(1.0)
                )
            if self.candidate_margin_aux_weight > 0.0:
                top2 = torch.topk(normalized_error, k=2, dim=-1, largest=False)
                margin = (top2.values[..., 1] - top2.values[..., 0]).clamp_min(0.0)
                confident = (margin >= self.candidate_margin_threshold).to(y)
                hard_ce = -torch.gather(log_probs, -1, top2.indices[..., :1]).squeeze(-1)
                hard_mask = weighted_mask * confident
                losses.append(
                    self.candidate_margin_aux_weight
                    * (hard_ce * hard_mask).sum()
                    / hard_mask.sum().clamp_min(1.0)
                )
        return torch.stack(losses).sum()

    def _horizon_context(
        self, pred_len: int, n_vars: int, batch: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if not self.use_horizon_context:
            return torch.zeros(batch, pred_len, n_vars, self.d_model, device=device, dtype=dtype)
        ids = torch.arange(pred_len, device=device).clamp_max(self.pred_len - 1)
        ctx = self.horizon_embedding(ids).to(dtype=dtype)
        return ctx.view(1, pred_len, 1, self.d_model).expand(batch, -1, n_vars, -1)

    @staticmethod
    def _ensure_marks(
        marks: Tensor | None,
        length: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        start: float = 0.0,
    ) -> Tensor:
        if marks is not None:
            marks = marks.to(device=device, dtype=dtype)
            if marks.ndim == 2:
                marks = marks.unsqueeze(-1)
            if marks.shape[0] == 1 and batch > 1:
                marks = marks.expand(batch, -1, -1)
            return marks[:, :length, :]
        denom = max(length - 1, 1)
        base = torch.arange(length, device=device, dtype=dtype).view(1, length, 1) / denom
        return (base + start).expand(batch, -1, -1)

    def _prepare_time_marks(
        self, x_mark: Tensor, y_mark: Tensor, pred_len: int
    ) -> tuple[Tensor, Tensor]:
        if self.time_mark_mode == "input_first":
            return (x_mark, y_mark)
        history_len = x_mark.shape[1]
        total_len = history_len + pred_len
        relative = torch.arange(total_len, device=x_mark.device, dtype=x_mark.dtype) / max(
            total_len - 1, 1
        )
        relative = relative.view(1, total_len, 1).expand(x_mark.shape[0], -1, -1)
        x_relative = relative[:, :history_len, :]
        y_relative = relative[:, history_len:, :]
        return (
            torch.cat([x_relative, x_mark], dim=-1),
            torch.cat([y_relative, y_mark[:, :pred_len, :]], dim=-1),
        )

    @staticmethod
    def _gap_feature(mask: Tensor) -> Tensor:
        length = mask.shape[1]
        denom = max(mask.shape[1] - 1, 1)
        positions = torch.arange(length, device=mask.device).view(1, length, 1)
        observed_positions = torch.where(mask > 0, positions, positions.new_full((), -1))
        last_observed = observed_positions.cummax(dim=1).values
        gap_steps = positions - last_observed
        return (gap_steps.to(dtype=mask.dtype) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _last_time(x_mark: Tensor, x_mask: Tensor) -> Tensor:
        bsz, _, n_vars = x_mask.shape
        length = x_mask.shape[1]
        positions = torch.arange(length, device=x_mask.device).view(1, length, 1)
        observed_positions = torch.where(x_mask > 0, positions, positions.new_full((), -1))
        last_index = observed_positions.cummax(dim=1).values[:, -1, :]
        obs_t = x_mark[:, :, :1].expand(-1, -1, n_vars)
        gathered = obs_t.gather(dim=1, index=last_index.clamp_min(0).unsqueeze(1)).squeeze(1)
        return torch.where(
            last_index >= 0,
            gathered,
            torch.zeros(bsz, n_vars, device=x_mask.device, dtype=x_mark.dtype),
        )

    @staticmethod
    def _periodic_distance(query_phase: Tensor, obs_phase: Tensor) -> Tensor:
        sin_gap = torch.sin(2.0 * math.pi * query_phase) - torch.sin(2.0 * math.pi * obs_phase)
        cos_gap = torch.cos(2.0 * math.pi * query_phase) - torch.cos(2.0 * math.pi * obs_phase)
        return (sin_gap.square() + cos_gap.square()).mean(dim=-1)

    @staticmethod
    def _periodic_phase_pair(
        x_mark: Tensor, y_mark: Tensor, pred_len: int
    ) -> tuple[Tensor, Tensor]:
        if x_mark.shape[-1] > 1 and y_mark.shape[-1] > 1:
            mark_dims = min(x_mark.shape[-1], y_mark.shape[-1])
            return (x_mark[:, :, 1:mark_dims], y_mark[:, :pred_len, 1:mark_dims])
        return (x_mark[:, :, :1], y_mark[:, :pred_len, :1])

    @staticmethod
    def _temporal_frequency_basis(
        marks: Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if marks.ndim != 3:
            raise ValueError(f"Expected marks with shape [B, L, D], got {tuple(marks.shape)}")
        if marks.shape[-1] <= 1:
            rel = marks[:, :, :1].to(device=device, dtype=dtype)
            return torch.cat([torch.ones_like(rel), rel], dim=-1)
        phase_marks = marks[:, :, 1:].to(device=device, dtype=dtype)
        harmonic_schedule = (1, 3, 2, 1, 1)
        features = [torch.ones_like(phase_marks[:, :, :1])]
        for idx in range(phase_marks.shape[-1]):
            phase = phase_marks[:, :, idx : idx + 1]
            n_harmonics = harmonic_schedule[idx] if idx < len(harmonic_schedule) else 1
            for harmonic in range(1, n_harmonics + 1):
                angle = 2.0 * math.pi * harmonic * phase
                features.append(torch.sin(angle))
                features.append(torch.cos(angle))
        return torch.cat(features, dim=-1)
