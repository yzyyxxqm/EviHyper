from dataclasses import asdict, dataclass


@dataclass
class EviHyperConfig:
    seq_len: int = 48
    pred_len: int = 24
    enc_in: int = 5
    c_out: int | None = None
    d_model: int = 64
    d_ff: int = 128
    dropout: float = 0.05
    topk: int = 12
    time_mark_mode: str = "input_first"
    horizon_context: bool = False
    intensity_floor: float = 0.05
    extrapolate_weight: float = 0.0
    value_readout_scale: float = 0.5
    residual_scale: float = 1.0
    residual_bound_ratio: float = 1.0
    residual_bound_min: float = 0.03
    hv_residual_bias_scale: float = 0.25
    soft_memory_patches: int = 6
    soft_memory_scale: float = 0.5
    soft_memory_bound_ratio: float = 0.5
    soft_memory_phase_scale: float = 0.6
    soft_memory_same_var_prior: float = 0.4
    process_decay_scale: float = 1.0
    process_decay_rate_scale: float = 1.0
    periodic_post_blend: float = 0.0
    candidate_prior_scale: float = 1.0
    sixth_candidate_logit_bias: float = 0.0
    sixth_candidate_max_mass: float = 1.0
    obs_aux_weight: float = 0.02
    candidate_oracle_aux_weight: float = 0.0
    candidate_oracle_temperature: float = 0.5
    candidate_oracle_horizon_gamma: float = 0.0
    candidate_margin_aux_weight: float = 0.0
    candidate_margin_threshold: float = 0.05
    circular_embedding_variables: str = ""
    circular_embedding_periods: str = ""
    station_count: int = 512
    station_time_steps: int = 0
    station_value_bias: bool = False
    station_value_bias_scale: float = 0.5
    station_phase_bias: bool = False
    station_phase_bias_scale: float = 0.5
    station_phase_components: int = 2

    def __post_init__(self):
        if self.c_out is None:
            self.c_out = self.enc_in
        for name in (
            "seq_len",
            "pred_len",
            "enc_in",
            "c_out",
            "d_model",
            "d_ff",
            "topk",
            "soft_memory_patches",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % 4 or self.c_out > self.enc_in:
            raise ValueError("d_model must be divisible by four and c_out cannot exceed enc_in")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        for name in (
            "residual_scale",
            "value_readout_scale",
            "soft_memory_scale",
            "process_decay_scale",
        ):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"The complete model requires positive finite {name}")
        if self.station_value_bias or self.station_phase_bias:
            for name in ("station_count", "station_time_steps"):
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"Station-dependent settings require positive integer {name}")

    def to_dict(self):
        return asdict(self)
