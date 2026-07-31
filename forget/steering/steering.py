import torch as t


def _ensure_2d(v: t.Tensor) -> t.Tensor:
    """(hidden,) → (1, hidden). Already 2D+ passes through."""
    return v if v.dim() >= 2 else v.unsqueeze(0)


class SteeringOp:
    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        raise NotImplementedError


class AddSteer(SteeringOp):
    """R' = R + scale * vec"""
    def __init__(self, vec: t.Tensor, scale: float = 1.0):
        self.vec = _ensure_2d(vec)
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v = self.scale * self.vec.to(hidden.dtype)        # (batch, hidden) or (1, hidden)
        return v.unsqueeze(1)                              # (batch, 1, hidden)


class SignedSteer(SteeringOp):
    """R' = R + sign(R @ v_detect) * scale * v_steer"""
    def __init__(self, v_detect: t.Tensor, v_steer: t.Tensor, scale: float = 1.0):
        self.v_detect = _ensure_2d(v_detect)
        self.v_steer = _ensure_2d(v_steer)
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v_d = self.v_detect.to(hidden.dtype)                     # (batch, hidden)
        v_s = self.v_steer.to(hidden.dtype)                      # (batch, hidden)
        proj = (hidden * v_d.unsqueeze(1)).sum(-1)               # (batch, seq)
        sign = proj.sign().unsqueeze(-1)                         # (batch, seq, 1)
        return sign * self.scale * v_s.unsqueeze(1)              # (batch, seq, hidden)


class ThreshSignedSteer(SteeringOp):
    """R' = R + sign(R @ v_detect - tau) * scale * v_steer"""
    def __init__(self, v_detect: t.Tensor, v_steer: t.Tensor, tau: float, scale: float = 1.0):
        self.v_detect = _ensure_2d(v_detect)
        self.v_steer = _ensure_2d(v_steer)
        self.tau = tau
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v_d = self.v_detect.to(hidden.dtype)
        v_s = self.v_steer.to(hidden.dtype)
        proj = (hidden * v_d.unsqueeze(1)).sum(-1)               # (batch, seq)
        sign = (proj - self.tau).sign().unsqueeze(-1)            # (batch, seq, 1)
        return sign * self.scale * v_s.unsqueeze(1)


class GatedSteer(SteeringOp):
    """R' = R + max(R @ v_detect - tau, 0) / |R @ v_detect - tau| * scale * v_steer"""
    def __init__(self, v_detect: t.Tensor, v_steer: t.Tensor, tau: float, scale: float = 1.0):
        self.v_detect = _ensure_2d(v_detect)
        self.v_steer = _ensure_2d(v_steer)
        self.tau = tau
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v_d = self.v_detect.to(hidden.dtype)
        v_s = self.v_steer.to(hidden.dtype)
        proj = (hidden * v_d.unsqueeze(1)).sum(-1) - self.tau    # (batch, seq)
        gate = (proj > 0).to(hidden.dtype).unsqueeze(-1)         # (batch, seq, 1)
        return gate * self.scale * v_s.unsqueeze(1)


class SoftGatedSteer(SteeringOp):
    def __init__(self, v_detect: t.Tensor, v_steer: t.Tensor, tau: float, scale: float = 1.0, temp: float = 1.0):
        self.v_detect = _ensure_2d(v_detect)
        self.v_steer = _ensure_2d(v_steer)
        self.tau = tau
        self.scale = scale
        self.temp = temp

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v_d = self.v_detect.to(hidden.dtype)
        v_s = self.v_steer.to(hidden.dtype)
        proj = (hidden * v_d.unsqueeze(1)).sum(-1) - self.tau    # (batch, seq)
        gate = t.sigmoid(proj / self.temp).unsqueeze(-1)         # (batch, seq, 1)
        return gate * self.scale * v_s.unsqueeze(1)
