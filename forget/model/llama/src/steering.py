import torch as t


class SteeringOp:
    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        raise NotImplementedError


class AddSteer(SteeringOp):
    """R' = R + scale * vec"""
    def __init__(self, vec: t.Tensor, scale: float = 1.0):
        self.vec = vec
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        return self.scale * self.vec.to(hidden.dtype)


class SignedSteer(SteeringOp):
    """R' = R + sign(R @ v_detect) * scale * v_steer"""
    def __init__(self, v_detect: t.Tensor, v_steer: t.Tensor, scale: float = 1.0):
        self.v_detect = v_detect
        self.v_steer = v_steer
        self.scale = scale

    def __call__(self, hidden: t.Tensor) -> t.Tensor:
        v_d = self.v_detect.to(hidden.dtype).squeeze()
        v_s = self.v_steer.to(hidden.dtype).squeeze()
        sign = (hidden @ v_d).sign().unsqueeze(-1)  # (*, 1)
        return sign * self.scale * v_s
