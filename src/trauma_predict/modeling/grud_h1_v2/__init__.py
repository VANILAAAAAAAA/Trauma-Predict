"""Single-resolution GRU-D baseline for the frozen six-M4 V2 task."""

from .model import (
    GRUDH1JointM4Config,
    GRUDH1JointM4Model,
    build_grud_h1_joint_m4_model,
)

__all__ = [
    "GRUDH1JointM4Config",
    "GRUDH1JointM4Model",
    "build_grud_h1_joint_m4_model",
]
