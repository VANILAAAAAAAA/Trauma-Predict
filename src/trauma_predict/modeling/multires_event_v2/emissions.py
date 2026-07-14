from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


POSITIVE_SCALE_FLOOR = 1e-4
MIN_SHAPE = POSITIVE_SCALE_FLOOR
CANONICAL_ARITHMETIC_RESIDUAL_ATOL = 1e-12
PHYSICAL_ARITHMETIC_ATOL = 1e-12
GRADABLE = 0
UNGRADABLE = 1


class EmissionSupportError(ValueError):
    """Raised when a target is outside the declared probability support."""


@dataclass(frozen=True)
class ZOILogitNormalParameters:
    mixture_logits: Tensor
    interior_loc: Tensor
    interior_scale_raw: Tensor


@dataclass(frozen=True)
class DenseValueParameters:
    range_logits: Tensor
    constant_value: ZOILogitNormalParameters
    minimum_coordinate: ZOILogitNormalParameters
    range_coordinate: ZOILogitNormalParameters
    last_coordinate: ZOILogitNormalParameters
    mean_coordinate: ZOILogitNormalParameters


@dataclass(frozen=True)
class DenseValueTarget:
    observed_hours: Tensor
    minimum: Tensor
    last: Tensor
    maximum: Tensor
    mean: Tensor


@dataclass(frozen=True)
class LabValueParameters:
    single_value: StudentTParameters
    range_logits: Tensor
    constant_value: StudentTParameters
    minimum: StudentTParameters
    log_range_loc: Tensor
    log_range_scale_raw: Tensor
    last_coordinate: ZOILogitNormalParameters


@dataclass(frozen=True)
class LabValueTarget:
    observation_count: Tensor
    minimum: Tensor
    last: Tensor
    maximum: Tensor


@dataclass(frozen=True)
class StudentTParameters:
    location: Tensor
    scale_raw: Tensor
    df_raw: Tensor


@dataclass(frozen=True)
class NEDParameters:
    zero_positive_logits: Tensor
    positive_max_loc: Tensor
    positive_max_scale_raw: Tensor
    last_ratio: ZOILogitNormalParameters
    mean_ratio: ZOILogitNormalParameters


@dataclass(frozen=True)
class NEDTarget:
    maximum: Tensor
    last: Tensor
    mean: Tensor
    compatible_vasopressor_duration: Tensor
    compatible_vasopressor_edge: Tensor


@dataclass(frozen=True)
class UOPParameters:
    zero_positive_logits: Tensor
    positive_loc: Tensor
    positive_scale_raw: Tensor


@dataclass(frozen=True)
class RespiratoryOccupancyParameters:
    active_set_logits: Tensor
    alr_location: Tensor
    alr_scale_raw: Tensor


def _require(condition: Tensor, message: str) -> None:
    if condition.numel() == 0:
        raise EmissionSupportError(message)
    valid = condition.all()
    if condition.device.type == "cuda":
        # Sampling/support checks must remain fail-closed without serializing
        # every factor through the host.  A failed CUDA assertion surfaces at
        # the next synchronization and invalidates that CUDA context.
        torch._assert_async(valid, message)
    elif not bool(valid.item()):
        raise EmissionSupportError(message)


def _require_shape(value: Tensor, shape: torch.Size, name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape={tuple(value.shape)} must equal {tuple(shape)}")


def _integer_target(value: Tensor, name: str) -> Tensor:
    if value.is_floating_point():
        _require(
            torch.isfinite(value) & value.eq(value.round()),
            f"{name} must contain finite exact integers",
        )
    return value.long()


def _positive(raw: Tensor) -> Tensor:
    return F.softplus(raw) + MIN_SHAPE


def _close(left: Tensor, right: Tensor | float) -> Tensor:
    return torch.isclose(
        left,
        torch.as_tensor(right, dtype=left.dtype, device=left.device),
        rtol=0.0,
        atol=PHYSICAL_ARITHMETIC_ATOL,
    )


def _canonical_unit(
    value: Tensor,
    name: str,
    *,
    exact_zero: Tensor | None = None,
    exact_one: Tensor | None = None,
) -> Tensor:
    """Construct an injective unit coordinate without tolerance-based snapping.

    Endpoint atoms are selected only by exact semantic equalities supplied by
    the caller.  A genuine interior value remains interior however close it is
    to zero or one.
    """

    coordinate = value
    if exact_zero is not None:
        _require_shape(exact_zero, value.shape, f"{name} exact-zero mask")
        coordinate = torch.where(exact_zero.bool(), torch.zeros_like(coordinate), coordinate)
    if exact_one is not None:
        _require_shape(exact_one, value.shape, f"{name} exact-one mask")
        coordinate = torch.where(exact_one.bool(), torch.ones_like(coordinate), coordinate)
    finite = torch.isfinite(coordinate)
    below = coordinate.lt(0.0)
    above = coordinate.gt(1.0)
    repairable_below = below & coordinate.ge(-CANONICAL_ARITHMETIC_RESIDUAL_ATOL)
    repairable_above = above & coordinate.le(1.0 + CANONICAL_ARITHMETIC_RESIDUAL_ATOL)
    _require(
        finite & ((~below) | repairable_below) & ((~above) | repairable_above),
        f"derived coordinate {name} exceeds [0,1] by more than "
        f"{CANONICAL_ARITHMETIC_RESIDUAL_ATOL:g}",
    )
    # This is a one-sided repair of impossible floating-point arithmetic
    # residue, not endpoint atomization: every already in-range value remains
    # untouched, however close it is to either endpoint.
    coordinate = torch.where(repairable_below, torch.zeros_like(coordinate), coordinate)
    coordinate = torch.where(repairable_above, torch.ones_like(coordinate), coordinate)
    return coordinate


def categorical_log_prob(logits: Tensor, target: Tensor) -> Tensor:
    """Normalized categorical log probability on the final logits dimension."""

    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("categorical logits must have a nonempty class dimension")
    _require_shape(target, logits.shape[:-1], "categorical target")
    index = _integer_target(target, "categorical target")
    _require(index.ge(0) & index.lt(logits.shape[-1]), "categorical target is out of support")
    return F.log_softmax(logits, dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)


def masked_categorical_log_prob(logits: Tensor, target: Tensor, valid_class_mask: Tensor) -> Tensor:
    """Categorical log probability renormalized over a per-example legal class set."""

    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("masked categorical logits need a class dimension")
    _require_shape(target, logits.shape[:-1], "masked categorical target")
    try:
        mask = torch.broadcast_to(valid_class_mask.bool(), logits.shape)
    except RuntimeError as error:
        raise ValueError("valid_class_mask is not broadcastable to logits") from error
    _require(mask.any(dim=-1), "every categorical row must retain at least one class")
    index = _integer_target(target, "masked categorical target")
    _require(index.ge(0) & index.lt(logits.shape[-1]), "masked target class is out of range")
    selected_is_valid = mask.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    _require(selected_is_valid, "masked categorical target is not legal for its conditioning state")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    return (
        (masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True))
        .gather(-1, index.unsqueeze(-1))
        .squeeze(-1)
    )


def bernoulli_log_prob(logits: Tensor, target: Tensor) -> Tensor:
    _require_shape(target, logits.shape, "Bernoulli target")
    numeric = target.to(dtype=logits.dtype)
    _require(numeric.eq(0) | numeric.eq(1), "Bernoulli target must be zero or one")
    return -F.binary_cross_entropy_with_logits(logits, numeric, reduction="none")


def hurdle_negative_binomial_log_prob(
    count: Tensor,
    gate_logits: Tensor,
    total_count_raw: Tensor,
    nb_logits: Tensor,
    *,
    force_positive: Tensor | None = None,
) -> Tensor:
    """Hurdle-NB log probability, optionally renormalized to ``N>=1``."""

    _require_shape(count, gate_logits.shape, "hurdle count")
    _require_shape(total_count_raw, gate_logits.shape, "hurdle total_count_raw")
    _require_shape(nb_logits, gate_logits.shape, "hurdle nb_logits")
    integer = _integer_target(count, "hurdle count")
    _require(integer.ge(0), "hurdle count must be nonnegative")
    forced = torch.zeros_like(integer, dtype=torch.bool)
    if force_positive is not None:
        _require_shape(force_positive, integer.shape, "hurdle force-positive mask")
        forced = force_positive.bool()
        _require((~forced) | integer.gt(0), "forced-positive hurdle count must be nonzero")
    positive = integer.gt(0)
    shifted = torch.where(positive, integer - 1, torch.zeros_like(integer))
    count_log_prob = torch.distributions.NegativeBinomial(
        total_count=_positive(total_count_raw),
        logits=nb_logits,
        validate_args=False,
    ).log_prob(shifted.to(dtype=gate_logits.dtype))
    ordinary = torch.where(
        positive,
        F.logsigmoid(gate_logits) + count_log_prob,
        F.logsigmoid(-gate_logits),
    )
    return torch.where(forced, count_log_prob, ordinary)


def sample_categorical(logits: Tensor, valid_class_mask: Tensor | None = None) -> Tensor:
    """Sample a normalized categorical, optionally restricted to legal classes."""

    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("categorical logits must have a nonempty class dimension")
    if valid_class_mask is None:
        sampled_logits = logits
    else:
        try:
            mask = torch.broadcast_to(valid_class_mask.bool(), logits.shape)
        except RuntimeError as error:
            raise ValueError("valid_class_mask is not broadcastable to logits") from error
        _require(mask.any(dim=-1), "every sampled categorical row needs a legal class")
        sampled_logits = logits.masked_fill(~mask, -torch.inf)
    _require(
        torch.isfinite(sampled_logits) | sampled_logits.eq(-torch.inf),
        "categorical sampling logits must be finite or masked negative infinity",
    )
    _require(
        torch.isfinite(sampled_logits).any(dim=-1),
        "every sampled categorical row needs at least one finite logit",
    )
    return torch.distributions.Categorical(
        logits=sampled_logits,
        validate_args=False,
    ).sample()


def sample_hurdle_negative_binomial(
    gate_logits: Tensor,
    total_count_raw: Tensor,
    nb_logits: Tensor,
    *,
    force_positive: Tensor | None = None,
) -> Tensor:
    _require_shape(total_count_raw, gate_logits.shape, "sampled hurdle total_count_raw")
    _require_shape(nb_logits, gate_logits.shape, "sampled hurdle nb_logits")
    _require(
        torch.isfinite(gate_logits)
        & torch.isfinite(total_count_raw)
        & torch.isfinite(nb_logits),
        "sampled hurdle parameters must be finite",
    )
    positive = torch.distributions.Bernoulli(
        logits=gate_logits,
        validate_args=False,
    ).sample().bool()
    if force_positive is not None:
        _require_shape(force_positive, gate_logits.shape, "sampled hurdle force-positive mask")
        positive = positive | force_positive.bool()
    shifted = torch.distributions.NegativeBinomial(
        total_count=_positive(total_count_raw),
        logits=nb_logits,
        validate_args=False,
    ).sample()
    return torch.where(positive, shifted + 1.0, torch.zeros_like(shifted))


def zoi_logit_normal_log_prob(
    value: Tensor,
    parameters: ZOILogitNormalParameters,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    component_mask: Tensor | None = None,
) -> Tensor:
    """Zero/one-inflated logit-Normal score on the canonical ``q`` measure.

    Exact endpoints are atoms.  For an interior normalized coordinate ``u``,
    ``q=logit(u)`` has a Normal density with respect to ``Lebesgue(dq)``.  No
    sigmoid or raw-interval Jacobian belongs in this canonical score.
    """

    _require_shape(
        parameters.mixture_logits,
        value.shape + (3,),
        "ZOI-logit-Normal mixture_logits",
    )
    _require_shape(parameters.interior_loc, value.shape, "ZOI-logit-Normal interior_loc")
    _require_shape(
        parameters.interior_scale_raw,
        value.shape,
        "ZOI-logit-Normal interior_scale_raw",
    )
    if not upper > lower:
        raise ValueError("ZOI-logit-Normal upper bound must exceed lower bound")
    lower_tensor = value.new_tensor(lower)
    upper_tensor = value.new_tensor(upper)
    _require(
        value.ge(lower_tensor) & value.le(upper_tensor),
        f"ZOI-logit-Normal target must lie in [{lower},{upper}]",
    )
    at_zero = value.eq(lower_tensor)
    at_one = value.eq(upper_tensor)
    interior = ~(at_zero | at_one)
    branch = torch.where(at_zero, 0, torch.where(at_one, 2, 1)).long()
    mask = (
        torch.ones(3, dtype=torch.bool, device=value.device)
        if component_mask is None
        else component_mask.to(device=value.device, dtype=torch.bool)
    )
    mixture = masked_categorical_log_prob(parameters.mixture_logits, branch, mask)
    normalized = (value - lower_tensor) / value.new_tensor(upper - lower)
    safe_normalized = torch.where(interior, normalized, torch.full_like(normalized, 0.5))
    q = (torch.log(safe_normalized) - torch.log1p(-safe_normalized)).to(
        dtype=parameters.interior_loc.dtype
    )
    interior_log_prob = torch.distributions.Normal(
        parameters.interior_loc,
        _positive(parameters.interior_scale_raw),
        validate_args=False,
    ).log_prob(q)
    return mixture + torch.where(
        interior,
        interior_log_prob,
        torch.zeros_like(interior_log_prob),
    )


def sample_zoi_logit_normal(
    parameters: ZOILogitNormalParameters,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    component_mask: Tensor | None = None,
) -> Tensor:
    """Sample from the mixed measure used by :func:`zoi_logit_normal_log_prob`."""

    if parameters.mixture_logits.shape[-1:] != (3,):
        raise ValueError("ZOI-logit-Normal sampling requires three mixture logits")
    shape = parameters.mixture_logits.shape[:-1]
    _require_shape(parameters.interior_loc, shape, "sampled ZOI-logit-Normal interior_loc")
    _require_shape(
        parameters.interior_scale_raw,
        shape,
        "sampled ZOI-logit-Normal interior_scale_raw",
    )
    if not upper > lower:
        raise ValueError("ZOI-logit-Normal upper bound must exceed lower bound")
    _require(
        torch.isfinite(parameters.mixture_logits),
        "sampled ZOI-logit-Normal mixture logits must be finite",
    )
    _require(
        torch.isfinite(parameters.interior_loc)
        & torch.isfinite(parameters.interior_scale_raw),
        "sampled ZOI-logit-Normal interior parameters must be finite",
    )
    branch = sample_categorical(parameters.mixture_logits.float(), component_mask)
    q = torch.distributions.Normal(
        parameters.interior_loc.double(),
        F.softplus(parameters.interior_scale_raw.double()) + POSITIVE_SCALE_FLOOR,
        validate_args=False,
    ).sample()
    # Keep the sampled canonical coordinate in float64 through its physical
    # pushforward.  Down-casting here can turn a legal interior draw into an
    # endpoint atom and breaks exact simplex/value support downstream.  The
    # feedback encoder owns the later neural-input float32 cast.
    interior = torch.sigmoid(q)
    _require(
        (~branch.eq(1))
        | (torch.isfinite(interior) & interior.gt(0.0) & interior.lt(1.0)),
        "sampled logit-Normal interior saturated to an endpoint",
    )
    unit = torch.where(
        branch.eq(0),
        torch.zeros_like(interior),
        torch.where(branch.eq(2), torch.ones_like(interior), interior),
    )
    # Python scalars are cast by the elementwise kernels to ``unit.dtype``.
    # Materializing two device scalars here adds two host-to-device copies per
    # draw without changing the physical pushforward.
    return lower + (upper - lower) * unit


def positive_log_coordinate_log_prob(
    value: Tensor,
    loc: Tensor,
    scale_raw: Tensor,
) -> Tensor:
    """Normal score for ``q=log(value)`` under the canonical q-coordinate."""

    _require_shape(loc, value.shape, "log-coordinate location")
    _require_shape(scale_raw, value.shape, "log-coordinate scale_raw")
    _require(value.gt(0), "log-coordinate target must be strictly positive")
    return torch.distributions.Normal(
        loc,
        _positive(scale_raw),
        validate_args=False,
    ).log_prob(torch.log(value).to(dtype=loc.dtype))


def student_t_log_prob(value: Tensor, parameters: StudentTParameters) -> Tensor:
    _require_shape(parameters.location, value.shape, "Student-t location")
    _require_shape(parameters.scale_raw, value.shape, "Student-t scale_raw")
    _require_shape(parameters.df_raw, value.shape, "Student-t df_raw")
    return torch.distributions.StudentT(
        df=2.0 + _positive(parameters.df_raw),
        loc=parameters.location,
        scale=_positive(parameters.scale_raw),
        validate_args=False,
    ).log_prob(value.to(dtype=parameters.location.dtype))


def sample_student_t(parameters: StudentTParameters) -> Tensor:
    _require_shape(parameters.scale_raw, parameters.location.shape, "sampled Student-t scale_raw")
    _require_shape(parameters.df_raw, parameters.location.shape, "sampled Student-t df_raw")
    _require(
        torch.isfinite(parameters.location)
        & torch.isfinite(parameters.scale_raw)
        & torch.isfinite(parameters.df_raw),
        "sampled Student-t parameters must be finite",
    )
    return torch.distributions.StudentT(
        df=2.0 + _positive(parameters.df_raw),
        loc=parameters.location,
        scale=_positive(parameters.scale_raw),
        validate_args=False,
    ).sample()


@lru_cache(maxsize=None)
def _legal_ordinal_triples_cpu(maximum: int) -> Tensor:
    if maximum < 1:
        raise ValueError("ordinal maximum must be positive")
    rows = [
        (minimum, last, upper)
        for minimum in range(1, maximum + 1)
        for last in range(minimum, maximum + 1)
        for upper in range(last, maximum + 1)
    ]
    return torch.tensor(rows, dtype=torch.long)


def legal_ordinal_triples(maximum: int, *, device: torch.device | None = None) -> Tensor:
    return _legal_ordinal_triples_cpu(maximum).to(device=device)


def ordinal_triple_class_mask(
    states: Tensor,
    observation_count: Tensor,
    *,
    source_semantics: Literal["raw_point", "hourly_sequence"],
) -> Tensor:
    """Return the exact count-conditioned ordered-triple support.

    Raw-point eye/motor extrema may include multiple observations in one hour,
    so every ordered triple is legal for every positive observed-hour count.
    Verbal is one value per gradable hour: count one is diagonal, count two
    forces LAST to one of the two endpoints, and count three or more is
    unrestricted.
    """

    if states.ndim != 2 or states.shape[-1] != 3:
        raise ValueError("ordinal states must be [classes,3]")
    count = _integer_target(observation_count, "ordinal observation_count")
    _require(count.ge(0), "ordinal observation_count must be nonnegative")
    full = torch.ones(count.shape + (states.shape[0],), dtype=torch.bool, device=states.device)
    if source_semantics == "raw_point":
        return full
    if source_semantics != "hourly_sequence":
        raise ValueError(f"unknown ordinal source semantics {source_semantics!r}")
    diagonal = states[:, 0].eq(states[:, 1]) & states[:, 1].eq(states[:, 2])
    endpoint_last = states[:, 1].eq(states[:, 0]) | states[:, 1].eq(states[:, 2])
    return torch.where(
        count.eq(1).unsqueeze(-1),
        diagonal,
        torch.where(count.eq(2).unsqueeze(-1), endpoint_last, full),
    )


def legal_gcs_triple_log_prob(
    logits: Tensor,
    target_triple: Tensor,
    *,
    maximum: int,
    observation_count: Tensor | None = None,
    source_semantics: Literal["raw_point", "hourly_sequence"] = "hourly_sequence",
) -> Tensor:
    """Categorical probability over legal ``MIN <= LAST <= MAX`` GCS triples.

    A zero observation count deactivates the primitive and returns log one.
    Positive-count support follows the declared source aggregation semantics.
    """

    states = legal_ordinal_triples(maximum, device=logits.device)
    expected_classes = math.comb(maximum + 2, 3)
    if logits.shape[-1] != expected_classes:
        raise ValueError(f"GCS K={maximum} requires {expected_classes} logits")
    _require_shape(target_triple, logits.shape[:-1] + (3,), "GCS target triple")
    leading_shape = logits.shape[:-1]
    if observation_count is None:
        active = torch.ones(leading_shape, dtype=torch.bool, device=logits.device)
        count = torch.full(leading_shape, 3, dtype=torch.long, device=logits.device)
    else:
        _require_shape(observation_count, leading_shape, "GCS observation_count")
        count = _integer_target(observation_count, "GCS observation_count")
        _require(count.ge(0), "GCS observation_count must be nonnegative")
        active = count.gt(0)
    numeric_target = _integer_target(target_triple, "GCS target triple")
    safe_target = torch.where(
        active.unsqueeze(-1),
        numeric_target,
        torch.ones_like(numeric_target),
    )
    matches = safe_target.unsqueeze(-2).eq(states).all(dim=-1)
    _require((~active) | matches.any(dim=-1), "GCS triple is outside the legal ordinal set")
    index = matches.to(dtype=torch.long).argmax(dim=-1)
    valid = ordinal_triple_class_mask(
        states,
        count,
        source_semantics=source_semantics,
    )
    log_prob = masked_categorical_log_prob(logits, index, valid)
    return torch.where(active, log_prob, torch.zeros_like(log_prob))


def gcs_verbal_ungradable_hours_log_prob(
    logits: Tensor,
    ungradable_hours: Tensor,
    observed_hours: Tensor,
) -> Tensor:
    if logits.shape[-1] != 5:
        raise ValueError("GCS verbal H_u requires five logits for counts 0..4")
    _require_shape(ungradable_hours, logits.shape[:-1], "GCS verbal H_u")
    _require_shape(observed_hours, logits.shape[:-1], "GCS verbal H_obs")
    observed = _integer_target(observed_hours, "GCS verbal H_obs")
    ungradable = _integer_target(ungradable_hours, "GCS verbal H_u")
    _require(observed.ge(0) & observed.le(4), "GCS verbal H_obs must lie in 0..4")
    _require(
        ungradable.ge(0) & ungradable.le(observed),
        "GCS verbal must satisfy 0 <= H_u <= H_obs",
    )
    classes = torch.arange(5, device=logits.device)
    mask = classes.le(observed.unsqueeze(-1))
    return masked_categorical_log_prob(logits, ungradable, mask)


def gcs_verbal_latest_status_log_prob(
    logits: Tensor,
    latest_status: Tensor,
    observed_hours: Tensor,
    ungradable_hours: Tensor,
) -> Tensor:
    if logits.shape[-1] != 2:
        raise ValueError("latest verbal status requires GRADABLE/UNGRADABLE logits")
    leading_shape = logits.shape[:-1]
    _require_shape(latest_status, leading_shape, "latest verbal status")
    _require_shape(observed_hours, leading_shape, "GCS verbal H_obs")
    _require_shape(ungradable_hours, leading_shape, "GCS verbal H_u")
    observed = _integer_target(observed_hours, "GCS verbal H_obs")
    ungradable = _integer_target(ungradable_hours, "GCS verbal H_u")
    _require(observed.ge(0) & observed.le(4), "GCS verbal H_obs must lie in 0..4")
    _require(
        ungradable.ge(0) & ungradable.le(observed),
        "GCS verbal must satisfy 0 <= H_u <= H_obs",
    )
    active = observed.gt(0)
    gradable_hours = observed - ungradable
    mask = torch.stack((gradable_hours.gt(0), ungradable.gt(0)), dim=-1)
    safe_mask = torch.where(active.unsqueeze(-1), mask, torch.ones_like(mask))
    safe_status = torch.where(active, latest_status, torch.zeros_like(latest_status))
    log_prob = masked_categorical_log_prob(logits, safe_status, safe_mask)
    return torch.where(active, log_prob, torch.zeros_like(log_prob))


def gcs_verbal_gradable_triple_log_prob(
    logits: Tensor,
    target_triple: Tensor,
    observed_hours: Tensor,
    ungradable_hours: Tensor,
) -> Tensor:
    observed = _integer_target(observed_hours, "GCS verbal H_obs")
    ungradable = _integer_target(ungradable_hours, "GCS verbal H_u")
    _require_shape(observed, logits.shape[:-1], "GCS verbal H_obs")
    _require_shape(ungradable, logits.shape[:-1], "GCS verbal H_u")
    _require(observed.ge(0) & observed.le(4), "GCS verbal H_obs must lie in 0..4")
    _require(
        ungradable.ge(0) & ungradable.le(observed),
        "GCS verbal must satisfy 0 <= H_u <= H_obs",
    )
    return legal_gcs_triple_log_prob(
        logits,
        target_triple,
        maximum=5,
        observation_count=observed - ungradable,
    )


def masked_count_vector_log_prob(logits: Tensor, count: Tensor, upper: Tensor) -> Tensor:
    """Componentwise masked categorical log probability.

    This helper is intentionally not registered as the dense-abnormal emission:
    the contract for that vector is autoregressive and therefore needs a
    conditional table for every component after the first.
    """

    if logits.ndim < 2:
        raise ValueError("count-vector logits require component and class dimensions")
    _require_shape(count, logits.shape[:-1], "count vector")
    _require_shape(upper, logits.shape[:-2], "count-vector upper")
    maximum = logits.shape[-1] - 1
    upper_integer = _integer_target(upper, "count-vector upper")
    _require(
        upper_integer.ge(0) & upper_integer.le(maximum),
        f"count-vector upper must lie in 0..{maximum}",
    )
    classes = torch.arange(maximum + 1, device=logits.device)
    mask = classes.le(upper_integer.unsqueeze(-1).unsqueeze(-1))
    return masked_categorical_log_prob(logits, count, mask).sum(dim=-1)


DENSE_ABNORMAL_CONDITION_KEYS: Mapping[str, tuple[str, ...]] = {
    "heart_rate": ("HR_LT40", "HR_GT120"),
    "systolic_bp": ("SBP_LT90",),
    "mean_arterial_pressure": ("MAP_LT65", "MAP_GE65_LT70"),
    "respiratory_rate": ("RR_LT8", "RR_GE22"),
    "temperature": ("TEMP_GE38",),
    "spo2": ("SPO2_LE93",),
}


def _presence_class_mask(
    observed: Tensor,
    predicate_present: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    classes = torch.arange(5, device=device)
    bounded = classes.le(observed.unsqueeze(-1))
    positive = bounded & classes.gt(0)
    zero = bounded & classes.eq(0)
    return torch.where(
        (observed.gt(0) & predicate_present).unsqueeze(-1),
        positive,
        zero,
    )


def dense_abnormal_class_masks(
    *,
    field: str,
    condition_keys: Sequence[str],
    observed_hours: Tensor,
    minimum: Tensor,
    maximum: Tensor,
    first_duration: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Exact registry-specific duration support implied by raw extrema.

    Predicate positivity is equivalent to the matching raw MIN/MAX threshold.
    MAP low and borderline durations share the same hourly-min reducer and are
    therefore mutually exclusive within each hour.
    """

    expected = DENSE_ABNORMAL_CONDITION_KEYS.get(field)
    if expected is None or tuple(condition_keys) != expected:
        raise ValueError(
            f"dense abnormal registry drift for {field}: {tuple(condition_keys)!r} != {expected!r}"
        )
    observed = _integer_target(observed_hours, "dense abnormal observed_hours")
    _require(observed.ge(0) & observed.le(4), "dense abnormal observed_hours must lie in 0..4")
    _require_shape(minimum, observed.shape, "dense abnormal minimum")
    _require_shape(maximum, observed.shape, "dense abnormal maximum")
    _require(
        (~observed.gt(0))
        | (torch.isfinite(minimum) & torch.isfinite(maximum) & minimum.le(maximum)),
        "active dense abnormal extrema must be finite and ordered",
    )
    if field == "heart_rate":
        first_present = minimum.lt(40.0)
        second_present = maximum.gt(120.0)
    elif field == "systolic_bp":
        first_present = minimum.lt(90.0)
        second_present = None
    elif field == "mean_arterial_pressure":
        first_present = minimum.lt(65.0)
        second_present = minimum.ge(65.0) & minimum.lt(70.0)
    elif field == "respiratory_rate":
        first_present = minimum.lt(8.0)
        second_present = maximum.ge(22.0)
    elif field == "temperature":
        first_present = maximum.ge(38.0)
        second_present = None
    elif field == "spo2":
        first_present = minimum.le(93.0)
        second_present = None
    else:  # pragma: no cover - guarded by the registry lookup above
        raise AssertionError(field)
    first_mask = _presence_class_mask(
        observed,
        first_present,
        device=minimum.device,
    )
    if len(expected) == 1:
        return first_mask, None
    if first_duration is None:
        raise ValueError(f"second dense abnormal condition for {field} requires first_duration")
    first = _integer_target(first_duration, "first dense abnormal duration")
    _require_shape(first, observed.shape, "first dense abnormal duration")
    if field != "mean_arterial_pressure":
        assert second_present is not None
        return first_mask, _presence_class_mask(
            observed,
            second_present,
            device=minimum.device,
        )
    classes = torch.arange(5, device=minimum.device)
    bounded = classes.le(observed.unsqueeze(-1))
    low_branch = bounded & classes.le((observed - first).unsqueeze(-1))
    borderline_branch = bounded & classes.gt(0)
    normal_branch = bounded & classes.eq(0)
    second_mask = torch.where(
        (observed.gt(0) & first_present).unsqueeze(-1),
        low_branch,
        torch.where(
            (observed.gt(0) & second_present).unsqueeze(-1),
            borderline_branch,
            normal_branch,
        ),
    )
    return first_mask, second_mask


def dense_abnormal_duration_log_prob(
    raw_parameters: Tensor,
    count: Tensor,
    upper: Tensor,
    *,
    field: str,
    condition_keys: Sequence[str],
    minimum: Tensor,
    maximum: Tensor,
) -> Tensor:
    """Autoregressive categorical duration contract for one or two conditions.

    The 30 raw parameters are ``5`` logits for the first duration followed by a
    ``5 x 5`` conditional table for the second.  The selected table row is
    indexed by the preceding *true* duration during teacher-forced scoring.
    Both categorical factors are renormalized to the legal classes ``0..H``.
    """

    condition_count = len(tuple(condition_keys))
    if condition_count not in (1, 2):
        raise ValueError("dense abnormal duration supports one or two ordered conditions")
    if raw_parameters.shape[-1] != 30:
        raise ValueError("dense abnormal duration requires 30 raw parameters")
    _require_shape(count, raw_parameters.shape[:-1] + (condition_count,), "abnormal durations")
    _require_shape(upper, raw_parameters.shape[:-1], "abnormal-duration upper")
    duration = _integer_target(count, "abnormal duration")
    observed = _integer_target(upper, "abnormal-duration upper")
    _require(observed.ge(0) & observed.le(4), "abnormal-duration upper must lie in 0..4")
    _require(
        duration.ge(0) & duration.le(observed.unsqueeze(-1)),
        "abnormal duration must lie in 0..H",
    )
    first_valid, _ = dense_abnormal_class_masks(
        field=field,
        condition_keys=condition_keys,
        observed_hours=observed,
        minimum=minimum,
        maximum=maximum,
        first_duration=duration[..., 0],
    )
    first = masked_categorical_log_prob(
        raw_parameters[..., :5],
        duration[..., 0],
        first_valid,
    )
    if condition_count == 1:
        return first
    conditional_table = raw_parameters[..., 5:30].reshape(raw_parameters.shape[:-1] + (5, 5))
    row_index = duration[..., 0].unsqueeze(-1).unsqueeze(-1).expand(duration.shape[:-1] + (1, 5))
    second_logits = conditional_table.gather(-2, row_index).squeeze(-2)
    _, second_valid = dense_abnormal_class_masks(
        field=field,
        condition_keys=condition_keys,
        observed_hours=observed,
        minimum=minimum,
        maximum=maximum,
        first_duration=duration[..., 0],
    )
    assert second_valid is not None
    second = masked_categorical_log_prob(
        second_logits,
        duration[..., 1],
        second_valid,
    )
    return first + second


def sample_dense_abnormal_duration(
    raw_parameters: Tensor,
    upper: Tensor,
    *,
    field: str,
    condition_keys: Sequence[str],
    minimum: Tensor,
    maximum: Tensor,
) -> Tensor:
    """Sample the dense-abnormal chain with the same conditional table as loss."""

    condition_count = len(tuple(condition_keys))
    if condition_count not in (1, 2):
        raise ValueError("dense abnormal duration supports one or two ordered conditions")
    if raw_parameters.shape[-1] != 30:
        raise ValueError("dense abnormal duration requires 30 raw parameters")
    _require_shape(upper, raw_parameters.shape[:-1], "abnormal-duration upper")
    observed = _integer_target(upper, "abnormal-duration upper")
    _require(observed.ge(0) & observed.le(4), "abnormal-duration upper must lie in 0..4")
    # The second mask needs the first draw, but the first mask does not depend
    # on that placeholder.
    first_valid, _ = dense_abnormal_class_masks(
        field=field,
        condition_keys=condition_keys,
        observed_hours=observed,
        minimum=minimum,
        maximum=maximum,
        first_duration=torch.zeros_like(observed),
    )
    first = sample_categorical(raw_parameters[..., :5], first_valid)
    if condition_count == 1:
        return first.unsqueeze(-1)
    table = raw_parameters[..., 5:30].reshape(raw_parameters.shape[:-1] + (5, 5))
    row_index = first.unsqueeze(-1).unsqueeze(-1).expand(first.shape + (1, 5))
    second_logits = table.gather(-2, row_index).squeeze(-2)
    _, second_valid = dense_abnormal_class_masks(
        field=field,
        condition_keys=condition_keys,
        observed_hours=observed,
        minimum=minimum,
        maximum=maximum,
        first_duration=first,
    )
    assert second_valid is not None
    second = sample_categorical(second_logits, second_valid)
    return torch.stack((first, second), dim=-1)


def lower_triangular_conditioned_parameters(
    raw_parameters: Tensor,
    preceding_values: Tensor,
    *,
    component_count: int,
    parameter_width: int,
    transform: str,
) -> Tensor:
    """Decode base parameters plus strict-lower conditional coefficients.

    For component ``j``, every preceding component ``k < j`` contributes
    ``coefficient[j,k] * transform(value[k])`` to each distribution parameter.
    The strict-lower layout makes future and self values structurally
    inaccessible and is shared by teacher-forced scoring and sampling.
    """

    if component_count < 1 or parameter_width < 1:
        raise ValueError("component_count and parameter_width must be positive")
    expected_width = parameter_width * (
        component_count + component_count * (component_count - 1) // 2
    )
    if raw_parameters.shape[-1] != expected_width:
        raise ValueError(
            f"autoregressive parameters require width {expected_width}, "
            f"got {raw_parameters.shape[-1]}"
        )
    _require_shape(
        preceding_values,
        raw_parameters.shape[:-1] + (component_count,),
        "autoregressive preceding values",
    )
    numeric = preceding_values.to(dtype=raw_parameters.dtype)
    if transform == "duration_fraction":
        transformed = numeric / 4.0
    elif transform == "binary":
        transformed = numeric
    elif transform == "log1p_count":
        _require(numeric.ge(0), "autoregressive count values must be nonnegative")
        transformed = torch.log1p(numeric)
    else:
        raise ValueError(f"unknown autoregressive transform {transform!r}")

    base_width = component_count * parameter_width
    base = raw_parameters[..., :base_width].reshape(
        raw_parameters.shape[:-1] + (component_count, parameter_width)
    )
    coefficient = raw_parameters[..., base_width:].reshape(
        raw_parameters.shape[:-1] + (component_count * (component_count - 1) // 2, parameter_width)
    )
    conditioned: list[Tensor] = []
    offset = 0
    for component in range(component_count):
        current = base[..., component, :]
        for preceding in range(component):
            current = current + coefficient[..., offset, :] * transformed[..., preceding].unsqueeze(
                -1
            )
            offset += 1
        conditioned.append(current)
    return torch.stack(conditioned, dim=-2)


def _lower_triangular_conditioned_component(
    raw_parameters: Tensor,
    preceding_values: Tensor,
    *,
    component: int,
    component_count: int,
    parameter_width: int,
    transform: str,
) -> Tensor:
    """Decode only one autoregressive row in legacy floating-point order.

    Sequential sampling needs row ``component`` only.  Calling
    :func:`lower_triangular_conditioned_parameters` at every step previously
    rebuilt all rows, including future rows that were immediately discarded.
    The increasing-``preceding`` accumulation below deliberately matches the
    full decoder operation-for-operation for the selected row; do not replace
    it with a reduction whose floating-point order can differ.
    """

    if component_count < 1 or parameter_width < 1:
        raise ValueError("component_count and parameter_width must be positive")
    if component < 0 or component >= component_count:
        raise ValueError(
            f"autoregressive component must lie in 0..{component_count - 1}"
        )
    expected_width = parameter_width * (
        component_count + component_count * (component_count - 1) // 2
    )
    if raw_parameters.shape[-1] != expected_width:
        raise ValueError(
            f"autoregressive parameters require width {expected_width}, "
            f"got {raw_parameters.shape[-1]}"
        )
    _require_shape(
        preceding_values,
        raw_parameters.shape[:-1] + (component_count,),
        "autoregressive preceding values",
    )
    numeric = preceding_values.to(dtype=raw_parameters.dtype)
    if transform == "duration_fraction":
        transformed = numeric / 4.0
    elif transform == "binary":
        transformed = numeric
    elif transform == "log1p_count":
        _require(numeric.ge(0), "autoregressive count values must be nonnegative")
        transformed = torch.log1p(numeric)
    else:
        raise ValueError(f"unknown autoregressive transform {transform!r}")

    base_width = component_count * parameter_width
    base = raw_parameters[..., :base_width].reshape(
        raw_parameters.shape[:-1] + (component_count, parameter_width)
    )
    coefficient = raw_parameters[..., base_width:].reshape(
        raw_parameters.shape[:-1]
        + (component_count * (component_count - 1) // 2, parameter_width)
    )
    current = base[..., component, :]
    offset = component * (component - 1) // 2
    for preceding in range(component):
        current = current + coefficient[..., offset + preceding, :] * transformed[
            ..., preceding
        ].unsqueeze(-1)
    return current


def sample_autoregressive_hurdle_count_vector(
    raw_parameters: Tensor,
    *,
    component_count: int,
    required_positive: Tensor | None = None,
    require_any_positive: Tensor | None = None,
) -> Tensor:
    leading_shape = raw_parameters.shape[:-1]
    if required_positive is None:
        required = torch.zeros(
            leading_shape + (component_count,),
            dtype=torch.bool,
            device=raw_parameters.device,
        )
    else:
        _require_shape(
            required_positive,
            leading_shape + (component_count,),
            "required-positive count vector",
        )
        required = required_positive.bool()
    if require_any_positive is None:
        require_any = torch.zeros(leading_shape, dtype=torch.bool, device=raw_parameters.device)
    else:
        _require_shape(require_any_positive, leading_shape, "require-any-positive count gate")
        require_any = require_any_positive.bool()
    sampled: list[Tensor] = []
    for component in range(component_count):
        preceding = (
            torch.stack(sampled, dim=-1)
            if sampled
            else raw_parameters.new_zeros(raw_parameters.shape[:-1] + (0,))
        )
        padded = F.pad(preceding, (0, component_count - component))
        decoded = _lower_triangular_conditioned_component(
            raw_parameters,
            padded,
            component=component,
            component_count=component_count,
            parameter_width=3,
            transform="log1p_count",
        )
        force = required[..., component]
        if component == component_count - 1:
            no_preceding_positive = (
                torch.stack(sampled, dim=-1).eq(0).all(dim=-1)
                if sampled
                else torch.ones(leading_shape, dtype=torch.bool, device=raw_parameters.device)
            )
            force = force | (require_any & no_preceding_positive)
        sampled.append(
            sample_hurdle_negative_binomial(
                decoded[..., 0],
                decoded[..., 1],
                decoded[..., 2],
                force_positive=force,
            )
        )
    return torch.stack(sampled, dim=-1)


def sample_autoregressive_binary_vector(
    raw_parameters: Tensor,
    *,
    component_count: int,
) -> Tensor:
    sampled: list[Tensor] = []
    for component in range(component_count):
        preceding = (
            torch.stack(sampled, dim=-1)
            if sampled
            else raw_parameters.new_zeros(raw_parameters.shape[:-1] + (0,))
        )
        padded = F.pad(preceding, (0, component_count - component))
        logits = _lower_triangular_conditioned_component(
            raw_parameters,
            padded,
            component=component,
            component_count=component_count,
            parameter_width=1,
            transform="binary",
        )[..., 0]
        _require(torch.isfinite(logits), "sampled binary-vector logits must be finite")
        sampled.append(
            torch.distributions.Bernoulli(
                logits=logits,
                validate_args=False,
            ).sample()
        )
    return torch.stack(sampled, dim=-1)


def sample_autoregressive_zoi_logit_normal_vector(
    raw_parameters: Tensor,
    *,
    component_count: int,
    span_hours: float = 4.0,
) -> Tensor:
    sampled: list[Tensor] = []
    for component in range(component_count):
        preceding = (
            torch.stack(sampled, dim=-1)
            if sampled
            else raw_parameters.new_zeros(raw_parameters.shape[:-1] + (0,))
        )
        padded = F.pad(preceding, (0, component_count - component))
        decoded = _lower_triangular_conditioned_component(
            raw_parameters,
            padded,
            component=component,
            component_count=component_count,
            parameter_width=5,
            transform="duration_fraction",
        )
        sampled.append(
            sample_zoi_logit_normal(
                ZOILogitNormalParameters(decoded[..., :3], decoded[..., 3], decoded[..., 4]),
                lower=0.0,
                upper=span_hours,
            )
        )
    return torch.stack(sampled, dim=-1)


def dense_joint_value_log_prob(
    parameters: DenseValueParameters,
    target: DenseValueTarget,
    *,
    lower: float,
    upper: float,
) -> Tensor:
    """One normalized mixed log probability for dense MIN/LAST/MAX/MEAN."""

    minimum = target.minimum
    shape = minimum.shape
    for name, value in (
        ("last", target.last),
        ("maximum", target.maximum),
        ("mean", target.mean),
        ("observed_hours", target.observed_hours),
    ):
        _require_shape(value, shape, f"dense {name}")
    _require_shape(parameters.range_logits, shape + (2,), "dense range_logits")
    hours = _integer_target(target.observed_hours, "dense observed_hours")
    _require(hours.ge(0) & hours.le(4), "dense observed_hours must lie in 0..4")
    active = hours.gt(0)
    midpoint = minimum.new_tensor((lower + upper) / 2.0)
    minimum = torch.where(active, minimum, midpoint)
    last = torch.where(active, target.last, midpoint)
    maximum = torch.where(active, target.maximum, midpoint)
    mean = torch.where(active, target.mean, midpoint)
    support = (
        minimum.ge(lower)
        & maximum.le(upper)
        & minimum.le(last)
        & last.le(maximum)
    )
    hour_float = hours.clamp_min(1).to(dtype=minimum.dtype)
    lower_mean = (last + (hour_float - 1.0) * minimum) / hour_float
    upper_mean = (last + (hour_float - 1.0) * maximum) / hour_float
    support &= (~hours.eq(1)) | mean.eq(last)

    value_range = maximum - minimum
    zero_range = value_range.eq(0)
    zero_support = last.eq(minimum) & mean.eq(minimum)
    support &= (~zero_range) | zero_support
    _require((~active) | support, "dense target violates its joint support")
    _require((~active) | (~zero_range) | zero_support, "zero-range dense branch must be constant")
    branch = (~zero_range).long()
    branch_log_prob = categorical_log_prob(parameters.range_logits, branch)
    safe_range = torch.where(zero_range, torch.ones_like(value_range), value_range)
    minimum_denominator = upper - minimum
    safe_minimum_denominator = torch.where(
        zero_range,
        torch.ones_like(minimum_denominator),
        minimum_denominator,
    )
    lower_tensor = minimum.new_tensor(lower)
    upper_tensor = minimum.new_tensor(upper)
    alpha = _canonical_unit(
        (minimum - lower_tensor) / (upper_tensor - lower_tensor),
        "dense alpha",
        exact_zero=minimum.eq(lower_tensor),
        exact_one=minimum.eq(upper_tensor),
    )
    constant_log_prob = zoi_logit_normal_log_prob(alpha, parameters.constant_value)
    beta = _canonical_unit(
        value_range / safe_minimum_denominator,
        "dense beta",
        exact_zero=zero_range,
        exact_one=maximum.eq(upper_tensor),
    )
    last_coordinate = _canonical_unit(
        (last - minimum) / safe_range,
        "dense u",
        exact_zero=last.eq(minimum),
        exact_one=(~zero_range) & last.eq(maximum),
    )
    mean_span = upper_mean - lower_mean
    safe_mean_span = torch.where(
        hours.gt(1) & (~zero_range),
        mean_span,
        torch.ones_like(mean_span),
    )
    raw_v = (mean - lower_mean) / safe_mean_span
    positive_range_multi_hour = hours.gt(1) & (~zero_range)
    mean_coordinate = _canonical_unit(
        torch.where(positive_range_multi_hour, raw_v, torch.full_like(raw_v, 0.5)),
        "dense v",
        exact_zero=positive_range_multi_hour & mean.eq(lower_mean),
        exact_one=positive_range_multi_hour & mean.eq(upper_mean),
    )
    dummy = torch.full_like(alpha, 0.5)
    alpha = torch.where(zero_range, dummy, alpha)
    beta = torch.where(zero_range, dummy, beta)
    last_coordinate = torch.where(zero_range, dummy, last_coordinate)
    mean_coordinate = torch.where(zero_range, dummy, mean_coordinate)
    no_upper_atom = minimum.new_tensor([True, True, False], dtype=torch.bool)
    no_zero_atom = minimum.new_tensor([False, True, True], dtype=torch.bool)
    positive_log_prob = (
        zoi_logit_normal_log_prob(
            alpha,
            parameters.minimum_coordinate,
            component_mask=no_upper_atom,
        )
        + zoi_logit_normal_log_prob(
            beta,
            parameters.range_coordinate,
            component_mask=no_zero_atom,
        )
        + zoi_logit_normal_log_prob(last_coordinate, parameters.last_coordinate)
    )
    mean_log_prob = zoi_logit_normal_log_prob(mean_coordinate, parameters.mean_coordinate)
    positive_log_prob = positive_log_prob + torch.where(
        hours.gt(1), mean_log_prob, torch.zeros_like(mean_log_prob)
    )
    result = branch_log_prob + torch.where(zero_range, constant_log_prob, positive_log_prob)
    return torch.where(active, result, torch.zeros_like(result))


def lab_joint_value_log_prob(
    parameters: LabValueParameters,
    target: LabValueTarget,
) -> Tensor:
    """Canonical-coordinate ordered lab-state log score.

    ``target`` is already in the field's frozen train-only affine coordinate.
    The score is normalized on the explicit mixed coordinates: one Student-t
    coordinate for ``N=1`` and the zero-range branch, or ``(z_min, r_z, u)``
    for positive range.  It is deliberately not described as a raw-space
    density over the redundant ``(MIN,LAST,MAX)`` tuple.
    """

    minimum = target.minimum
    shape = minimum.shape
    for name, value in (
        ("last", target.last),
        ("maximum", target.maximum),
        ("observation_count", target.observation_count),
    ):
        _require_shape(value, shape, f"lab {name}")
    _require_shape(parameters.range_logits, shape + (2,), "lab range_logits")
    count = _integer_target(target.observation_count, "lab observation_count")
    _require(count.ge(0), "lab observation count must be nonnegative")
    active = count.gt(0)
    minimum = torch.where(active, minimum, torch.zeros_like(minimum))
    last = torch.where(active, target.last, torch.zeros_like(target.last))
    maximum = torch.where(active, target.maximum, torch.zeros_like(target.maximum))
    support = (
        torch.isfinite(minimum)
        & torch.isfinite(last)
        & torch.isfinite(maximum)
        & minimum.le(last)
        & last.le(maximum)
    )
    support &= (~count.eq(1)) | (minimum.eq(last) & last.eq(maximum))
    support &= (~count.eq(2)) | maximum.eq(minimum) | last.eq(minimum) | last.eq(maximum)
    _require((~active) | support, "lab target violates count-conditioned support")

    single_log_prob = student_t_log_prob(last, parameters.single_value)
    value_range = maximum - minimum
    zero_range = value_range.eq(0)
    _require(
        (~active) | (~zero_range) | (minimum.eq(last) & last.eq(maximum)),
        "zero-range lab branch must be exactly constant",
    )
    branch = (~zero_range).long()
    branch_log_prob = categorical_log_prob(parameters.range_logits, branch)
    constant_log_prob = student_t_log_prob(minimum, parameters.constant_value)
    safe_range = torch.where(zero_range, torch.ones_like(value_range), value_range)
    u = _canonical_unit(
        (last - minimum) / safe_range,
        "lab u",
        exact_zero=last.eq(minimum),
        exact_one=(~zero_range) & last.eq(maximum),
    )
    dummy = torch.full_like(u, 0.5)
    u = torch.where(zero_range, dummy, u)
    _require_shape(parameters.log_range_loc, minimum.shape, "lab log-range location")
    _require_shape(parameters.log_range_scale_raw, minimum.shape, "lab log-range scale_raw")
    log_range = torch.log(safe_range).to(dtype=parameters.log_range_loc.dtype)
    log_range_log_prob = torch.distributions.Normal(
        parameters.log_range_loc,
        _positive(parameters.log_range_scale_raw),
        validate_args=False,
    ).log_prob(log_range)
    endpoint_only = count.eq(2) & (~zero_range)
    all_components = torch.ones(
        count.shape + (3,),
        dtype=torch.bool,
        device=count.device,
    )
    endpoint_components = torch.tensor(
        [True, False, True],
        dtype=torch.bool,
        device=count.device,
    )
    last_component_mask = torch.where(
        endpoint_only.unsqueeze(-1),
        endpoint_components,
        all_components,
    )
    positive_log_prob = (
        student_t_log_prob(minimum, parameters.minimum)
        + log_range_log_prob
        + zoi_logit_normal_log_prob(
            u,
            parameters.last_coordinate,
            component_mask=last_component_mask,
        )
    )
    repeated_log_prob = branch_log_prob + torch.where(
        zero_range, constant_log_prob, positive_log_prob
    )
    result = torch.where(count.eq(1), single_log_prob, repeated_log_prob)
    return torch.where(active, result, torch.zeros_like(result))


def ned_joint_value_log_prob(parameters: NEDParameters, target: NEDTarget) -> Tensor:
    maximum = target.maximum
    shape = maximum.shape
    for name, value in (
        ("last", target.last),
        ("mean", target.mean),
        ("compatible_vasopressor_duration", target.compatible_vasopressor_duration),
        ("compatible_vasopressor_edge", target.compatible_vasopressor_edge),
    ):
        _require_shape(value, shape, f"NED {name}")
    _require_shape(parameters.zero_positive_logits, shape + (2,), "NED branch logits")
    _require(
        torch.isfinite(maximum)
        & torch.isfinite(target.last)
        & torch.isfinite(target.mean)
        & maximum.ge(0)
        & target.last.ge(0.0)
        & target.mean.ge(0.0)
        & target.last.le(maximum)
        & target.mean.le(maximum),
        "NED target violates 0<=LAST,MEAN<=MAX",
    )
    zero = maximum.eq(0)
    _require(
        (~zero) | (target.last.eq(0.0) & target.mean.eq(0.0)),
        "NED MAX=0 must force LAST=MEAN=0",
    )
    _require(
        zero | target.compatible_vasopressor_duration.bool(),
        "positive NED MAX requires compatible positive vasopressor duration",
    )
    _require(
        target.last.eq(0.0) | target.compatible_vasopressor_edge.bool(),
        "positive NED LAST requires a compatible vasopressor right-edge state",
    )
    _require(
        zero | target.mean.gt(0.0),
        "positive NED MAX requires a strictly positive MEAN",
    )
    branch_mask = torch.stack(
        (
            torch.ones_like(zero, dtype=torch.bool),
            target.compatible_vasopressor_duration.bool(),
        ),
        dim=-1,
    )
    branch_log_prob = masked_categorical_log_prob(
        parameters.zero_positive_logits,
        (~zero).long(),
        branch_mask,
    )
    safe_maximum = torch.where(zero, torch.ones_like(maximum), maximum)
    positive_max = positive_log_coordinate_log_prob(
        safe_maximum,
        parameters.positive_max_loc,
        parameters.positive_max_scale_raw,
    )
    raw_last_ratio = target.last / safe_maximum
    raw_mean_ratio = target.mean / safe_maximum
    last_ratio = _canonical_unit(
        torch.where(zero, torch.full_like(maximum, 0.5), raw_last_ratio),
        "NED LAST/MAX",
        exact_zero=(~zero) & target.last.eq(0.0),
        exact_one=(~zero) & target.last.eq(maximum),
    )
    mean_ratio = _canonical_unit(
        torch.where(zero, torch.full_like(maximum, 0.5), raw_mean_ratio),
        "NED MEAN/MAX",
        exact_zero=(~zero) & target.mean.eq(0.0),
        exact_one=(~zero) & target.mean.eq(maximum),
    )
    all_components = torch.ones(shape + (3,), dtype=torch.bool, device=maximum.device)
    last_positive_allowed = target.compatible_vasopressor_edge.bool()
    last_mask = torch.stack(
        (
            torch.ones_like(last_positive_allowed),
            last_positive_allowed,
            last_positive_allowed,
        ),
        dim=-1,
    )
    mean_mask = torch.tensor(
        [False, True, True],
        dtype=torch.bool,
        device=maximum.device,
    ).expand(shape + (3,))
    last_mask = torch.where(zero.unsqueeze(-1), all_components, last_mask)
    mean_mask = torch.where(zero.unsqueeze(-1), all_components, mean_mask)
    positive_state = (
        positive_max
        + zoi_logit_normal_log_prob(
            last_ratio,
            parameters.last_ratio,
            component_mask=last_mask,
        )
        + zoi_logit_normal_log_prob(
            mean_ratio,
            parameters.mean_ratio,
            component_mask=mean_mask,
        )
    )
    return branch_log_prob + torch.where(zero, torch.zeros_like(positive_state), positive_state)


def uop_sum_log_prob(
    parameters: UOPParameters,
    amount_sum: Tensor,
    observation_count: Tensor,
) -> Tensor:
    shape = amount_sum.shape
    _require_shape(observation_count, shape, "UOP observation_count")
    _require_shape(parameters.zero_positive_logits, shape + (2,), "UOP branch logits")
    count = _integer_target(observation_count, "UOP observation_count")
    _require(count.ge(0), "UOP observation_count must be nonnegative")
    active = count.gt(0)
    _require(
        (~active) | (torch.isfinite(amount_sum) & amount_sum.ge(0)),
        "observed UOP SUM must be finite and nonnegative",
    )
    safe_sum = torch.where(active, amount_sum, torch.zeros_like(amount_sum))
    positive = safe_sum.gt(0)
    branch_log_prob = categorical_log_prob(parameters.zero_positive_logits, positive.long())
    positive_value = positive_log_coordinate_log_prob(
        torch.where(positive, safe_sum, torch.ones_like(safe_sum)),
        parameters.positive_loc,
        parameters.positive_scale_raw,
    )
    result = branch_log_prob + torch.where(
        positive, positive_value, torch.zeros_like(positive_value)
    )
    return torch.where(active, result, torch.zeros_like(result))


def respiratory_edge_evidence_log_prob(
    edge_logits: Tensor,
    edge_evidence: Tensor,
    block_evidence: Tensor,
) -> Tensor:
    _require_shape(edge_evidence, edge_logits.shape, "respiratory E_edge")
    _require_shape(block_evidence, edge_logits.shape, "respiratory E_block")
    edge = _integer_target(edge_evidence, "respiratory E_edge")
    block = _integer_target(block_evidence, "respiratory E_block")
    _require(edge.ge(0) & edge.le(1) & block.ge(0) & block.le(1), "respiratory evidence is binary")
    _require(edge.le(block), "respiratory support requires E_edge <= E_block")
    stochastic = bernoulli_log_prob(edge_logits, edge)
    return torch.where(block.bool(), stochastic, torch.zeros_like(stochastic))


def respiratory_occupancy_log_prob(
    parameters: RespiratoryOccupancyParameters,
    durations: Tensor,
    *,
    block_evidence: Tensor | None = None,
    span_hours: float = 4.0,
) -> Tensor:
    """Active-set categorical plus diagonal-Normal score in ALR coordinates."""

    if durations.ndim < 1 or durations.shape[-1] != 5:
        raise ValueError("respiratory durations must contain D0 plus four modalities")
    leading_shape = durations.shape[:-1]
    _require_shape(parameters.active_set_logits, leading_shape + (31,), "active-set logits")
    _require_shape(parameters.alr_location, leading_shape + (4,), "ALR location")
    _require_shape(parameters.alr_scale_raw, leading_shape + (4,), "ALR scale_raw")
    if block_evidence is None:
        active = torch.ones(leading_shape, dtype=torch.bool, device=durations.device)
    else:
        _require_shape(block_evidence, leading_shape, "respiratory E_block")
        block = _integer_target(block_evidence, "respiratory E_block")
        _require(block.ge(0) & block.le(1), "respiratory E_block must be binary")
        active = block.bool()
    dummy = torch.zeros_like(durations)
    dummy[..., 0] = span_hours
    safe_durations = torch.where(active.unsqueeze(-1), durations, dummy)
    _require(
        (~active).unsqueeze(-1) | (torch.isfinite(safe_durations) & safe_durations.ge(0)),
        "respiratory durations must be nonnegative",
    )
    _require(
        (~active) | _close(safe_durations.sum(dim=-1), span_hours),
        "D0 plus documented respiratory durations must equal the block span",
    )
    positive = safe_durations.gt(0)
    _require(positive.any(dim=-1), "respiratory occupancy must have a nonempty active set")
    powers = torch.tensor([1, 2, 4, 8, 16], device=durations.device, dtype=torch.long)
    active_code = (positive.long() * powers).sum(dim=-1)
    active_index = active_code - 1
    active_set_log_prob = categorical_log_prob(parameters.active_set_logits, active_index)

    flat_duration = safe_durations.reshape(-1, 5)
    flat_positive = positive.reshape(-1, 5)
    flat_code = active_code.reshape(-1)
    flat_location = parameters.alr_location.reshape(-1, 4)
    flat_scale = _positive(parameters.alr_scale_raw).reshape(-1, 4)
    coordinate_log_prob = parameters.alr_location.new_zeros(flat_code.shape)
    for code in range(1, 32):
        row_index = torch.nonzero(flat_code.eq(code), as_tuple=False).squeeze(-1)
        if row_index.numel() == 0:
            continue
        component_index = torch.nonzero(
            torch.tensor([(code >> bit) & 1 for bit in range(5)], device=durations.device),
            as_tuple=False,
        ).squeeze(-1)
        component_count = int(component_index.numel())
        if component_count == 1:
            selected_duration = flat_duration.index_select(0, row_index).index_select(
                1, component_index
            )
            _require(
                _close(selected_duration.squeeze(-1), span_hours),
                "one-component respiratory branch must occupy the full block",
            )
            continue
        selected_duration = flat_duration.index_select(0, row_index).index_select(
            1, component_index
        )
        selected_positive = flat_positive.index_select(0, row_index).index_select(
            1, component_index
        )
        _require(selected_positive, "active respiratory simplex components must be positive")
        proportion = selected_duration / span_hours
        alr = torch.log(proportion[..., :-1] / proportion[..., -1:]).to(
            dtype=parameters.alr_location.dtype
        )
        dimension = component_count - 1
        density = (
            torch.distributions.Normal(
                flat_location.index_select(0, row_index)[..., :dimension],
                flat_scale.index_select(0, row_index)[..., :dimension],
                validate_args=False,
            )
            .log_prob(alr)
            .sum(dim=-1)
        )
        coordinate_log_prob = coordinate_log_prob.index_copy(0, row_index, density)
    result = active_set_log_prob + coordinate_log_prob.reshape(leading_shape)
    return torch.where(active, result, torch.zeros_like(result))


def respiratory_edge_state_log_prob(
    logits: Tensor,
    state: Tensor,
    edge_evidence: Tensor,
) -> Tensor:
    if logits.shape[-1] != 4:
        raise ValueError("respiratory edge state requires four modality logits")
    _require_shape(state, logits.shape[:-1], "respiratory edge state")
    _require_shape(edge_evidence, logits.shape[:-1], "respiratory E_edge")
    evidence = _integer_target(edge_evidence, "respiratory E_edge")
    _require(evidence.ge(0) & evidence.le(1), "respiratory E_edge must be binary")
    safe_state = torch.where(evidence.bool(), state, torch.zeros_like(state))
    log_prob = categorical_log_prob(logits, safe_state)
    return torch.where(evidence.bool(), log_prob, torch.zeros_like(log_prob))


def autoregressive_hurdle_count_vector_log_prob(
    raw_parameters: Tensor,
    count: Tensor,
    *,
    active: Tensor | None = None,
    required_positive: Tensor | None = None,
    require_any_positive: Tensor | None = None,
) -> Tensor:
    """Lower-triangular hurdle-NB vector log probability.

    Every component has three base parameters.  Every strict-lower edge has
    three learned coefficients applied to ``log1p`` of the preceding count.
    """

    if count.ndim < 1:
        raise ValueError("count vector needs a component dimension")
    component_count = count.shape[-1]
    if active is None:
        active_bool = torch.ones(count.shape[:-1], dtype=torch.bool, device=count.device)
    else:
        _require_shape(active, count.shape[:-1], "count-vector active gate")
        active_bool = active.bool()
    safe_count = torch.where(active_bool.unsqueeze(-1), count, torch.zeros_like(count))
    if required_positive is None:
        required = torch.zeros_like(safe_count, dtype=torch.bool)
    else:
        _require_shape(required_positive, count.shape, "required-positive count vector")
        required = required_positive.bool() & active_bool.unsqueeze(-1)
    if require_any_positive is None:
        require_any = torch.zeros_like(active_bool)
    else:
        _require_shape(
            require_any_positive,
            active_bool.shape,
            "require-any-positive count gate",
        )
        require_any = require_any_positive.bool() & active_bool
    integer_count = _integer_target(safe_count, "autoregressive hurdle count")
    _require(
        (~require_any) | integer_count.gt(0).any(dim=-1),
        "count vector must contain at least one positive component",
    )
    force_positive = required.clone()
    force_positive[..., -1] |= require_any & integer_count[..., :-1].eq(0).all(dim=-1)
    parameters = lower_triangular_conditioned_parameters(
        raw_parameters,
        safe_count,
        component_count=component_count,
        parameter_width=3,
        transform="log1p_count",
    )
    component = hurdle_negative_binomial_log_prob(
        safe_count,
        parameters[..., 0],
        parameters[..., 1],
        parameters[..., 2],
        force_positive=force_positive,
    )
    result = component.sum(dim=-1)
    return torch.where(active_bool, result, torch.zeros_like(result))


def respiratory_onset_log_prob(
    count: Tensor,
    raw_parameters: Tensor,
    block_evidence: Tensor,
    documented_durations: Tensor,
    edge_evidence: Tensor,
    edge_state: Tensor,
) -> Tensor:
    if count.shape[-1:] != (4,):
        raise ValueError("respiratory onset requires four modality counts")
    leading_shape = count.shape[:-1]
    _require_shape(documented_durations, count.shape, "respiratory documented durations")
    _require_shape(edge_evidence, leading_shape, "respiratory edge evidence")
    _require_shape(edge_state, leading_shape, "respiratory edge state")
    block = _integer_target(block_evidence, "respiratory block evidence")
    edge = _integer_target(edge_evidence, "respiratory edge evidence")
    state = _integer_target(edge_state, "respiratory edge state")
    _require(block.ge(0) & block.le(1), "respiratory block evidence must be binary")
    _require(edge.ge(0) & edge.le(block), "respiratory edge evidence requires block evidence")
    _require((~edge.bool()) | (state.ge(0) & state.lt(4)), "respiratory edge state is invalid")
    _require(
        torch.isfinite(documented_durations)
        & documented_durations.ge(0.0)
        & documented_durations.le(4.0),
        "respiratory documented durations must lie in [0,4]",
    )
    one_hot_edge = F.one_hot(state.clamp(0, 3), num_classes=4).bool()
    required = edge.bool().unsqueeze(-1) & one_hot_edge & documented_durations.eq(0.0)
    require_any = block.bool() & (~documented_durations.gt(0.0).any(dim=-1))
    return autoregressive_hurdle_count_vector_log_prob(
        raw_parameters,
        count,
        active=block,
        required_positive=required,
        require_any_positive=require_any,
    )


def vasopressor_duration_log_prob(
    durations: Tensor,
    raw_parameters: Tensor,
    *,
    span_hours: float = 4.0,
) -> Tensor:
    if durations.ndim < 1 or durations.shape[-1] != 6:
        raise ValueError("vasopressor durations require six registered agents")
    _require(
        torch.isfinite(durations) & durations.ge(0) & durations.le(span_hours),
        f"vasopressor durations must lie in [0,{span_hours}]",
    )
    decoded = lower_triangular_conditioned_parameters(
        raw_parameters,
        durations,
        component_count=6,
        parameter_width=5,
        transform="duration_fraction",
    )
    duration_coordinate = _canonical_unit(
        durations / span_hours,
        "vasopressor duration fraction",
        exact_zero=durations.eq(0.0),
        exact_one=durations.eq(span_hours),
    )
    return zoi_logit_normal_log_prob(
        duration_coordinate,
        ZOILogitNormalParameters(decoded[..., :3], decoded[..., 3], decoded[..., 4]),
    ).sum(dim=-1)


def autoregressive_binary_vector_log_prob(raw_parameters: Tensor, target: Tensor) -> Tensor:
    if target.ndim < 1:
        raise ValueError("binary vector needs a component dimension")
    numeric = target.to(dtype=raw_parameters.dtype)
    _require(numeric.eq(0) | numeric.eq(1), "binary-vector target must be zero or one")
    component_count = target.shape[-1]
    logits = lower_triangular_conditioned_parameters(
        raw_parameters,
        numeric,
        component_count=component_count,
        parameter_width=1,
        transform="binary",
    ).squeeze(-1)
    return bernoulli_log_prob(logits, target).sum(dim=-1)


def vasopressor_onset_log_prob(
    count: Tensor,
    raw_parameters: Tensor,
    durations: Tensor,
    edge_state: Tensor,
) -> Tensor:
    if count.shape[-1:] != (6,):
        raise ValueError("vasopressor onset requires six agent counts")
    _require_shape(durations, count.shape, "vasopressor durations")
    _require_shape(edge_state, count.shape, "vasopressor edge state")
    _require(
        torch.isfinite(durations) & durations.ge(0.0) & durations.le(4.0),
        "vasopressor durations must lie in [0,4]",
    )
    edge = _integer_target(edge_state, "vasopressor edge state")
    _require(edge.ge(0) & edge.le(1), "vasopressor edge state must be binary")
    required = edge.bool() & durations.eq(0.0)
    return autoregressive_hurdle_count_vector_log_prob(
        raw_parameters,
        count,
        required_positive=required,
    )


CORE_EMISSION_LOG_PROBS: Mapping[str, Callable[..., Tensor]] = {
    "categorical_hours_0_4": categorical_log_prob,
    "dense_joint_value_state": dense_joint_value_log_prob,
    "dense_abnormal_duration_vector": dense_abnormal_duration_log_prob,
    "gcs_ordinal_triple": legal_gcs_triple_log_prob,
    "gcs_verbal_ungradable_hours_given_observed": gcs_verbal_ungradable_hours_log_prob,
    "gcs_verbal_latest_status": gcs_verbal_latest_status_log_prob,
    "gcs_verbal_gradable_ordinal_triple": gcs_verbal_gradable_triple_log_prob,
    "hurdle_negative_binomial_count": hurdle_negative_binomial_log_prob,
    "lab_joint_value_state": lab_joint_value_log_prob,
    "respiratory_block_evidence": bernoulli_log_prob,
    "respiratory_edge_evidence_given_block": respiratory_edge_evidence_log_prob,
    "respiratory_occupancy_vector": respiratory_occupancy_log_prob,
    "respiratory_edge_state": respiratory_edge_state_log_prob,
    "respiratory_onset_vector": respiratory_onset_log_prob,
    "vasopressor_duration_vector": vasopressor_duration_log_prob,
    "vasopressor_edge_state_vector": autoregressive_binary_vector_log_prob,
    "vasopressor_onset_vector": vasopressor_onset_log_prob,
    "ned_joint_value_state": ned_joint_value_log_prob,
    "uop_sum_given_count": uop_sum_log_prob,
}


__all__ = [
    "CORE_EMISSION_LOG_PROBS",
    "DenseValueParameters",
    "DenseValueTarget",
    "EmissionSupportError",
    "DENSE_ABNORMAL_CONDITION_KEYS",
    "GRADABLE",
    "LabValueParameters",
    "LabValueTarget",
    "NEDParameters",
    "NEDTarget",
    "RespiratoryOccupancyParameters",
    "StudentTParameters",
    "UNGRADABLE",
    "UOPParameters",
    "ZOILogitNormalParameters",
    "autoregressive_binary_vector_log_prob",
    "autoregressive_hurdle_count_vector_log_prob",
    "bernoulli_log_prob",
    "categorical_log_prob",
    "dense_abnormal_duration_log_prob",
    "dense_abnormal_class_masks",
    "dense_joint_value_log_prob",
    "gcs_verbal_gradable_triple_log_prob",
    "gcs_verbal_latest_status_log_prob",
    "gcs_verbal_ungradable_hours_log_prob",
    "hurdle_negative_binomial_log_prob",
    "lab_joint_value_log_prob",
    "legal_gcs_triple_log_prob",
    "legal_ordinal_triples",
    "ordinal_triple_class_mask",
    "masked_categorical_log_prob",
    "masked_count_vector_log_prob",
    "ned_joint_value_log_prob",
    "positive_log_coordinate_log_prob",
    "lower_triangular_conditioned_parameters",
    "respiratory_edge_evidence_log_prob",
    "respiratory_edge_state_log_prob",
    "respiratory_occupancy_log_prob",
    "respiratory_onset_log_prob",
    "sample_autoregressive_binary_vector",
    "sample_autoregressive_hurdle_count_vector",
    "sample_autoregressive_zoi_logit_normal_vector",
    "sample_categorical",
    "sample_dense_abnormal_duration",
    "sample_hurdle_negative_binomial",
    "sample_student_t",
    "sample_zoi_logit_normal",
    "student_t_log_prob",
    "uop_sum_log_prob",
    "vasopressor_duration_log_prob",
    "vasopressor_onset_log_prob",
    "zoi_logit_normal_log_prob",
]
