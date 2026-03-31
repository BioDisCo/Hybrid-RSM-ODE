"""Centralized growth models for microalgae plate cultures.

All Monod-based kinetic models consolidated in one place. Each formula
implemented ONCE and reused throughout the codebase.

Models include:
- Steady state (N_max): Haldane photoinhibition model
- Growth rate (μ_max): Three Haldane variants (light-only, light+nutrient, synergistic)
"""

from enum import Enum
from typing import Union
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

# Reference light intensity for converting real L0 values to factors
# L0_factor = L0_real / L0_REF
L0_REF = 170.0  # µmol·m⁻²·s⁻¹


class SteadyStateModel(Enum):
    """Available steady state models."""

    HALDANE = "haldane"


class GrowthRateModel(Enum):
    """Available growth rate models."""

    MONOD_SIMPLE = "monod_simple"
    HALDANE_LIGHT_ONLY = "haldane_light"
    HALDANE_LIGHT_AND_NUTRIENT = "haldane_both"
    HALDANE_SYNERGISTIC = "haldane_synergistic"


# =============================================================================
# STEADY STATE MODELS
# =============================================================================


def steady_state_haldane(xy, n_max_ref, k_c, k_l, k_i):
    """Haldane photoinhibition model for plate steady state.

    This is the primary N_max model used across all plate analyses.

    Formula:
        N_max(C,L) = N_max_ref × [C/(K_C+C)] × [L/(K_L+L+L²/K_I)]

    The Haldane denominator [K_L + L + L²/K_I] captures:
    - K_L: Substrate affinity (half-saturation constant)
    - L²/K_I: Photoinhibition term (growth suppression at high light)

    Parameters
    ----------
    xy : tuple of (ndarray, ndarray)
        (C, L) where:
        - C: Nutrient concentration (nutrient units, typically mM or µM)
        - L: Light intensity (µmol·m⁻²·s⁻¹)
    n_max_ref : float
        Reference maximum cell density at saturating C and L (cells/mL)
    k_c : float
        Nutrient half-saturation constant (same units as C)
    k_l : float
        Light half-saturation constant (µmol·m⁻²·s⁻¹)
    k_i : float
        Photoinhibition constant (higher = less inhibition)
        Fitted value: ~0.647 for plate cultures

    Returns
    -------
    ndarray
        N_max: Predicted steady state cell density (cells/mL)
    """
    c, l = xy
    return n_max_ref * (c / (k_c + c)) * (l / (k_l + l + l**2 / k_i))


# =============================================================================
# GROWTH RATE MODELS
# =============================================================================


def growth_rate_monod_simple(xy, mu_max_ref, k_c, k_l):
    """Simple double Monod model (no inhibition terms).

    Formula:
        μ(C,L) = μ_max_ref × [C/(K_C+C)] × [L/(K_L+L)]

    This is the simplest form with pure Monod kinetics for both nutrient and light.
    No photoinhibition, no nutrient inhibition, no synergistic terms.

    Parameters
    ----------
    xy : tuple of (ndarray, ndarray)
        (C, L) where C is nutrient concentration, L is light intensity
    mu_max_ref : float
        Reference maximum growth rate at saturating C and L (h⁻¹)
    k_c : float
        Nutrient half-saturation constant
    k_l : float
        Light half-saturation constant (µmol·m⁻²·s⁻¹)

    Returns
    -------
    ndarray
        μ: Predicted specific growth rate (h⁻¹)
    """
    c, l = xy
    return mu_max_ref * (c / (k_c + c)) * (l / (k_l + l))


def growth_rate_haldane_light_only(xy, mu_max_ref, k_c, k_l, k_i):
    """Model A: Haldane photoinhibition on light only.

    Formula:
        μ(C,L) = μ_max_ref × [C/(K_C+C)] × [L/(K_L+L+L²/K_I)]

    Assumes Monod kinetics for nutrient, Haldane kinetics for light.
    No nutrient photoinhibition or L×C synergistic terms.

    Parameters
    ----------
    xy : tuple of (ndarray, ndarray)
        (C, L) where C is nutrient concentration, L is light intensity
    mu_max_ref : float
        Reference maximum growth rate at saturating C and L (h⁻¹)
    k_c : float
        Nutrient half-saturation constant
    k_l : float
        Light half-saturation constant (µmol·m⁻²·s⁻¹)
    k_i : float
        Photoinhibition constant for light

    Returns
    -------
    ndarray
        μ: Predicted specific growth rate (h⁻¹)
    """
    c, l = xy
    return mu_max_ref * (c / (k_c + c)) * (l / (k_l + l + l**2 / k_i))


def growth_rate_haldane_both(xy, mu_max_ref, k_c, k_l, k_i_l, k_i_c):
    """Model B: Haldane inhibition for both light and nutrients.

    Formula:
        μ(C,L) = μ_max_ref × [C/(K_C+C+C²/K_I_C)] × [L/(K_L+L+L²/K_I_L)]

    Assumes Haldane kinetics for both nutrient and light dimensions.
    No synergistic L×C term.

    Parameters
    ----------
    xy : tuple of (ndarray, ndarray)
        (C, L)
    mu_max_ref : float
        Reference maximum growth rate (h⁻¹)
    k_c : float
        Nutrient half-saturation constant
    k_l : float
        Light half-saturation constant (µmol·m⁻²·s⁻¹)
    k_i_l : float
        Photoinhibition constant for light
    k_i_c : float
        Photoinhibition constant for nutrient (typically very high, ~uninhibited)

    Returns
    -------
    ndarray
        μ: Predicted specific growth rate (h⁻¹)
    """
    c, l = xy
    return mu_max_ref * (c / (k_c + c + c**2 / k_i_c)) * (l / (k_l + l + l**2 / k_i_l))


def growth_rate_haldane_synergistic(xy, mu_max_ref, k_c, k_l, k_i, alpha):
    """Model C: Haldane light inhibition + synergistic L×C interaction.

    Formula:
        μ(C,L) = μ_max_ref × [C/(K_C+C)] × [L/(K_L+L+L²/K_I)] / (1 + α·L·C)

    The denominator (1 + α·L·C) captures synergistic inhibition where
    high light AND high nutrient together suppress growth more than
    either factor alone.

    Parameters
    ----------
    xy : tuple of (ndarray, ndarray)
        (C, L)
    mu_max_ref : float
        Reference maximum growth rate (h⁻¹)
    k_c : float
        Nutrient half-saturation constant
    k_l : float
        Light half-saturation constant (µmol·m⁻²·s⁻¹)
    k_i : float
        Photoinhibition constant for light
    alpha : float
        Synergistic inhibition coefficient (L×C interaction strength)
        For plate cultures, typically ~0 (no synergy detected)

    Returns
    -------
    ndarray
        μ: Predicted specific growth rate (h⁻¹)
    """
    c, l = xy
    return (
        mu_max_ref
        * (c / (k_c + c))
        * (l / (k_l + l + l**2 / k_i))
        / (1 + alpha * l * c)
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def infer_growth_rate_model(params: tuple) -> GrowthRateModel:
    """Infer growth rate model type from fitted parameter tuple.

    When growth rate optimization returns variable-length tuples, this function
    determines which model was fitted based on parameter count and values.

    Logic:
    - 3 parameters: Simple Monod model (no inhibition)
    - 4 parameters: Either Model A (light-only) or Model C (synergistic, alpha=0)
    - 5 parameters: Either Model B (both) or Model C (synergistic, alpha≠0)

    Heuristic: Model B has k_i_l typically in range 10-10000 (photoinhibition constant),
    while Model C has alpha typically near 0 (no synergy found in plate data).
    If param[3] > 10.0, assume Model B; otherwise Model C.

    Parameters
    ----------
    params : tuple
        Fitted parameter tuple (variable length: 3, 4 or 5 elements)

    Returns
    -------
    GrowthRateModel
        Inferred model enum

    Raises
    ------
    ValueError
        If parameter count is not 3, 4 or 5
    """
    if len(params) == 3:
        # 3 params: Simple Monod model
        return GrowthRateModel.MONOD_SIMPLE
    elif len(params) == 4:
        # 4 params: typically Model A (light-only) or Model C with alpha=0
        # For plates, both converged to same solution, so either is valid
        return GrowthRateModel.HALDANE_LIGHT_ONLY
    elif len(params) == 5:
        # 5 params: distinguish between Model B and Model C based on param values
        # Model B: k_i_c typically 0.1-100 (nutrient photoinhibition constant)
        # Model C: alpha typically ~0-0.01 (synergistic interaction coefficient)
        # Heuristic: if params[4] < 0.1, likely Model C (small alpha); otherwise Model B
        if params[4] < 0.1:
            return GrowthRateModel.HALDANE_SYNERGISTIC
        else:
            return GrowthRateModel.HALDANE_LIGHT_AND_NUTRIENT
    else:
        raise ValueError(f"Invalid parameter count: {len(params)}. Expected 3, 4 or 5.")


def get_growth_rate_function(model: GrowthRateModel):
    """Factory function to get the appropriate growth rate model callable.

    Parameters
    ----------
    model : GrowthRateModel
        Model enum specifying which growth rate function to return

    Returns
    -------
    callable
        Function with signature (xy, *params) -> ndarray

    Raises
    ------
    ValueError
        If model is not a valid GrowthRateModel enum member
    """
    if model == GrowthRateModel.MONOD_SIMPLE:
        return growth_rate_monod_simple
    elif model == GrowthRateModel.HALDANE_LIGHT_ONLY:
        return growth_rate_haldane_light_only
    elif model == GrowthRateModel.HALDANE_LIGHT_AND_NUTRIENT:
        return growth_rate_haldane_both
    elif model == GrowthRateModel.HALDANE_SYNERGISTIC:
        return growth_rate_haldane_synergistic
    else:
        raise ValueError(f"Unknown growth rate model: {model}")


def evaluate_growth_rate(
    c: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    params: tuple,
    model: GrowthRateModel,
    use_l0_factors: bool = False,
) -> Union[float, np.ndarray]:
    """Evaluate growth rate for given (C,L) and fitted parameters.

    This is a convenience wrapper that handles model selection and parameter
    unpacking. Useful for simulations where model type must be inferred from
    saved parameter tuples.

    Parameters
    ----------
    c : float or ndarray
        Nutrient concentration (scalar or array)
    l : float or ndarray
        Light intensity in µmol·m⁻²·s⁻¹ (scalar or array)
        Will be converted to L0 factor if use_l0_factors=True
    params : tuple
        Fitted parameter tuple (4 or 5 elements depending on model)
    model : GrowthRateModel
        Model type to use for evaluation
    use_l0_factors : bool, optional
        If True, the fitted parameters expect L0 as factors (0-1)
        and l will be converted: l_factor = l / L0_REF
        Default is False (parameters expect real L0 values)

    Returns
    -------
    float or ndarray
        μ: Predicted specific growth rate(s)

    Raises
    ------
    ValueError
        If model is invalid
    """
    # Convert L0 to factor if needed
    if use_l0_factors:
        l_input = l / L0_REF
    else:
        l_input = l

    func = get_growth_rate_function(model)
    return func((c, l_input), *params)


def evaluate_steady_state(
    c: Union[float, np.ndarray],
    l: Union[float, np.ndarray],
    params: tuple,
    use_l0_factors: bool = False,
) -> Union[float, np.ndarray]:
    """Evaluate steady state for given (C,L) and fitted parameters.

    Parameters
    ----------
    c : float or ndarray
        Nutrient concentration (scalar or array)
    l : float or ndarray
        Light intensity in µmol·m⁻²·s⁻¹ (scalar or array)
        Will be converted to L0 factor if use_l0_factors=True
    params : tuple
        Fitted parameter tuple (n_max_ref, k_c, k_l) for Simple Monod
        or (n_max_ref, k_c, k_l, k_i) for Haldane with photoinhibition
    use_l0_factors : bool, optional
        If True, the fitted parameters expect L0 as factors (0-1)
        and l will be converted: l_factor = l / L0_REF
        Default is False (parameters expect real L0 values)

    Returns
    -------
    float or ndarray
        N_max: Predicted steady state cell density
    """
    # Convert L0 to factor if needed
    if use_l0_factors:
        l_input = l / L0_REF
    else:
        l_input = l

    # Handle both Simple Monod (3 params) and Haldane (4 params)
    if len(params) == 3:
        # Simple Monod model
        n_max_ref, k_c, k_l = params
        return n_max_ref * (c / (k_c + c)) * (l_input / (k_l + l_input))
    else:
        # Haldane model with photoinhibition
        return steady_state_haldane((c, l_input), *params)
