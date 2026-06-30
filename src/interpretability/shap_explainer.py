"""
SHAP interpretability for the Optuna-tuned LSTM model.

Uses shap.DeepExplainer (with GradientExplainer fallback) to compute
SHAP values for the test set, then generates:
1. A global beeswarm summary plot
2. A regime comparison: High-Volatility vs Low-Volatility days

Public API
----------
- `compute_shap_values`: compute SHAP values for a Keras model + background + test data.
- `plot_shap_summary`: generate the beeswarm summary plot.
- `plot_regime_comparison`: split test data by realized vol and plot side-by-side beeswarms.
- `run_shap_analysis`: end-to-end helper.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Import only for type hints — tensorflow is heavy and not needed at
    # import time. The actual model object is passed in by callers.
    import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------
def compute_shap_values(
    model: tf.keras.Model,
    background: np.ndarray,
    test_data: np.ndarray,
    feature_names: list[str],
    explainer_type: str = "deep",
) -> tuple[np.ndarray, str]:
    """Compute SHAP values for a Keras model.

    Tries DeepExplainer → GradientExplainer → KernelExplainer in order.
    The first one that works is used.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    background : np.ndarray, shape (n_bg, timesteps, n_features)
        Background data for the explainer.
    test_data : np.ndarray, shape (n_test, timesteps, n_features)
        Data to explain.
    feature_names : list[str]
    explainer_type : str
        Preferred explainer: "deep", "gradient", or "kernel".

    Returns
    -------
    shap_values : np.ndarray, shape (n_test, n_features)
    used_explainer : str
    """
    import numpy as np
    import shap

    # Limit background for performance
    if len(background) > 50:
        bg_idx = np.random.choice(len(background), 50, replace=False)
        background = background[bg_idx]
    # Limit test data for performance
    if len(test_data) > 150:
        ts_idx = np.random.choice(len(test_data), 150, replace=False)
        test_data = test_data[ts_idx]

    n_features = test_data.shape[-1]

    # ---- Try DeepExplainer ----
    if explainer_type in ("deep", "auto"):
        try:
            logger.info("Trying shap.DeepExplainer ...")
            explainer = shap.DeepExplainer(model, background)
            sv = explainer.shap_values(test_data)
            if isinstance(sv, list):
                sv = sv[0]
            if sv.ndim == 3:
                sv = sv.squeeze(axis=1)
            logger.info(f"DeepExplainer succeeded. Shape: {sv.shape}")
            return sv, "deep"
        except Exception as e:
            logger.warning(f"DeepExplainer failed ({type(e).__name__}: {e})")

    # ---- Try GradientExplainer ----
    if explainer_type in ("gradient", "auto"):
        try:
            logger.info("Trying shap.GradientExplainer ...")
            explainer = shap.GradientExplainer(model, background)
            sv = explainer.shap_values(test_data)
            if isinstance(sv, list):
                sv = sv[0]
            if sv.ndim == 3:
                sv = sv.squeeze(axis=1)
            logger.info(f"GradientExplainer succeeded. Shape: {sv.shape}")
            return sv, "gradient"
        except Exception as e:
            logger.warning(f"GradientExplainer failed ({type(e).__name__}: {e})")

    # ---- Fallback: KernelExplainer (model-agnostic) ----
    logger.info("Using shap.KernelExplainer (model-agnostic fallback) ...")

    # Flatten 3D → 2D for KernelExplainer
    bg_2d = background.squeeze(axis=1) if background.ndim == 3 else background
    ts_2d = test_data.squeeze(axis=1) if test_data.ndim == 3 else test_data

    # Prediction wrapper: reshape 2D → 3D, call model, flatten output
    def predict_fn(x_2d: np.ndarray) -> np.ndarray:
        """Wrap the Keras model so KernelExplainer can call it with 2D input."""
        x_3d = x_2d.reshape(-1, 1, n_features)
        return model.predict(x_3d, verbose=0).ravel()

    explainer = shap.KernelExplainer(predict_fn, bg_2d)
    sv = explainer.shap_values(ts_2d, nsamples=100, silent=True)

    if isinstance(sv, list):
        sv = sv[0]
    if sv.ndim == 3:
        sv = sv.squeeze(axis=1)

    logger.info(f"KernelExplainer succeeded. Shape: {sv.shape}")
    return sv, "kernel"


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_shap_summary(
    shap_values: np.ndarray,
    test_data: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str = "SHAP Feature Importance (Test Set)",
) -> None:
    """Generate a beeswarm summary plot."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import shap

    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Squeeze test_data if it has a timestep dimension
    if test_data.ndim == 3:
        test_data_2d = test_data.squeeze(axis=1)
    else:
        test_data_2d = test_data

    # Build an Explanation object for the modern SHAP API
    expl_obj = shap.Explanation(
        values=shap_values,
        data=test_data_2d,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(expl_obj, max_display=len(feature_names), show=False)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_regime_comparison(
    shap_values: np.ndarray,
    test_data: np.ndarray,
    feature_names: list[str],
    realized_vol: np.ndarray,
    output_path: Path,
    vol_threshold: float | None = None,
) -> dict:
    """Generate side-by-side beeswarm plots for high-vol vs low-vol regimes."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import shap

    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if test_data.ndim == 3:
        test_data_2d = test_data.squeeze(axis=1)
    else:
        test_data_2d = test_data

    if vol_threshold is None:
        vol_threshold = float(np.median(realized_vol))

    high_vol_mask = realized_vol >= vol_threshold
    low_vol_mask = realized_vol < vol_threshold

    n_high = int(high_vol_mask.sum())
    n_low = int(low_vol_mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # High-vol regime
    plt.sca(axes[0])
    if n_high > 0:
        expl_high = shap.Explanation(
            values=shap_values[high_vol_mask],
            data=test_data_2d[high_vol_mask],
            feature_names=feature_names,
        )
        shap.plots.beeswarm(expl_high, max_display=min(15, len(feature_names)), show=False)
    axes[0].set_title(
        f"High Volatility Regime (n={n_high})\n(vol >= {vol_threshold:.2f} annualized)",
        fontsize=12, fontweight="bold",
    )

    # Low-vol regime
    plt.sca(axes[1])
    if n_low > 0:
        expl_low = shap.Explanation(
            values=shap_values[low_vol_mask],
            data=test_data_2d[low_vol_mask],
            feature_names=feature_names,
        )
        shap.plots.beeswarm(expl_low, max_display=min(15, len(feature_names)), show=False)
    axes[1].set_title(
        f"Low Volatility Regime (n={n_low})\n(vol < {vol_threshold:.2f} annualized)",
        fontsize=12, fontweight="bold",
    )

    fig.suptitle("SHAP Regime Comparison: Feature Importance by Volatility",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved {output_path}")

    return {
        "vol_threshold": vol_threshold,
        "n_low_vol": n_low,
        "n_high_vol": n_high,
    }


# ---------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------
def run_shap_analysis(
    model: tf.keras.Model,
    train_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: list[str],
    test_close: np.ndarray,
    output_dir: Path,
) -> dict:
    """Run full SHAP analysis: compute values, summary plot, regime comparison.

    Parameters
    ----------
    model : tf.keras.Model
    train_x : np.ndarray
        Training data (used for background).
    test_x : np.ndarray
        Test data to explain.
    feature_names : list[str]
    test_close : np.ndarray
        Close prices for the test window (used to compute realized vol).
    output_dir : Path
        Where to save the plots.

    Returns
    -------
    dict with keys: explainer_used, shap_values_shape, regime_info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute SHAP values
    shap_values, used = compute_shap_values(
        model=model,
        background=train_x,
        test_data=test_x,
        feature_names=feature_names,
    )

    # 1. Summary plot
    plot_shap_summary(
        shap_values=shap_values,
        test_data=test_x,
        feature_names=feature_names,
        output_path=output_dir / "shap_summary.png",
        title="SHAP Feature Importance — Optuna-Tuned LSTM (Test Set)",
    )

    # 2. Regime comparison
    # Compute 20-day realized vol aligned with test_x
    rets = np.diff(test_close) / test_close[:-1]
    # Pad to align with test_close
    rets_padded = np.concatenate([[0], rets])
    # Use min_periods=2 (std needs at least 2 data points)
    vol_series = pd.Series(rets_padded).rolling(20, min_periods=2).std().values * np.sqrt(252)
    # Forward-fill any remaining NaNs at the start
    vol_series = pd.Series(vol_series).ffill().bfill().values
    # Align with test_x length
    if len(vol_series) > len(shap_values):
        vol_series = vol_series[: len(shap_values)]
    elif len(vol_series) < len(shap_values):
        vol_series = np.pad(vol_series, (0, len(shap_values) - len(vol_series)), mode="edge")

    regime_info = plot_regime_comparison(
        shap_values=shap_values,
        test_data=test_x,
        feature_names=feature_names,
        realized_vol=vol_series,
        output_path=output_dir / "shap_regime_comparison.png",
    )

    # 3. Mean absolute SHAP per feature (global importance ranking)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    print("\nTop 10 features by mean |SHAP|:")
    print(importance_df.head(10).to_string(index=False))

    return {
        "explainer_used": used,
        "shap_values_shape": shap_values.shape,
        "regime_info": regime_info,
    }
