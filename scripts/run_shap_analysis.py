"""Run SHAP interpretability analysis on the best Optuna model.

Loads the saved model, computes SHAP values, generates summary + regime plots.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INTERIM = ROOT / "notebooks" / "interim"
OUTPUTS = ROOT / "outputs"


def main():
    print("=" * 60)
    print("SHAP Interpretability Analysis")
    print("=" * 60)

    # Load features — use safe loader with SHA256 integrity check
    from src.utils.safe_pickle import safe_load_bundle
    bundle = safe_load_bundle()

    train_x = bundle["train_x"]
    test_x = bundle["test_x"]
    test_close = bundle["test_close"]
    feature_names = bundle["feature_cols"]

    # Load trained model
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(INTERIM / "best_optuna_model.keras")
    print(f"Model loaded: {model.name}")
    print(f"Train X: {train_x.shape}  Test X: {test_x.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    from src.interpretability.shap_explainer import run_shap_analysis

    result = run_shap_analysis(
        model=model,
        train_x=train_x,
        test_x=test_x,
        feature_names=feature_names,
        test_close=test_close,
        output_dir=OUTPUTS,
    )

    print(f"\n{'='*60}")
    print("SHAP Analysis Complete")
    print(f"  Explainer used: {result['explainer_used']}")
    print(f"  SHAP values shape: {result['shap_values_shape']}")
    print(f"  Regime split: {result['regime_info']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
