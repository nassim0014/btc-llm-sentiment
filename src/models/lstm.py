"""
Parametric LSTM model factory for Optuna hyperparameter search.

Public API
----------
- `build_lstm_with_params`: build a compiled Keras LSTM model from a
  dictionary of hyperparameters.

The hyperparameter schema matches the Optuna search space:
    lr:         float in [1e-4, 1e-2] (log-uniform)
    units:      int in {32, 64, 128}
    dropout:    float in {0.0, 0.2, 0.4}
    num_layers: int in {1, 2}
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_lstm_with_params(
    lr: float,
    units: int,
    dropout: float,
    num_layers: int,
    n_features: int,
    l2_reg: float = 0.0,
    dense_units: int = 16,
) -> tf.keras.Model:
    """Build a compiled LSTM classifier from hyperparameters.

    Parameters
    ----------
    lr : float
        Adam learning rate.
    units : int
        LSTM units per layer (same for all layers).
    dropout : float
        LSTM dropout + recurrent dropout. Applied uniformly.
    num_layers : int
        Number of stacked LSTM layers (1 or 2). All but the last layer
        return_sequences=True.
    n_features : int
        Number of input features.
    l2_reg : float
        L2 kernel regularizer strength. Default 0.0 (no regularization).
    dense_units : int
        Dense layer units between LSTM and output. Default 16.

    Returns
    -------
    tf.keras.Model
        Compiled model with sigmoid output for binary classification.
    """
    inp = layers.Input(shape=(1, n_features), name="features")
    x = inp

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)  # last layer returns sequences=False
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=dropout,
            recurrent_dropout=dropout,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f"lstm_{i+1}",
        )(x)

    x = layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_up")(x)

    model = models.Model(inp, out, name=f"lstm_u{units}_l{num_layers}_d{dropout}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_model_factory(hp: dict, n_features: int):
    """Return a zero-arg callable that builds a fresh model from `hp`.

    This is the interface expected by `run_walk_forward(model_factory=...)`.
    """
    def factory() -> tf.keras.Model:
        """Build and return a fresh compiled LSTM model from the captured hyperparameters."""
        return build_lstm_with_params(
            lr=hp["lr"],
            units=hp["units"],
            dropout=hp["dropout"],
            num_layers=hp["num_layers"],
            n_features=n_features,
            l2_reg=hp.get("l2_reg", 0.0),
            dense_units=hp.get("dense_units", 16),
        )
    return factory
