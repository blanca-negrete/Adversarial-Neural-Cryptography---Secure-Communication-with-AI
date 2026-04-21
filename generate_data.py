"""
Phase 1.2 — Synthetic Data Generation
Generates random binary messages and keys (16-bit), saves to .npy files.
"""

import numpy as np
import os

MSG_BITS  = 16
KEY_BITS  = 16
N_TRAIN   = 50_000
N_VAL     = 5_000
N_TEST    = 1_000
SEED      = 42
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")


def generate_dataset(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (messages, keys) arrays of shape (n_samples, 16) with values in {-1, +1}."""
    messages = rng.integers(0, 2, size=(n_samples, MSG_BITS)).astype(np.float32) * 2 - 1
    keys     = rng.integers(0, 2, size=(n_samples, KEY_BITS)).astype(np.float32) * 2 - 1
    return messages, keys


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    splits = {
        "train": N_TRAIN,
        "val":   N_VAL,
        "test":  N_TEST,
    }

    for split, n in splits.items():
        messages, keys = generate_dataset(n, rng)
        np.save(os.path.join(DATA_DIR, f"{split}_messages.npy"), messages)
        np.save(os.path.join(DATA_DIR, f"{split}_keys.npy"),     keys)
        print(f"[{split:>5}]  messages {messages.shape}  keys {keys.shape}")

    print(f"\nData saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()