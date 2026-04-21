"""
Phase 4.1 — Final Evaluation  (1,000 unseen messages/keys)
Phase 4.2 — Demo  (show Alice encrypts, Bob decrypts, Eve fails)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_alice, build_bob, build_eve, MSG_BITS, KEY_BITS

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
LOGS_DIR  = os.path.join(os.path.dirname(__file__), "logs")


def bit_accuracy(y_true, y_pred):
    return float(tf.reduce_mean(
        tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)).numpy())


def load_models():
    alice = build_alice()
    bob   = build_bob()
    eve   = build_eve()

    alice.load_weights(os.path.join(MODEL_DIR, "alice.weights.h5"))
    bob.load_weights(os.path.join(MODEL_DIR,   "bob.weights.h5"))
    eve.load_weights(os.path.join(MODEL_DIR,   "eve.weights.h5"))
    return alice, bob, eve


# ── Phase 4.1: Final Evaluation ─────────────────────────────────────────────

def evaluate(alice, bob, eve):
    test_msgs = np.load(os.path.join(DATA_DIR, "test_messages.npy"))
    test_keys = np.load(os.path.join(DATA_DIR, "test_keys.npy"))

    cipher    = alice([test_msgs, test_keys], training=False)
    bob_guess = bob([cipher, test_keys],      training=False)
    eve_guess = eve(cipher,                   training=False)

    bob_acc = bit_accuracy(test_msgs, bob_guess)
    eve_acc = bit_accuracy(test_msgs, eve_guess)

    print("\n══════════════════════════════════════════")
    print("          FINAL EVALUATION  (n=1,000)     ")
    print("══════════════════════════════════════════")
    print(f"  Bob bit accuracy : {bob_acc*100:.2f}%   (target > 95%)")
    print(f"  Eve bit accuracy : {eve_acc*100:.2f}%   (target ≈ 50%)")
    print("══════════════════════════════════════════\n")

    verdict_bob = "✓ PASS" if bob_acc >= 0.95 else "✗ needs more training"
    verdict_eve = "✓ PASS" if abs(eve_acc - 0.5) <= 0.05 else "✗ Eve may be learning"
    print(f"  Bob verdict: {verdict_bob}")
    print(f"  Eve verdict: {verdict_eve}\n")

    return bob_acc, eve_acc, test_msgs, test_keys, cipher, bob_guess, eve_guess


# ── Phase 4.2: Demo ──────────────────────────────────────────────────────────

def demo(alice, bob, eve, n_examples: int = 5):
    rng = np.random.default_rng(99)
    msgs = (rng.integers(0, 2, (n_examples, MSG_BITS)).astype(np.float32) * 2 - 1)
    keys = (rng.integers(0, 2, (n_examples, KEY_BITS)).astype(np.float32) * 2 - 1)

    cipher    = alice([msgs, keys], training=False).numpy()
    bob_out   = bob([cipher, keys], training=False).numpy()
    eve_out   = eve(cipher,         training=False).numpy()

    # Convert ±1 to 0/1 bits for display
    def to_bits(arr): return ((arr > 0).astype(int))

    print("════════════════════════════════════════════════════════════════════")
    print("  DEMO: Alice encrypts — Bob decrypts — Eve guesses")
    print("════════════════════════════════════════════════════════════════════")
    for i in range(n_examples):
        orig  = to_bits(msgs[i])
        bdec  = to_bits(bob_out[i])
        edec  = to_bits(eve_out[i])
        match_bob = np.sum(orig == bdec)
        match_eve = np.sum(orig == edec)
        print(f"\n  Example {i+1}:")
        print(f"    Original : {''.join(map(str, orig))}")
        print(f"    Bob dec  : {''.join(map(str, bdec))}  ({match_bob}/{MSG_BITS} bits correct)")
        print(f"    Eve guess: {''.join(map(str, edec))}  ({match_eve}/{MSG_BITS} bits correct)")


# ── Phase 4.3: Final Visualisations ─────────────────────────────────────────

def plot_accuracy_bar(bob_acc, eve_acc):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Bob (decryptor)", "Eve (eavesdropper)"],
                  [bob_acc * 100, eve_acc * 100],
                  color=["#2196F3", "#F44336"], width=0.4, zorder=2)
    ax.axhline(50,  color="gray",   linewidth=1.2, linestyle="--", label="Random baseline (50%)", zorder=1)
    ax.axhline(95,  color="#4CAF50", linewidth=1.2, linestyle="--", label="Bob target (95%)",    zorder=1)
    ax.set_ylabel("Bit Accuracy (%)")
    ax.set_title("Final Evaluation — Bob vs Eve")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar, val in zip(bars, [bob_acc * 100, eve_acc * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "final_accuracy_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved → {path}")


def plot_per_bit_accuracy(test_msgs, bob_guess, eve_guess):
    """Show accuracy per individual bit position."""
    correct_bob = (np.sign(test_msgs) == np.sign(bob_guess)).mean(axis=0) * 100
    correct_eve = (np.sign(test_msgs) == np.sign(eve_guess)).mean(axis=0) * 100

    x = np.arange(MSG_BITS)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, correct_bob, 0.35, label="Bob", color="#2196F3", alpha=0.85)
    ax.bar(x + 0.2, correct_eve, 0.35, label="Eve", color="#F44336", alpha=0.85)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Bit Accuracy — Bob vs Eve")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_bit_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-bit chart saved → {path}")


def plot_history_summary():
    """Reload and re-plot history for Phase 4.3 final report."""
    path = os.path.join(LOGS_DIR, "history.npy")
    if not os.path.exists(path):
        print("No history.npy found — skipping history plot.")
        return
    history = np.load(path, allow_pickle=True).item()
    epochs = range(1, len(history["bob_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [v * 100 for v in history["val_bob_acc"]],
            label="Bob (val)", color="#2196F3", linewidth=2)
    ax.plot(epochs, [v * 100 for v in history["val_eve_acc"]],
            label="Eve (val)", color="#F44336", linewidth=2)
    ax.axhline(50, color="gray", linewidth=1, linestyle="--", label="Random baseline")
    ax.fill_between(epochs,
                    [v * 100 for v in history["val_bob_acc"]],
                    [v * 100 for v in history["val_eve_acc"]],
                    alpha=0.08, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Bit Accuracy (%)")
    ax.set_title("Training Dynamics — Validation Accuracy over Epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path_out = os.path.join(PLOTS_DIR, "history_summary.png")
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"History summary saved → {path_out}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    alice, bob, eve = load_models()

    bob_acc, eve_acc, test_msgs, test_keys, cipher, bob_guess, eve_guess = evaluate(alice, bob, eve)
    demo(alice, bob, eve)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_accuracy_bar(bob_acc, eve_acc)
    plot_per_bit_accuracy(test_msgs, bob_guess, eve_guess)
    plot_history_summary()

    print("\nAll evaluation outputs written to plots/")


if __name__ == "__main__":
    main()