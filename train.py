"""
Phase 2 — Model Implementation & Adversarial Training Loop
Covers Tasks 2.1 (forward-pass sanity check), 2.2 (alternating training), 2.3 (metric logging).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_alice, build_bob, build_eve, MSG_BITS, KEY_BITS, CIPHER_BITS

# ── Hyper-parameters (Phase 3.1 tuning knobs) ──────────────────────────────
BATCH_SIZE       = 256
N_EPOCHS         = 30          # outer loop epochs
EVE_STEPS        = 2           # Eve inner update steps per outer step
AB_STEPS         = 1           # Alice+Bob inner update steps per outer step
LR_EVE           = 1e-3
LR_AB            = 8e-4
DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
LOGS_DIR         = os.path.join(os.path.dirname(__file__), "logs")
PLOTS_DIR        = os.path.join(os.path.dirname(__file__), "plots")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(LOGS_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Loss helpers ────────────────────────────────────────────────────────────

def reconstruction_loss(y_true, y_pred):
    """Mean squared error over bits (values in [-1, +1])."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def bit_accuracy(y_true, y_pred):
    """Fraction of bits correctly reconstructed (sign match)."""
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)
    )


# ── Forward pass sanity test (Task 2.1) ─────────────────────────────────────

def sanity_check(alice, bob, eve):
    print("\n── Sanity Check (Task 2.1) ──")
    rng = np.random.default_rng(0)
    msg = (rng.integers(0, 2, (4, MSG_BITS)).astype(np.float32) * 2 - 1)
    key = (rng.integers(0, 2, (4, KEY_BITS)).astype(np.float32) * 2 - 1)

    cipher  = alice([msg, key], training=False)
    cipher_np = cipher.numpy()
    bob_out = bob([cipher_np, key], training=False)
    eve_out = eve(cipher_np, training=False)

    print(f"  Alice ciphertext   shape: {cipher.shape}  range [{cipher.numpy().min():.2f}, {cipher.numpy().max():.2f}]")
    print(f"  Bob reconstruction shape: {bob_out.shape}")
    print(f"  Eve guess          shape: {eve_out.shape}")
    bob_acc = bit_accuracy(msg, bob_out).numpy()
    eve_acc = bit_accuracy(msg, eve_out).numpy()
    print(f"  Bob bit accuracy (untrained): {bob_acc:.3f}")
    print(f"  Eve bit accuracy (untrained): {eve_acc:.3f}\n")


# ── Adversarial training (Task 2.2) ─────────────────────────────────────────

@tf.function
def train_eve_step(alice, eve, opt_eve, msgs, keys):
    """Train Eve to minimise reconstruction loss on ciphertexts."""
    with tf.GradientTape() as tape:
        cipher    = alice([msgs, keys], training=False)
        eve_guess = eve(cipher, training=True)
        loss_eve  = reconstruction_loss(msgs, eve_guess)
    grads = tape.gradient(loss_eve, eve.trainable_variables)
    opt_eve.apply_gradients(zip(grads, eve.trainable_variables))
    return loss_eve, bit_accuracy(msgs, eve_guess)


@tf.function
def train_ab_step(alice, bob, eve, opt_ab, msgs, keys):
    """
    Train Alice + Bob jointly.
    Alice/Bob loss = Bob's reconstruction loss
                   + penalty when Eve's loss is TOO LOW  (push Eve toward random).
    We want Eve's loss to stay near the 'chance' level (MSG_BITS / 4 for MSE on ±1 data).
    """
    chance_loss = tf.constant(1.0, dtype=tf.float32)   # MSE of pure-guess baseline
    with tf.GradientTape() as tape:
        cipher    = alice([msgs, keys], training=True)
        bob_guess = bob([cipher, keys], training=True)
        eve_guess = eve(cipher,         training=False)

        loss_bob  = reconstruction_loss(msgs, bob_guess)
        loss_eve  = reconstruction_loss(msgs, eve_guess)

        # Penalise Alice/Bob when Eve does better than chance
        eve_penalty = tf.maximum(0.0, chance_loss - loss_eve)
        total_loss  = loss_bob + eve_penalty

    ab_vars = alice.trainable_variables + bob.trainable_variables
    grads = tape.gradient(total_loss, ab_vars)
    opt_ab.apply_gradients(zip(grads, ab_vars))
    return loss_bob, bit_accuracy(msgs, bob_guess), loss_eve, bit_accuracy(msgs, eve_guess)


def run_training(alice, bob, eve, train_msgs, train_keys, val_msgs, val_keys):
    opt_eve = keras.optimizers.Adam(learning_rate=LR_EVE)
    opt_ab  = keras.optimizers.Adam(learning_rate=LR_AB)

    n = len(train_msgs)
    history = {k: [] for k in
               ("bob_loss", "bob_acc", "eve_loss", "eve_acc",
                "val_bob_acc", "val_eve_acc")}

    print(f"Training  {N_EPOCHS} epochs | batch {BATCH_SIZE} | {n} samples")
    print("─" * 70)

    for epoch in range(1, N_EPOCHS + 1):
        # Shuffle
        idx = np.random.permutation(n)
        msgs_s, keys_s = train_msgs[idx], train_keys[idx]

        ep_bob_loss = ep_bob_acc = ep_eve_loss = ep_eve_acc = 0.0
        n_batches = 0

        for start in range(0, n, BATCH_SIZE):
            mb_msg = msgs_s[start:start + BATCH_SIZE]
            mb_key = keys_s[start:start + BATCH_SIZE]

            # 1) Train Eve
            for _ in range(EVE_STEPS):
                e_loss, e_acc = train_eve_step(alice, eve, opt_eve, mb_msg, mb_key)

            # 2) Train Alice + Bob
            for _ in range(AB_STEPS):
                b_loss, b_acc, e_loss2, e_acc2 = train_ab_step(
                    alice, bob, eve, opt_ab, mb_msg, mb_key)

            ep_bob_loss += b_loss.numpy()
            ep_bob_acc  += b_acc.numpy()
            ep_eve_loss += e_loss2.numpy()
            ep_eve_acc  += e_acc2.numpy()
            n_batches   += 1

        # ── Epoch averages ──
        ep_bob_loss /= n_batches
        ep_bob_acc  /= n_batches
        ep_eve_loss /= n_batches
        ep_eve_acc  /= n_batches

        # ── Validation ──
        val_cipher     = alice([val_msgs, val_keys], training=False)
        val_cipher_np  = val_cipher.numpy()
        val_bob_acc    = bit_accuracy(val_msgs, bob([val_cipher_np, val_keys], training=False)).numpy()
        val_eve_acc    = bit_accuracy(val_msgs, eve(val_cipher_np, training=False)).numpy()

        for k, v in zip(history.keys(),
                        [ep_bob_loss, ep_bob_acc, ep_eve_loss, ep_eve_acc,
                         val_bob_acc, val_eve_acc]):
            history[k].append(v)

        print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
              f"bob_loss={ep_bob_loss:.4f}  bob_acc={ep_bob_acc:.3f}  "
              f"eve_loss={ep_eve_loss:.4f}  eve_acc={ep_eve_acc:.3f}  "
              f"| val_bob={val_bob_acc:.3f}  val_eve={val_eve_acc:.3f}")

    # Save metrics
    np.save(os.path.join(LOGS_DIR, "history.npy"), history)
    print(f"\nHistory saved → {LOGS_DIR}/history.npy")
    return history


# ── Plotting (Task 2.3) ─────────────────────────────────────────────────────

def plot_training_curves(history: dict):
    epochs = range(1, len(history["bob_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Adversarial Training Curves", fontsize=14, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, history["bob_acc"],     label="Bob (train)", color="#2196F3")
    ax.plot(epochs, history["val_bob_acc"], label="Bob (val)",   color="#2196F3", linestyle="--")
    ax.plot(epochs, history["eve_acc"],     label="Eve (train)", color="#F44336")
    ax.plot(epochs, history["val_eve_acc"], label="Eve (val)",   color="#F44336", linestyle="--")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", label="Random baseline (0.5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bit Accuracy")
    ax.set_title("Bit Accuracy over Training")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.05)
    ax.grid(alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(epochs, history["bob_loss"], label="Bob loss", color="#2196F3")
    ax.plot(epochs, history["eve_loss"], label="Eve loss", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Reconstruction Loss over Training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load data
    def load(split):
        m = np.load(os.path.join(DATA_DIR, f"{split}_messages.npy"))
        k = np.load(os.path.join(DATA_DIR, f"{split}_keys.npy"))
        return m, k

    train_msgs, train_keys = load("train")
    val_msgs,   val_keys   = load("val")

    # Build models
    alice = build_alice()
    bob   = build_bob()
    eve   = build_eve()

    # Task 2.1 — forward pass
    sanity_check(alice, bob, eve)

    # Task 2.2 — adversarial training
    history = run_training(alice, bob, eve, train_msgs, train_keys, val_msgs, val_keys)

    # Save weights
    alice.save_weights(os.path.join(MODEL_DIR, "alice.weights.h5"))
    bob.save_weights(os.path.join(MODEL_DIR,   "bob.weights.h5"))
    eve.save_weights(os.path.join(MODEL_DIR,   "eve.weights.h5"))
    print("Weights saved → models/")

    # Task 2.3 — plot
    plot_training_curves(history)


if __name__ == "__main__":
    main()