# Adversarial-Neural-Cryptography - Blanca Negrte & Molly Corgan
## Neural Cryptography — Adversarial AI Encryption

Three neural networks compete in an arms race: **Alice** encrypts, **Bob** decrypts,
and **Eve** tries to crack the cipher without the key.

## Project Structure

```
neural_crypto/
├── generate_data.py   Phase 1.2 — synthetic binary data generation
├── models.py          Phase 1.3 — Alice, Bob, Eve architectures
├── train.py           Phase 2   — adversarial training loop + metric logging
├── evaluate.py        Phase 4   — final evaluation, demo, and report plots
├── run_all.py         Master runner — executes all phases in sequence
├── data/              .npy datasets (train / val / test)
├── models/            saved model weights (.h5)
├── logs/              training history (history.npy)
└── plots/             output figures (PNG)
```

## Architecture

All three models use a **Dense → Reshape → Conv1D** pipeline operating on 16-bit
binary vectors encoded as ±1 floats.

| Model | Input | Output | Role |
|-------|-------|--------|------|
| Alice | message ‖ key (32-d) | ciphertext (16-d, tanh) | Encrypt |
| Bob   | ciphertext ‖ key (32-d) | plaintext (16-d, tanh) | Decrypt |
| Eve   | ciphertext only (16-d) | plaintext guess (16-d) | Attack |

## Training Loop (Phase 2.2)

The alternating adversarial loop per epoch:

1. **Train Eve** — minimise MSE between her guess and the true plaintext.
2. **Freeze Eve, train Alice+Bob** — minimise Bob's reconstruction error
   *plus* a penalty when Eve's loss drops below the random-guess baseline.

This incentivises Alice and Bob to develop an encryption scheme that is
simultaneously decodable by Bob (who has the key) and unintelligible to Eve.

## Hyperparameters (Phase 3.1)

Tunable in `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BATCH_SIZE` | 64 | try 32 if unstable |
| `N_EPOCHS` | 60 | increase if Bob < 95% |
| `LR_EVE` | 1e-3 | reduce if Eve diverges |
| `LR_AB` | 8e-4 | reduce if Bob unstable |
| `EVE_STEPS` | 2 | Eve inner updates per step |

## Success Criteria (Phase 4.1)

- **Bob bit accuracy ≥ 95%** on 1,000 unseen messages
- **Eve bit accuracy ≈ 50%** (random baseline)

## Output Plots (Phase 4.3)

| File | Description |
|------|-------------|
| `plots/training_curves.png` | Bob & Eve accuracy + loss over epochs |
| `plots/final_accuracy_bar.png` | Bar chart — final Bob vs Eve accuracy |
| `plots/per_bit_accuracy.png` | Per-bit accuracy across all 16 positions |
| `plots/history_summary.png` | Validation accuracy dynamics |

## Limitations

- The encryption is **not cryptographically proven secure** — it is an
  emergent, learned cipher with no formal security guarantees.
- Performance depends heavily on random seed and training duration.
- Eve can sometimes converge on a partial solution if given enough capacity
  or training steps — counter by increasing Alice/Bob network depth.
- 16-bit messages are a toy setting; real applications would require much
  longer keys and messages with proper threat modelling.