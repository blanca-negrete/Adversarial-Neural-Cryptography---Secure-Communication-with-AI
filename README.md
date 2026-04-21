# Adversarial-Neural-Cryptography - Blanca Negrete & Molly Corgan
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

All models operate on **16-bit binary vectors encoded as ±1 floats** and use a Dense → Reshape → Conv1D pipeline.

| Model | Input | Output | Role |
|-------|-------|--------|------|
| Alice | message ‖ key (32-d) | ciphertext (16-d, tanh) | Encrypt |
| Bob   | ciphertext ‖ key (32-d) | plaintext (16-d, tanh) | Decrypt |
| Eve   | ciphertext only (16-d) | plaintext guess (16-d) | Attack |

## Training Loop (Phase 2)

The alternating adversarial loop per epoch:

1. **Train Eve** — Minimizes reconstruction error (MSE) between predicted and true plaintext
2. **Freeze Eve, train Alice+Bob** — Minimizes Bob reconstruction loss
  - Adds pressure to reduce Eve’s ability to decode ciphertext

This incentivises Alice and Bob to develop an encryption scheme that is
simultaneously decodable by Bob (who has the key) and unintelligible to Eve.

## Hyperparameters (Phase 3)

Tunable in `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BATCH_SIZE` | 64 | try 32 if unstable |
| `N_EPOCHS` | 60 | increase if Bob < 95% |
| `LR_EVE` | 1e-3 | reduce if Eve diverges |
| `LR_AB` | 8e-4 | reduce if Bob unstable |
| `EVE_STEPS` | 2 | Eve inner updates per step |

## Success Criteria (Phase 4)

- **Bob bit accuracy ≥ 95%** on 1,000 unseen messages
- **Eve bit accuracy ≈ 50%** (random baseline)

## Output Plots (Phase 4)

| File | Description |
|------|-------------|
| `plots/training_curves.png` | Bob & Eve accuracy + loss over epochs |
| `plots/final_accuracy_bar.png` | Bar chart — final Bob vs Eve accuracy |
| `plots/per_bit_accuracy.png` | Per-bit accuracy across all 16 positions |
| `plots/history_summary.png` | Validation accuracy dynamics |

## Training Results Summary

### Bob vs Eve over time

- Bob accuracy rapidly increases and stabilizes around **~99%**
- Eve accuracy starts near random (~50–55%) and stabilizes around **~58–60%**

### Key insight

- Bob successfully learns to decrypt with near-perfect accuracy
- Eve improves but does not reach full random equilibrium (50%)
- The system reaches a **stable adversarial balance**, not perfect cryptographic security

## Adversarial Training Insights

- Alice and Bob learn a shared encoding scheme that hides structure from Eve
- Eve extracts partial statistical information but cannot fully decode messages
- Training converges to a **competitive equilibrium**
- Ciphertext still contains some exploitable patterns (Eve > 50%)

## Limitations

When using AI generated encryption there can be several limitations to the program. The AI models often prioritze code functionality over safe and secure code. When prioritizing code functionality it can lead to a number of vulnerabilites that put your security at risk. Another limitation of AI generated encryption is it is non-deterministic. Building off of the lack of security, the AI can produce the code that works for testing purposes but the functionallity of real world often is insecure. Lastly, data privacy can be a concern. In real scenarios of companys using the AI tools to generate encryption code any company information will be stored as AI training methods. If there were to ever be a leak in the system your company data that was imputed could be at risk for exposure.