"""
Phase 1.3 — Model Architectures
Alice, Bob, and Eve neural networks built with the Keras Functional API.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MSG_BITS = 16
KEY_BITS = 16
CIPHER_BITS = 16


def build_alice(msg_bits: int = MSG_BITS, key_bits: int = KEY_BITS,
                cipher_bits: int = CIPHER_BITS, name: str = "Alice") -> keras.Model:
    """
    Alice: Encryptor.
    Input:  (message ‖ key)  →  concatenated vector of length msg+key bits
    Output: ciphertext of length cipher_bits in [-1, +1]
    """
    inp_msg = keras.Input(shape=(msg_bits,),  name="message")
    inp_key = keras.Input(shape=(key_bits,),  name="key")
    x = layers.Concatenate(name="concat_msg_key")([inp_msg, inp_key])

    x = layers.Dense(128, activation="selu", name="alice_fc1")(x)
    x = layers.Dense(128, activation="selu", name="alice_fc2")(x)
    x = layers.Reshape((cipher_bits, 8),     name="alice_reshape")(x)   # (16, 8)
    x = layers.Conv1D(2, 4, padding="same",  activation="selu", name="alice_conv1")(x)
    x = layers.Conv1D(4, 2, padding="same",  activation="selu", name="alice_conv2")(x)
    x = layers.Conv1D(4, 1, padding="same",  activation="selu", name="alice_conv3")(x)
    x = layers.Conv1D(1, 1, padding="same",  activation="tanh", name="alice_out")(x)
    out = layers.Flatten(name="ciphertext")(x)                            # (16,)

    return keras.Model(inputs=[inp_msg, inp_key], outputs=out, name=name)


def build_bob(key_bits: int = KEY_BITS, cipher_bits: int = CIPHER_BITS,
              msg_bits: int = MSG_BITS, name: str = "Bob") -> keras.Model:
    """
    Bob: Decryptor (has the key).
    Input:  (ciphertext ‖ key)
    Output: reconstructed plaintext of length msg_bits in [-1, +1]
    """
    inp_cipher = keras.Input(shape=(cipher_bits,), name="ciphertext")
    inp_key    = keras.Input(shape=(key_bits,),    name="key")
    x = layers.Concatenate(name="concat_cipher_key")([inp_cipher, inp_key])

    x = layers.Dense(128, activation="selu", name="bob_fc1")(x)
    x = layers.Dense(128, activation="selu", name="bob_fc2")(x)
    x = layers.Reshape((msg_bits, 8),        name="bob_reshape")(x)
    x = layers.Conv1D(2, 4, padding="same",  activation="selu", name="bob_conv1")(x)
    x = layers.Conv1D(4, 2, padding="same",  activation="selu", name="bob_conv2")(x)
    x = layers.Conv1D(4, 1, padding="same",  activation="selu", name="bob_conv3")(x)
    x = layers.Conv1D(1, 1, padding="same",  activation="tanh", name="bob_out")(x)
    out = layers.Flatten(name="plaintext_hat")(x)

    return keras.Model(inputs=[inp_cipher, inp_key], outputs=out, name=name)


def build_eve(cipher_bits: int = CIPHER_BITS, msg_bits: int = MSG_BITS,
              name: str = "Eve") -> keras.Model:
    """
    Eve: Eavesdropper (no key).
    Input:  ciphertext only
    Output: guessed plaintext of length msg_bits in [-1, +1]
    """
    inp_cipher = keras.Input(shape=(cipher_bits,), name="ciphertext")
    x = layers.Reshape((cipher_bits, 1),     name="eve_reshape")(inp_cipher)

    x = layers.Conv1D(2, 4, padding="same",  activation="selu", name="eve_conv1")(x)
    x = layers.Conv1D(4, 2, padding="same",  activation="selu", name="eve_conv2")(x)
    x = layers.Conv1D(4, 1, padding="same",  activation="selu", name="eve_conv3")(x)
    x = layers.Conv1D(1, 1, padding="same",  activation="selu", name="eve_conv4")(x)
    x = layers.Flatten(name="eve_flat")(x)
    x = layers.Dense(128, activation="selu", name="eve_fc1")(x)
    x = layers.Dense(128, activation="selu", name="eve_fc2")(x)
    out = layers.Dense(msg_bits, activation="tanh", name="guess")(x)

    return keras.Model(inputs=inp_cipher, outputs=out, name=name)


if __name__ == "__main__":
    alice = build_alice()
    bob   = build_bob()
    eve   = build_eve()

    print("=== Alice ===")
    alice.summary()
    print("\n=== Bob ===")
    bob.summary()
    print("\n=== Eve ===")
    eve.summary()