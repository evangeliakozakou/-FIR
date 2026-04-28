import numpy as np
import matplotlib.pyplot as plt


def generate_signal_pam4(num_bits, bit_rate, samples_per_symbol, seed=1):
  

    rng = np.random.default_rng(seed)

    # Αν τα bits είναι μονός αριθμός, βάζουμε άλλο ένα 0 στο τέλος
    if num_bits % 2 != 0:
        num_bits = num_bits + 1

    bits = rng.integers(0, 2, num_bits)

   
    bit_pairs = bits.reshape(-1, 2)

 
    symbols = []
    for pair in bit_pairs:
        b1, b2 = pair

        if b1 == 0 and b2 == 0:
            symbols.append(-3.0)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1.0)
        elif b1 == 1 and b2 == 0:
            symbols.append(1.0)
        else:
            symbols.append(3.0)

    symbols = np.array(symbols)

    # Επανάληψη κάθε συμβόλου σε πολλά δείγματα
    signal = np.repeat(symbols, samples_per_symbol)

    
    symbol_rate = bit_rate / 2
    Tsymbol = 1 / symbol_rate
    Ts = Tsymbol / samples_per_symbol
    Fs = 1 / Ts
    t = np.arange(len(signal)) * Ts

    return bits, symbols, signal, t, Ts, Tsymbol, Fs


def plot_time(signal, t, title, num_symbols_to_show, samples_per_symbol):
    n_show = num_symbols_to_show * samples_per_symbol

    plt.figure(figsize=(10, 4))
    plt.plot(t[:n_show], signal[:n_show])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_eye(signal, samples_per_symbol, title, traces=100):
    plt.figure(figsize=(6, 4))

    eye_len = 2 * samples_per_symbol
    max_segments = min(traces, len(signal) // samples_per_symbol - 2)

    for i in range(max_segments):
        start = i * samples_per_symbol
        segment = signal[start:start + eye_len]

        if len(segment) == eye_len:
            x = np.arange(eye_len) / samples_per_symbol - 0.5
            plt.plot(x, segment, alpha=0.5)

    plt.title(title)
    plt.xlabel("Time (symbol periods)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def sample_and_detect_pam4(signal, samples_per_symbol):
   
    center = samples_per_symbol // 2
    sampled = signal[center::samples_per_symbol]

    detected_symbols = np.zeros(len(sampled))

    for i, x in enumerate(sampled):
        if x < -2:
            detected_symbols[i] = -3
        elif x < 0:
            detected_symbols[i] = -1
        elif x < 2:
            detected_symbols[i] = 1
        else:
            detected_symbols[i] = 3

    return sampled, detected_symbols


def pam4_symbols_to_bits(symbols):
    """
    Mapping:
    -3 -> 00
    -1 -> 01
     1 -> 10
     3 -> 11
    """
    bits_out = []

    for s in symbols:
        if s == -3:
            bits_out.extend([0, 0])
        elif s == -1:
            bits_out.extend([0, 1])
        elif s == 1:
            bits_out.extend([1, 0])
        elif s == 3:
            bits_out.extend([1, 1])

    return np.array(bits_out)


''' εστω εκτέλεση με δεδομένα από τις περιπτώσεις '''
num_bits = 512
bit_rate = 10e9
samples_per_symbol = 16

bits, symbols, signal, t, Ts, Tsymbol, Fs = generate_signal_pam4(
    num_bits=num_bits,
    bit_rate=bit_rate,
    samples_per_symbol=samples_per_symbol,
    seed=1
)

plot_time(signal, t, "PAM-4 signal", 20, samples_per_symbol)
plot_eye(signal, samples_per_symbol, "PAM-4 Eye Diagram", 100)

sampled, detected_symbols = sample_and_detect_pam4(signal, samples_per_symbol)
rx_bits = pam4_symbols_to_bits(detected_symbols)



print("Αρχικά bits:", len(bits))
print("Σύμβολα PAM-4:", len(symbols))
