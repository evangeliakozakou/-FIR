import numpy as np
import matplotlib.pyplot as plt


def generate_signal(num_bits, bit_rate, samples_per_bit, modulation="TRUE", seed=1):
   #Δημιουργία φίλτρου μετάδοσης
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_bits)
    #Δημιουργία ίδιου μήκους τυχαίας σειράς 1 , 0 bits

    if modulation == "TRUE":
        symbols = bits.astype(float)

    signal = np.repeat(symbols, samples_per_bit)

    Tb = 1 / bit_rate
    Ts = Tb / samples_per_bit
    Fs = 1 / Ts
    t = np.arange(len(signal)) * Ts

    return bits, signal, t, Ts, Tb, Fs



def apply_dispersion(signal, D, wavelength, fiber_length, Ts):
    
    #Χρωματική διασπορά στο πεδίο της συχνότητας.
    
    c = 299792458.0
    N = len(signal)
    f = np.fft.fftfreq(N, d=Ts)
    omega = 2 * np.pi * f

    H = np.exp(-1j * (D * wavelength**2 * fiber_length / (4 * np.pi * c)) * omega**2)

    Y = np.fft.ifft(np.fft.fft(signal) * H)

    return np.real(Y), H


def design_fir(D, wavelength, fiber_length, Ts):
#Σχεδίαση FIR φίλτρου αντιστάθμισης χρωματικής διασποράς.
    c = 299792458.0

    N = 2 * int(np.floor((abs(D) * wavelength**2 * fiber_length) / (2 * c * Ts**2))) + 1

    half = N // 2
    k = np.arange(-half, half + 1)   

    b = np.sqrt((1j * c * Ts**2) / (D * wavelength**2 * fiber_length)) * \
        np.exp(-1j * (np.pi * c * Ts**2 / (D * wavelength**2 * fiber_length)) * (k**2))

    return b, len(b), k


def fir_filter(signal, b):
    y = np.convolve(signal, b, mode="same")
    return np.real(y)



def new_threshold(signal, samples_per_bit, offset=None, threshold=None, num_bits=None):
    
    if offset is None:
        offset = samples_per_bit // 2    #Παίρνω το μεσαίο bit (8o) επειδή είναι πιο σταθερό

    sampled = signal[offset::samples_per_bit]  #Μεσαίο δείγμα από όλα τα bits

    if num_bits is not None:
        sampled = sampled[:num_bits]

    
    if threshold is None:
        low_level = np.percentile(sampled, 10)
        #λόγο διασποράς το threshold δεν είναι 0.5 άρα το υπολογίζω σύμφωνα με το amplitude 
        high_level = np.percentile(sampled, 90)
        threshold = 0.5 * (low_level + high_level)

    detected_bits = (sampled >= threshold).astype(int)

    return sampled, detected_bits, threshold


def count_symbol_errors(tx_bits, rx_bits):
    
    if len(tx_bits) != len(rx_bits):
        raise ValueError(
            f"Διαφορετικό πλήθος bits: tx={len(tx_bits)}, rx={len(rx_bits)}"
        )

    return int(np.sum(tx_bits != rx_bits))


def plot_time(signal, t, title, num_symbols_to_show, samples_per_bit):
    n_show = num_symbols_to_show * samples_per_bit
    plt.figure(figsize=(10, 4))
    plt.plot(t[:n_show], signal[:n_show])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_eye(signal, samples_per_bit, title, traces=100):
    plt.figure(figsize=(6, 4))

    eye_len = 2 * samples_per_bit
    max_segments = min(traces, len(signal) // samples_per_bit - 2)

    for i in range(max_segments):
        start = i * samples_per_bit
        segment = signal[start:start + eye_len]

        if len(segment) == eye_len:
            x = np.arange(eye_len) / samples_per_bit - 0.5
            plt.plot(x, segment, alpha=0.5)

    plt.title(title)
    plt.xlabel("Time (symbol periods)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_beta2(D, wavelength):
    # Μετατροπή από D σε beta2
    # beta2 = -(D * λ^2) / (2πc)
    c = 299792458.0
    beta2 = -(D * wavelength**2) / (2 * np.pi * c)
    return beta2


def compute_dispersion_length(Tb, beta2):
    # Ερώτημα β:
    # T0 = Tsymbol = Tb
    T0 = Tb
    L_D = (T0**2) / abs(beta2)
    return T0, L_D


def run_case(case_name, Di, wavelength_nm, bit_rate_Gbps, samples_per_bit, fiber_km,
             num_bits=512, seed=1, show_plots=True):

    # Μετατροπές μονάδων
    D = Di * 1e-6
    wavelength = wavelength_nm * 1e-9
    bit_rate = bit_rate_Gbps * 1e9
    fiber_length = fiber_km * 1e3

    #Αρχικό σήμα
    bits, tx_signal, t, Ts, Tb, Fs = generate_signal(
        num_bits=num_bits,
        bit_rate=bit_rate,
        samples_per_bit=samples_per_bit,
        modulation="TRUE",
        seed=seed
    )

   
    # Υπολογισμοσ β2
    beta2 = compute_beta2(D, wavelength)
    T0, L_D = compute_dispersion_length(Tb, beta2)
    dispersion_ratio = fiber_length / L_D

    #Σχεδίαση FIR
    b, N_taps, k = design_fir(
        D=D,
        wavelength=wavelength,
        fiber_length=fiber_length,
        Ts=Ts
    )

    #(Padding γιατι εχουμε πολλα σφαλματα χωρις)
    pad = max(4 * samples_per_bit, 2 * len(b))

    tx_padded = np.pad(tx_signal, (pad, pad), mode="constant")

    #Διασπορά
    rx_disp_padded, H = apply_dispersion(
        signal=tx_padded,
        D=D,
        wavelength=wavelength,
        fiber_length=fiber_length,
        Ts=Ts
    )

    
    rx_comp_padded = fir_filter(rx_disp_padded, b)


    # Κόβουμε πίσω το κεντρικό κομμάτι που αντιστοιχεί στο αρχικό σήμα
    rx_disp = rx_disp_padded[pad:pad + len(tx_signal)]
    rx_comp = rx_comp_padded[pad:pad + len(tx_signal)]


    # Δειγματοληψία μεσαίου δείγματος και απόφαση
    sampled_disp, bits_after_disp, thr_disp = new_threshold(
        rx_disp,
        samples_per_bit=samples_per_bit,
        num_bits=len(bits)
    )

    sampled_comp, bits_after_comp, thr_comp = new_threshold(
        rx_comp,
        samples_per_bit=samples_per_bit,
        num_bits=len(bits)
    )

    #Σφάλματα
    errors_without = count_symbol_errors(bits, bits_after_disp)
    errors_with = count_symbol_errors(bits, bits_after_comp)

    if show_plots:
        plot_time(tx_signal, t, f"{case_name} - Αρχικό σήμα", 20, samples_per_bit)
        plot_eye(tx_signal, samples_per_bit, f"{case_name} - Eye diagram αρχικού σήματος")

        plot_time(rx_disp, t, f"{case_name} - Με διασπορά", 20, samples_per_bit)
        plot_eye(rx_disp, samples_per_bit, f"{case_name} - Eye diagram με διασπορά")

        plot_time(rx_comp, t, f"{case_name} - Μετά το FIR", 20, samples_per_bit)
        plot_eye(rx_comp, samples_per_bit, f"{case_name} - Eye diagram μετά το FIR")

    results = {
        "case_name": case_name,
        "Di": Di,
        "wavelength_nm": wavelength_nm,
        "bit_rate_Gbps": bit_rate_Gbps,
        "samples_per_bit": samples_per_bit,
        "fiber_km": fiber_km,
        "Ts": Ts,
        "Tb": Tb,
        "Fs": Fs,
        "beta2": beta2,
        "T0": T0,
        "L_D_m": L_D,
        "L_D_km": L_D / 1e3,
        "fiber_length_m": fiber_length,
        "fiber_length_km": fiber_length / 1e3,
        "L_over_LD": dispersion_ratio,
        "filter_taps": len(b),
        "threshold_disp": float(thr_disp),
        "threshold_comp": float(thr_comp),
        "errors_without_fir": int(errors_without),
        "errors_with_fir": int(errors_with),
    }

    return results


def main():
    cases = [
        {"case_name": "Case 1", "Di": 17, "wavelength_nm": 1550, "bit_rate_Gbps": 10, "samples_per_bit": 16, "fiber_km": 50},
        {"case_name": "Case 2", "Di": 17, "wavelength_nm": 1550, "bit_rate_Gbps": 32, "samples_per_bit": 16, "fiber_km": 50},
        {"case_name": "Case 3", "Di": 20, "wavelength_nm": 1550, "bit_rate_Gbps": 32, "samples_per_bit": 16, "fiber_km": 100},
        {"case_name": "Case 4", "Di": 20, "wavelength_nm": 1550, "bit_rate_Gbps": 32, "samples_per_bit": 16, "fiber_km": 400},
    ]

    all_results = []

    for case in cases:
        results = run_case(**case, num_bits=512, seed=1, show_plots=True)
        all_results.append(results)

        print("\n" + "=" * 50)
        print(results["case_name"])

        
    
        print(f"L_D = {results['L_D_m']:.4e} m = {results['L_D_km']:.4f} km")
        print(f"L   = {results['fiber_length_m']:.4e} m = {results['fiber_length_km']:.4f} km")
       


        print(f"\nFilter taps: {results['filter_taps']}")
        print(f"Threshold without FIR: {results['threshold_disp']:.4f}")
        print(f"Threshold with FIR:    {results['threshold_comp']:.4f}")
        print(f"Errors without FIR:    {results['errors_without_fir']}")
        print(f"Errors with FIR:       {results['errors_with_fir']}")

    return all_results


if __name__ == "__main__":
    main()