# pyright: reportMissingImports=false
"""
bb84_simulation.py

Single-file BB84 Quantum Key Distribution simulation using Qiskit.

Features:
- Alice generates random bits & bases
- Bob generates random measurement bases and measures qubits
- Sifting step (keep bits where bases match)
- QBER estimation by publicly comparing a random sample
- Final key extraction
- Noise simulation using Qiskit's NoiseModel (depolarizing)
- Sweep noise levels and plot QBER and key rate vs noise
- Simple comparison output (BB84 vs "Ideal classical PQC" placeholder)

Requirements:
- qiskit (with Aer): pip install qiskit
- matplotlib, numpy

Run:
    python bb84_simulation.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer, AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except Exception as e:
    print("Qiskit (with Aer) is required. Install with: pip install qiskit")
    raise

import random
from typing import List, Tuple, Dict

# ---------------------------
# Parameters (tweak as needed)
# ---------------------------
N = 200                      # number of qubits/trials per run
SAMPLE_FRACTION = 0.20       # fraction of sifted bits used to estimate QBER
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20,0.25,0.30,0.35,0.40,0.45,0.50]   # depolarizing probabilities to test
QBER_THRESHOLD = 0.11        # example threshold (e.g., 11% typical BB84 limit)
SEED = 42                    # random seed for reproducibility
SHOTS_PER_TRIAL = 1          # run each single-qubit circuit with 1 shot

random.seed(SEED)
np.random.seed(SEED)

# Absolute path of this script (so saves go next to the file even if run elsewhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------
# Utility functions
# ---------------------------
def generate_random_bits_and_bases(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random bits (0/1) and bases (0: rectilinear, 1: diagonal)."""
    bits = np.random.randint(0, 2, size=n)
    bases = np.random.randint(0, 2, size=n)
    return bits, bases


def prepare_single_qubit_circuit(bit: int, basis: int) -> QuantumCircuit:
    """
    Prepare a 1-qubit circuit encoding `bit` in `basis`.
    basis: 0 -> Z (|0>,|1>), 1 -> X (|+>,|->)
    """
    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)    # prepare |1> if bit=1 (in Z basis)
    if basis == 1:
        qc.h(0)    # convert to X-basis state if needed (|+> or |->)
    # we won't measure here; measurement/possible H for Bob will be added later
    return qc


def measure_in_basis_and_run(qc: QuantumCircuit, bob_basis: int, backend, noise_model=None) -> int:
    """
    Take a prepared single-qubit circuit, apply Bob's measurement basis (if needed),
    measure, execute on simulator (with optional noise model), and return measured bit.
    """
    qc = qc.copy()
    if bob_basis == 1:
        qc.h(0)   # rotate to X-basis for measurement
    qc.measure(0, 0)
    # Use backend.run instead of deprecated execute
    job = backend.run(qc, shots=SHOTS_PER_TRIAL, memory=True)
    result = job.result()
    mem = result.get_memory()
    return int(mem[0])


def build_depolarizing_noise_model(p: float) -> NoiseModel:
    """
    Create a simple noise model with single-qubit depolarizing error probability p.
    We attach the error to X and H operations (these are the gates we use).
    """
    nm = NoiseModel()
    if p <= 0:
        return nm
    # depolarizing_error requires the probability and number of qubits affected
    single_qubit_error = depolarizing_error(p, 1)
    # attach to 'x' and 'h' gates (the names used in our circuits)
    nm.add_all_qubit_quantum_error(single_qubit_error, ['x', 'h'])
    return nm


# ---------------------------
# Core BB84 routine
# ---------------------------
def run_bb84_once(n: int, sample_fraction: float, noise_prob: float, backend) -> Dict:
    """
    Run BB84 for `n` single-qubit trials.
    Returns dictionary with keys:
    - qber (float)
    - key_rate (float) fraction of original bits that remain as final key
    - sifted_key_len (int)
    - final_key (list of ints)
    - sifted_positions (list of indices kept)
    - detected_eavesdropper (bool)
    """
    # Generate random bits/bases
    alice_bits, alice_bases = generate_random_bits_and_bases(n)
    bob_bases = np.random.randint(0, 2, size=n)

    # Prepare circuits for Alice's encoding
    prepared_circuits = [prepare_single_qubit_circuit(int(alice_bits[i]), int(alice_bases[i])) for i in range(n)]

    # Optional noise model
    noise_model = build_depolarizing_noise_model(noise_prob)
    # Build a simulator with the noise model for this run
    simulator = AerSimulator(noise_model=noise_model)

    # Bob measures each qubit
    bob_results = []
    for i in range(n):
        measured_bit = measure_in_basis_and_run(prepared_circuits[i], int(bob_bases[i]), simulator, noise_model=noise_model)
        bob_results.append(measured_bit)

    bob_results = np.array(bob_results)

    # Sifting: keep indices where bases match
    sifted_positions = [i for i in range(n) if alice_bases[i] == bob_bases[i]]
    sifted_bits_alice = [int(alice_bits[i]) for i in sifted_positions]
    sifted_bits_bob = [int(bob_results[i]) for i in sifted_positions]

    sifted_len = len(sifted_positions)

    if sifted_len == 0:
        return {
            "qber": None,
            "key_rate": 0.0,
            "sifted_key_len": 0,
            "final_key": [],
            "sifted_positions": sifted_positions,
            "detected_eavesdropper": True
        }

    # Error rate estimation: sample a random subset of sifted indices
    sample_size = max(1, int(sample_fraction * sifted_len))
    sample_indices_local = random.sample(range(sifted_len), sample_size)

    # Compute mismatches in the sample (this simulates publicly comparing these bits)
    mismatches = sum(1 for idx in sample_indices_local if sifted_bits_alice[idx] != sifted_bits_bob[idx])
    qber = mismatches / sample_size

    # Remove sampled bits from the key (publicly revealed)
    final_key = [sifted_bits_bob[i] for i in range(sifted_len) if i not in sample_indices_local]
    final_key_len = len(final_key)

    # Decide if eavesdropper detected (QBER above threshold)
    detected_eavesdropper = qber > QBER_THRESHOLD

    # Key rate: fraction of original N bits that remain as secret key
    key_rate = final_key_len / n

    return {
        "qber": qber,
        "key_rate": key_rate,
        "sifted_key_len": sifted_len,
        "final_key": final_key,
        "sifted_positions": sifted_positions,
        "detected_eavesdropper": detected_eavesdropper
    }


# ---------------------------
# Experiment runner and plotting
# ---------------------------
def run_experiments_across_noise_levels(noise_levels: List[float], n: int, sample_fraction: float):
    backend = None  # backend constructed per noise level inside run_bb84_once

    # Ensure rolling progress CSV exists in main directory
    progress_csv_path = os.path.join(SCRIPT_DIR, 'progress.csv')
    if not os.path.exists(progress_csv_path):
        with open(progress_csv_path, 'w', encoding='utf-8') as f:
            f.write('NoiseLevel,QBER,KeyRate,DetectedEavesdropper\n')

    qber_values = []
    key_rates = []
    detection_flags = []

    print(f"Running BB84 experiments: N={n}, sample_fraction={sample_fraction}, trials per noise level=1")
    for p in noise_levels:
        print(f"\n-- Noise level (depolarizing prob) = {p:.3f} --")
        stats = run_bb84_once(n, sample_fraction, p, backend)
        if stats['qber'] is None:
            qber_values.append(None)
            key_rates.append(0.0)
            detection_flags.append(True)
            print("No sifted bits (very unlikely).")
            continue

        qber_values.append(stats['qber'])
        key_rates.append(stats['key_rate'])
        detection_flags.append(stats['detected_eavesdropper'])

        print(f" Sifted key length: {stats['sifted_key_len']}")
        print(f" Sample QBER (estimated): {stats['qber']*100:.2f}%")
        print(f" Final key length: {len(stats['final_key'])}")
        print(f" Key rate (final_key / N): {stats['key_rate']:.3f}")
        print(f" Eavesdropper detected? {'YES' if stats['detected_eavesdropper'] else 'NO'}")

        # Append per-noise-level autosave row to progress CSV
        qber_out = '' if stats['qber'] is None else f"{stats['qber']:.6f}"
        with open(progress_csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{p:.6f},{qber_out},{stats['key_rate']:.6f},{int(stats['detected_eavesdropper'])}\n")

    # Plot QBER and Key Rate vs Noise
    fig, ax1 = plt.subplots(figsize=(8,5))

    xs = noise_levels
    ys_qber = [q*100 if q is not None else 0 for q in qber_values]  # percent
    ys_key_rate = key_rates

    ax1.set_xlabel('Depolarizing noise probability')
    ax1.set_ylabel('QBER (%)', fontsize=12)
    ax1.plot(xs, ys_qber, marker='o', label='QBER (%)')
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Key rate (final_key / N)', fontsize=12)
    ax2.plot(xs, ys_key_rate, marker='s', linestyle='--', label='Key rate')
    ax2.set_ylim(0, 1.0)

    plt.title('BB84: QBER (%) and Key Rate vs Depolarizing Noise')
    ax1.grid(True)
    fig.tight_layout()
    plt.show()

    return {
        "noise_levels": noise_levels,
        "qber_values": qber_values,
        "key_rates": key_rates,
        "detection_flags": detection_flags
    }


# ---------------------------
# Simple "comparison" printout
# ---------------------------
def print_comparison_table(results):
    print("\nComparison (BB84 simulation results vs idealized classical PQC placeholder):")
    print("{:<25} {:<20} {:<20}".format("Approach", "QBER (est)", "Key rate (final/N)"))
    print("-"*65)
    for p, q, kr, det in zip(results['noise_levels'], results['qber_values'], results['key_rates'], results['detection_flags']):
        q_str = f"{q*100:.2f}%" if q is not None else "N/A"
        det_str = " (Eve Detected)" if det else ""
        print(f"BB84 (p={p:.3f}){'':5} {q_str:<20} {kr:.3f}{det_str}")
    # idealized PQC line (placeholder)
    print(f"{'Lattice-based PQC (idealized)':<25} {'0.00%':<20} {1.000}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    results = run_experiments_across_noise_levels(NOISE_LEVELS, N, SAMPLE_FRACTION)
    print_comparison_table(results)
    # Save outputs next to the script regardless of current working directory
    png_path = os.path.join(SCRIPT_DIR, 'qber_vs_noise.png')
    csv_path = os.path.join(SCRIPT_DIR, 'keyrate_vs_noise.csv')
    plt.savefig(png_path)
    plt.close()
    np.savetxt(csv_path,
               np.column_stack((results['noise_levels'], results['qber_values'], results['key_rates'])),
               delimiter=',',
               header='NoiseLevel,QBER,KeyRate',
               fmt='%.4f')
    print("\nDone. You can tweak N, SAMPLE_FRACTION, NOISE_LEVELS, and QBER_THRESHOLD at top of script.")