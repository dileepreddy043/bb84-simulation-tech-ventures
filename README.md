# bb84_simulation
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

