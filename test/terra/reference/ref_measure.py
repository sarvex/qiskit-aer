# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Test circuits and reference outputs for measure instruction.
"""

from numpy import array
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction


# ==========================================================================
# Single-qubit measurements with deterministic output
# ==========================================================================


def measure_circuits_deterministic(allow_sampling=True):
    """Measure test circuits with deterministic count output."""

    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    # Measure |00> state
    circuit = QuantumCircuit(qr, cr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits = [circuit]
    # Measure |01> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Measure |10> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Measure |11> state
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    circuits.append(circuit)

    # Measure a single qubit (qubit 1) in |1> state
    qr = QuantumRegister(3)
    cr = ClassicalRegister(1)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 2)
    circuit.barrier(qr)
    circuit.measure(1, 0)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(1, 0)
    circuits.append(circuit)

    return circuits


def measure_counts_deterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        targets.extend(
            ({"0x0": shots}, {"0x1": shots}, {"0x2": shots}, {"0x3": shots})
        )
    else:
        targets.extend(({"00": shots}, {"01": shots}, {"10": shots}, {"11": shots}))
    # Measure a single qubit (qubit 1) in |1> state
    targets.append({"0x1": shots})
    return targets


def measure_memory_deterministic(shots, hex_counts=True):
    """Measure test circuits reference memory."""
    targets = []
    if hex_counts:
        targets.extend(
            (
                shots * ["0x0"],
                shots * ["0x1"],
                shots * ["0x2"],
                shots * ["0x3"],
            )
        )
    else:
        targets.extend(
            (shots * ["00"], shots * ["01"], shots * ["10"], shots * ["11"])
        )
    return targets


def measure_statevector_deterministic():
    """Measure test circuits reference counts."""

    targets = [array([1, 0, 0, 0])]
    # Measure |01> state
    targets.append(array([0, 1, 0, 0]))
    # Measure |10> state
    targets.append(array([0, 0, 1, 0]))
    # Measure |11> state
    targets.append(array([0, 0, 0, 1]))
    return targets


# ==========================================================================
# Single-qubit measurements with non-deterministic output
# ==========================================================================


def measure_circuits_nondeterministic(allow_sampling=True):
    """ "Measure test circuits with non-deterministic count output."""

    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)

    # Measure |++> state (sampled)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.measure(qr, cr)
    return [circuit]


def measure_counts_nondeterministic(shots, hex_counts=True):
    """Measure test circuits reference counts."""

    targets = []
    if hex_counts:
        # Measure |++> state
        targets.append({"0x0": shots / 4, "0x1": shots / 4, "0x2": shots / 4, "0x3": shots / 4})
    else:
        # Measure |++> state
        targets.append({"00": shots / 4, "01": shots / 4, "10": shots / 4, "11": shots / 4})
    return targets


# ==========================================================================
# Multi-qubit measurements with deterministic output
# ==========================================================================


def multiqubit_measure_circuits_deterministic(allow_sampling=True):
    """Multi-qubit measure test circuits with deterministic count output."""

    circuits = []

    def measure_n(num_qubits):
        """Multi-qubit measure instruction."""
        return Instruction("measure", num_qubits, num_qubits, [])

    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.barrier(qr)
    circuit.append(measure_n(2), [0, 1], [0, 1])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.append(measure_n(2), [0, 1], [0, 1])
    circuits.append(circuit)

    # 3-qubit measure |101>
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[0])
    circuit.x(qr[2])
    circuit.barrier(qr)
    circuit.append(measure_n(3), [0, 1, 2], [0, 1, 2])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.append(measure_n(3), [0, 1, 2], [0, 1, 2])
    circuits.append(circuit)

    # 4-qubit measure |1010>
    qr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr[1])
    circuit.x(qr[3])
    circuit.barrier(qr)
    circuit.append(measure_n(4), [0, 1, 2, 3], [0, 1, 2, 3])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.append(measure_n(4), [0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(circuit)

    return circuits


def multiqubit_measure_counts_deterministic(shots, hex_counts=True):
    """Multi-qubit measure test circuits reference counts."""

    targets = []
    if hex_counts:
        targets.extend(({"0x2": shots}, {"0x5": shots}, {"0xa": shots}))
    else:
        targets.extend(({"10": shots}, {"101": shots}, {"1010": shots}))
    return targets


def multiqubit_measure_memory_deterministic(shots, hex_counts=True):
    """Multi-qubit measure test circuits reference memory."""

    targets = []
    if hex_counts:
        targets.extend((shots * ["0x2"], shots * ["0x5"], shots * ["0xa"]))
    else:
        targets.extend((shots * ["10"], shots * ["101"], shots * ["1010"]))
    return targets


def multiqubit_measure_statevector_deterministic():
    """Multi-qubit measure test circuits reference counts."""

    targets = [array([0, 0, 1, 0])]
    # 3-qubit measure |101>
    targets.append(array([0, 0, 0, 0, 0, 1, 0, 0]))
    # 4-qubit measure |1010>
    targets.append(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    return targets


# ==========================================================================
# Multi-qubit measurements with non-deterministic output
# ==========================================================================


def multiqubit_measure_circuits_nondeterministic(allow_sampling=True):
    """Multi-qubit measure test circuits with non-deterministic count output."""
    circuits = []

    def measure_n(num_qubits):
        """Multi-qubit measure instruction."""
        return Instruction("measure", num_qubits, num_qubits, [])

    # 2-qubit measure |++>
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.append(measure_n(2), [0, 1], [0, 1])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.append(measure_n(2), [0, 1], [0, 1])
    circuits.append(circuit)

    # 3-qubit measure |++0>
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circuit = QuantumCircuit(qr, cr)
    circuit.h(qr[0])
    circuit.h(qr[1])
    circuit.barrier(qr)
    circuit.append(measure_n(3), [0, 1, 2], [0, 1, 2])
    if not allow_sampling:
        circuit.barrier(qr)
        circuit.i(qr)
        circuit.barrier(qr)
        circuit.append(measure_n(3), [0, 1, 2], [0, 1, 2])
    circuits.append(circuit)

    return circuits


def multiqubit_measure_counts_nondeterministic(shots, hex_counts=True):
    """Multi-qubit measure test circuits reference counts."""

    targets = []
    if hex_counts:
        targets.extend(
            (
                {
                    "0x0": shots / 4,
                    "0x1": shots / 4,
                    "0x2": shots / 4,
                    "0x3": shots / 4,
                },
                {
                    "0x0": shots / 4,
                    "0x1": shots / 4,
                    "0x2": shots / 4,
                    "0x3": shots / 4,
                },
            )
        )
    else:
        targets.extend(
            (
                {
                    "00": shots / 4,
                    "01": shots / 4,
                    "10": shots / 4,
                    "11": shots / 4,
                },
                {
                    "000": shots / 4,
                    "001": shots / 4,
                    "010": shots / 4,
                    "011": shots / 4,
                },
            )
        )
    return targets
