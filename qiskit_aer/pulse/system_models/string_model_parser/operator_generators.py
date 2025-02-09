# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name, no-name-in-module, import-error

"""Operators to use in simulator"""

import numpy as np
from qiskit.quantum_info import Operator
from . import gen_operator


def sigmax(dim=2):
    """Qiskit wrapper of sigma-X operator."""
    if dim == 2:
        return gen_operator.sigmax()
    else:
        raise Exception("Invalid level specification of the qubit subspace")


def sigmay(dim=2):
    """Qiskit wrapper of sigma-Y operator."""
    if dim == 2:
        return gen_operator.sigmay()
    else:
        raise Exception("Invalid level specification of the qubit subspace")


def sigmaz(dim=2):
    """Qiskit wrapper of sigma-Z operator."""
    if dim == 2:
        return gen_operator.sigmaz()
    else:
        raise Exception("Invalid level specification of the qubit subspace")


def sigmap(dim=2):
    """Qiskit wrapper of sigma-plus operator."""
    return gen_operator.create(dim)


def sigmam(dim=2):
    """Qiskit wrapper of sigma-minus operator."""
    return gen_operator.destroy(dim)


def create(dim):
    """Qiskit wrapper of creation operator."""
    return gen_operator.create(dim)


def destroy(dim):
    """Qiskit wrapper of annihilation operator."""
    return gen_operator.destroy(dim)


def num(dim):
    """Qiskit wrapper of number operator."""
    return gen_operator.num(dim)


def qeye(dim):
    """Qiskit wrapper of identity operator."""
    return gen_operator.identity(dim)


def project(dim, states):
    """Qiskit wrapper of projection operator."""
    ket, bra = states
    if ket in range(dim) and bra in range(dim):
        return gen_operator.basis(dim, ket) * gen_operator.basis(dim, bra).adjoint()
    else:
        raise Exception(
            f"States are specified on the outside of Hilbert space {states}"
        )


def tensor(list_qobj):
    """Qiskit wrapper of tensor product"""
    return gen_operator.tensor(list_qobj)


def basis(level, pos):
    """Qiskit wrapper of basis"""
    return gen_operator.basis(level, pos)


def state(state_vec):
    """Qiskit wrapper of qobj"""
    return gen_operator.state(state_vec)


def fock_dm(level, eigv):
    """Qiskit wrapper of fock_dm"""
    return gen_operator.fock_dm(level, eigv)


def qubit_occ_oper_dressed(target_qubit, estates, h_osc, h_qub, level=0):
    """Builds the occupation number operator for a target qubit
    in a qubit oscillator system, where the oscillator are the first
    subsystems, and the qubit last. This does it for a dressed systems
    assuming estates has the same ordering

    Args:
        target_qubit (int): Qubit for which operator is built.
        estates (list): eigenstates in the dressed frame
        h_osc (dict): Dict of number of levels in each oscillator.
        h_qub (dict): Dict of number of levels in each qubit system.
        level (int): Level of qubit system to be measured.

    Returns:
        Qobj: Occupation number operator for target qubit.
    """
    # reverse sort by index
    rev_h_osc = sorted(h_osc.items(), key=lambda x: x[0])[::-1]
    rev_h_qub = sorted(h_qub.items(), key=lambda x: x[0])[::-1]

    proj_op = 0 * fock_dm(len(estates), 0)
    states = [basis(dd, 0) for ii, dd in rev_h_osc]
    for ii, dd in rev_h_qub:
        if ii == target_qubit:
            states.append(Operator(basis(dd, level).data.flatten()))
        else:
            states.append(Operator(np.ones(dd)))

    out_state = tensor(states)

    for ii, estate in enumerate(estates):
        if out_state.data.flatten()[ii] == 1:
            proj_op += estate & estate.adjoint()

    return proj_op


def get_oper(name, *args):
    """Return quantum operator of given name"""
    return __operdict.get(name, qeye)(*args)


__operdict = {
    "X": sigmax,
    "Y": sigmay,
    "Z": sigmaz,
    "Sp": create,
    "Sm": destroy,
    "I": qeye,
    "O": num,
    "P": project,
    "A": destroy,
    "C": create,
    "N": num,
}
