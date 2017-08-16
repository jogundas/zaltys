"""Helper functions pertaining to Time evolution needed to run zaltys.
"""

# TODO: Importing separate functions instead of whole modules is not advisable.
import numpy as np
from multiprocessing import Pool
from functools import partial
import scipy.sparse as sps


def energy_jumps(spectrum, max_state_number, jump_size):
    """Find energy manifolds by detecting jumps in the spectrum.

    Args:
        spectrum: list of eigenvalues.
        max_state_number: how many eigenvalues to consider.
        jump_size: energy difference which results in a new manifold.

    Returns:
        List of numbers where the jumps occur in the spectrum.
    """
    ee_jumps = []
    ee_jumps.append(0)
    for i1 in range(0, max_state_number):
        if abs(spectrum[i1] - spectrum[i1 + 1]) > jump_size:
            ee_jumps.append(i1 + 1)
    return ee_jumps


def single_point(x_operator, ysz_operator,
                 eigenvectors, spectrum, proj_coeff_list,
                 x0, t):
    """Return <X> and <YSZ> for a given state and time.

    Args:
        x_operator, ysz_operator: cf. zaltys_so.generate_x_operator.
        eigenvectors: eigenvectors of the rashba_hamiltonian.
        spectrum: list of eigenvalues of the rashba_hamiltonian.
        proj_coeff_list: a list of projection (to eigenstates) coefficients.
        x0: translation amplitude.
        t: time (w.r.t. to spatial translation) when the point is evaluated.

    Returns:
        [ <X>, <YSZ> ]
    """

    ev_state = np.sum(proj_coeff_list * np.exp(-1j * t * spectrum)
                      * eigenvectors.T, axis=1)

    x_avg = np.dot(ev_state.conj(), sps.csr_matrix.dot(x_operator, ev_state))

    ysz_avg = np.dot(ev_state.conj(),
                     sps.csr_matrix.dot(ysz_operator, ev_state))

    return [np.real(x_avg.item(0)).item(0) / x0,
            np.real(ysz_avg.item(0)).item(0) / x0]


def plot_state_evolution(translation_operator, x_operator, ysz_operator,
                         eigenvectors, spectrum,
                         x0, no_of_processes,
                         state_number, t_list):
    """Return <X> and <YSZ> for a given state and time points.

    Args:
        translation_operator: cf. zaltys_so.generate_translation_operator.
        x_operator, ysz_operator: cf. zaltys_so.generate_x_operator.
        eigenvectors: eigenvectors of the rashba_hamiltonian.
        spectrum: list of eigenvalues of the rashba_hamiltonian.
        x0: translation amplitude.
        no_of_processes: number of concurrent processes to use for
            computations.
        state_number: which eigenstate to translate.
        t_list: list of time points to evaluate.

    Returns:
        [ [ <X_0>, <YSZ_0> ],  [ <X_1>, <YSZ_1> ], ... ]
    """

    initial_state = np.array(sps.csr_matrix.dot(translation_operator,
                                                eigenvectors[state_number].T))

    proj_coeff_list = np.dot(eigenvectors.conj(), initial_state)

    pool = Pool(processes=no_of_processes)
    single_point_partial = partial(single_point, x_operator, ysz_operator,
                                   eigenvectors, spectrum, proj_coeff_list, x0)

    plotList = pool.map(single_point_partial, t_list, 1)
    pool.close()
    pool.join()

    return plotList


def zero_kelvin_fermions(translation_operator, x_operator, ysz_operator,
                         eigenvectors, spectrum,
                         x0, no_of_processes,
                         start_state, stop_state,
                         t_list):
    """Returns <X> and <YSZ> averaged over several states with equal weights
       (corresponding to T=0 fermions) for given time points.

    Args:
        translation_operator: cf. zaltys_so.generate_translation_operator.
        x_operator, ysz_operator: cf. zaltys_so.generate_x_operator.
        eigenvectors: eigenvectors of the rashba_hamiltonian.
        spectrum: list of eigenvalues of the rashba_hamiltonian.
        x0: translation amplitude.
        no_of_processes: number of concurrent processes to use for
            computations.
        start_state: the lowest-energy occupied eigenstate
        stop_state: the highest-energy occupied eigenstate is stop_state-1
        t_list: list of time points to evaluate.

    Returns:
        [ [ <X_0>, <YSZ_0> ],  [ <X_1>, <YSZ_1> ], ... ]
    """
    average_x = np.zeros(len(t_list))
    average_ysz = np.zeros(len(t_list))

    weight = 1. / (stop_state - start_state)

    for state_number in range(start_state, stop_state):
        plot_list = plot_state_evolution(
            translation_operator,
            x_operator,
            ysz_operator,
            eigenvectors,
            spectrum,
            x0,
            no_of_processes,
            state_number,
            t_list)
        plot_list = np.array(plot_list)
        average_x += weight * plot_list[:, 0]
        average_ysz += weight * plot_list[:, 1]

    return [average_x, average_ysz]
