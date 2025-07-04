import math
import numpy as np
from scipy.stats import norm
from multiprocess import Pool
import warnings

from . import common_args
from ..util import read_param_file, ResultDict
from ..util.jit import optional_njit


def analyze(
    problem,
    Y,
    M=4,
    num_resamples=100,
    conf_level=0.95,
    print_to_console=False,
    seed=None,
    parallel: bool = False,
    n_processors: int = None,
):
    """Perform extended Fourier Amplitude Sensitivity Test on model outputs.

    Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
    size D (the number of parameters) containing the indices in the same order
    as the parameter file.

    Notes
    -----
    Compatible with:
        `fast_sampler` : :func:`SALib.sample.fast_sampler.sample`

    **Warning:** The FAST method, like other variance-based methods such as
    Sobol, assumes independent input parameters. If the parameters in your
    model are correlated, the standard FAST indices (S1, ST) can be
    misleading. The interpretation of these indices relies on the independence
    assumption. Using this method with correlated inputs may require
    specialized interpretation or alternative formulations not currently
    implemented here. Consider using methods more robust to correlations
    (like Delta Moment-Independent) or transforming inputs if correlations
    are present.

    Examples
    --------
        >>> X = fast_sampler.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = fast.analyze(problem, Y, print_to_console=False)

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    M : int
        The interference parameter, i.e., the number of harmonics to sum in
        the Fourier series decomposition (default 4)
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed to generate a random number
    parallel : bool, optional
        If True, parallelizes the bootstrap resampling computations. Default is False.
    n_processors : int, optional
        Number of processors to use when `parallel` is True. If None, defaults
        to all available CPU cores. Default is None.

    References
    ----------
    1. Cukier, R. I., C. M. Fortuin, K. E. Shuler, A. G. Petschek, and
       J. H. Schaibly (1973).
       Study of the sensitivity of coupled reaction systems to
       uncertainties in rate coefficients.
       J. Chem. Phys., 59(8):3873-3878
       doi:10.1063/1.1680571

    2. Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).
       A Quantitative Model-Independent Method for Global Sensitivity Analysis
       of Model Output.
       Technometrics, 41(1):39-56,
       doi:10.1080/00401706.1999.10485594.

    3. Pujol, G. (2006)
       fast99 - R `sensitivity` package
       https://github.com/cran/sensitivity/blob/master/R/fast99.R
    """
    if seed:
        np.random.seed(seed)

    if problem.get("corr_matrix") is not None:
        warnings.warn("Warning: The FAST method assumes independent input parameters. "
                      "The problem definition includes a 'corr_matrix', indicating correlated inputs. "
                      "Standard FAST indices (S1, ST) may be misleading under correlation. "
                      "Interpret results with caution. See documentation for more details.", UserWarning)

    D = problem["num_vars"]

    if Y.size % (D) == 0:
        N = int(Y.size / D)
    else:
        msg = """
        Error: Number of samples in model output file must be a multiple of D,
        where D is the number of parameters.
        """
        raise ValueError(msg)

    # Recreate the vector omega used in the sampling
    omega_0 = math.floor((N - 1) / (2 * M))

    # Calculate and Output the First and Total Order Values
    Si = ResultDict((k, [None] * D) for k in ["S1", "ST", "S1_conf", "ST_conf"])
    Si["names"] = problem["names"]
    for i in range(D):
        z = np.arange(i * N, (i + 1) * N)

        Y_l = Y[z]

        S1, ST = compute_orders(Y_l, N, M, omega_0)
        Si["S1"][i] = S1
        Si["ST"][i] = ST

        S1_d_conf, ST_d_conf = bootstrap(
            Y_l,
            M,
            num_resamples,
            conf_level,
            parallel=parallel,
            n_processors=n_processors,
        )
        Si["S1_conf"][i] = S1_d_conf
        Si["ST_conf"][i] = ST_d_conf

    if print_to_console:
        print(Si.to_df())

    return Si


@optional_njit(nopython=False, cache=True)
def compute_orders(outputs: np.ndarray, N: int, M: int, omega: int):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, math.ceil(N / 2))]) / N, 2)

    V = 2.0 * np.sum(Sp)

    # Calculate first and total order
    D1 = 2.0 * np.sum(Sp[np.arange(1, M + 1) * omega - 1])
    Dt = 2.0 * np.sum(Sp[np.arange(math.floor(omega / 2.0))])

    return (D1 / V), (1.0 - Dt / V)


<<<<<<< HEAD
# Worker function for FAST bootstrap
def _fast_bootstrap_worker(args):
    """
    Performs a single bootstrap resample for FAST S1 and ST.
    Designed for use with multiprocess.Pool.map.

    Parameters
    ----------
    args : tuple
        (Y_data_local, M_local, T_data_local, n_size_local, iter_index_or_seed)
        Y_data_local : full Y data for the current parameter
        M_local : interference parameter
        T_data_local : Y_data_local.shape[0]
        n_size_local : math.ceil(T_data_local * 0.5)

    Returns
    -------
    tuple
        (S1, ST) for the resample
    """
    Y_data_local, M_local, T_data_local, n_size_local, _ = args
    sample_idx = np.random.choice(T_data_local, replace=True, size=n_size_local)
    Y_rs = Y_data_local[sample_idx]

    N_rs = len(Y_rs)
    omega_rs = math.floor((N_rs - 1) / (2 * M_local)) # Changed N to N_rs, M to M_local

    return compute_orders(Y_rs, N_rs, M_local, omega_rs) # Changed M to M_local


@optional_njit(nopython=False, cache=True)
def bootstrap(
    Y: np.ndarray,
    M: int,
    resamples: int,
    conf_level: float,
    parallel: bool = False,
    n_processors: int = None,
):
=======
@optional_njit(nopython=False, cache=True)
def bootstrap(Y: np.ndarray, M: int, resamples: int, conf_level: float):
>>>>>>> origin/codex/optimize-functions-with-numba.njit
    """Compute CIs.

    Infers ``N`` from results of sub-sample ``Y`` and re-estimates omega (ω)
    for the above ``N``.
    """
    # Use half of available data each time
    T_data = Y.shape[0]
    n_size = math.ceil(T_data * 0.5)

    if resamples == 0:
        return np.nan, np.nan

    if parallel:
        tasks = [(Y, M, T_data, n_size, i) for i in range(resamples)]
        pool_processors = n_processors
        with Pool(processes=pool_processors) as pool:
            results = pool.map(_fast_bootstrap_worker, tasks)

        res_S1 = np.array([r[0] for r in results])
        res_ST = np.array([r[1] for r in results])
    else: # Serial execution
        res_S1 = np.zeros(resamples)
        res_ST = np.zeros(resamples)
        for i in range(resamples):
            sample_idx = np.random.choice(T_data, replace=True, size=n_size)
            Y_rs = Y[sample_idx]

            # Note: N in compute_orders is len(Y_rs), M is the original M
            N_bootstrap = len(Y_rs)
            omega_bootstrap = math.floor((N_bootstrap - 1) / (2 * M))

            S1, ST = compute_orders(Y_rs, N_bootstrap, M, omega_bootstrap)
            res_S1[i] = S1
            res_ST[i] = ST

    bnd = norm.ppf(0.5 + conf_level / 2.0)
    S1_conf = bnd * res_S1.std(ddof=1)
    ST_conf = bnd * res_ST.std(ddof=1)
    return S1_conf, ST_conf


# No additional arguments required for FAST
def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    -------
    Updated argparse object
    """
    parser.add_argument(
        "-M", "--M", type=int, required=False, default=4, help="Inference parameter"
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=100,
        help="Number of bootstrap resamples for Sobol " "confidence intervals",
    )

    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )

    analyze(
        problem,
        Y,
        M=args.M,
        num_resamples=args.resamples,
        print_to_console=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
