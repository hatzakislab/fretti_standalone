from typing import List

import hmmlearn.hmm
import numpy as np
import pandas as pd
import sklearn
import sklearn.mixture
import scipy.optimize


def find_transitions(states, fret):
    """
    Finds transitions and their lifetimes, given states and FRET signal
    """
    hf = pd.DataFrame()
    hf["state"] = states
    hf["y_obs"] = fret
    hf["y_fit"] = hf.groupby(["state"], as_index=False)["y_obs"].transform(
        "median"
    )

    hf["time"] = hf["y_fit"].index + 1

    # Calculate lifetimes now, by making a copy to work on
    lf = hf.copy()

    # # Find y_after from y_before
    lf["state+1"] = np.roll(lf["state"], -1)

    # Find out when there's a change in state, depending on the minimum
    # transition size set
    lf["state_jump"] = lf["state"].transform(
        lambda group: (abs(group.diff()) > 0).cumsum()
    )

    # Drop duplicates
    lf.drop_duplicates(subset="state_jump", keep="last", inplace=True)

    # Find the difference for every time
    lf["lifetime"] = np.append(np.nan, np.diff(lf["time"]))

    lf = lf[["state", "state+1", "lifetime"]]
    lf = lf[:-1]
    lf.dropna(inplace=True)

    idealized = hf["y_fit"].values
    idealized_idx = hf["time"].values
    transitions = lf
    return idealized, idealized_idx, transitions


def seq_probabilities(yi, skip_threshold=0.5, skip_column=0):
    """
    Calculates class-wise probabilities over the entire trace for a one-hot
    encoded sequence prediction. Skips values where the first value is above
    threshold (bleaching).
    """
    assert len(yi.shape) == 2

    # Discard frames where bleaching (column 0) is above threshold (0.5)
    p = yi[yi[:, skip_column] < skip_threshold]
    if len(p) > 0:
        # Sum frame values for each class
        p = p.sum(axis=0) / len(p)

        # Normalize to 1
        p = p / p.sum()

        # don't ignore bleached frames entirely,
        # as it's easier to deal with a tiny number of edge cases
        # p[skip_column] = 0
    else:
        p = np.zeros(yi.shape[1])

    # sum static and dynamic smFRET scores (they shouldn't compete)
    confidence = p[4:].sum()
    return p, confidence


def calc_E(intensities, alpha=0, delta=0, clip_range=(-0.3, 1.3)):
    """
    Calculates raw FRET efficiency from donor (Dexc-Dem) and acceptor
    (Dexc-Aem). Note that iSMS has the option of subtracting background or not,
    and calculate E (apparent E) from that.
    """

    cmin, cmax = clip_range

    F_DA, I_DD, I_DA, I_AA = correct_DA(intensities, alpha, delta)

    E = F_DA / (I_DD + F_DA)
    E = np.clip(E, cmin, cmax, out=E)
    E = np.reshape(E, -1)

    return E


def calc_S(
        intensities, alpha=0, delta=0, beta=1, gamma=1, clip_range=(-0.3, 1.3)
):
    """
    Calculates raw calc_S from donor (Dexc-Dem), acceptor (Dexc-Aem) and direct
    emission of acceptor ("ALEX", Aexc-Aem) Note that iSMS has the option of
    subtracting background or not, and calculate S (apparent S) from that.
    """
    cmin, cmax = clip_range

    F_DA, I_DD, I_DA, I_AA = correct_DA(intensities, alpha, delta)

    inv_beta = 1 / beta

    S = (gamma * I_DD + F_DA) / (gamma * I_DD + F_DA + (inv_beta * I_AA))
    S = np.clip(S, cmin, cmax, out=S)
    S = np.reshape(S, -1)

    return S


def correct_DA(intensities, alpha=0, delta=0):
    """
    Calculates corrected Dexc-Aem intensity, for use in E and S calculations.
    """
    grn_int, grn_bg, acc_int, acc_bg, red_int, red_bg = intensities

    I_DD = grn_int - grn_bg
    I_DA = acc_int - acc_bg
    I_AA = red_int - red_bg

    if contains_nan(I_AA):
        F_DA = I_DA - (alpha * I_DD)
    else:
        F_DA = I_DA - (alpha * I_DD) - (delta * I_AA)

    return F_DA, I_DD, I_DA, I_AA


def contains_nan(array):
    """
    Returns True if array contains nan values
    """
    return np.isnan(np.sum(array))


def fit_gaussian_mixture(
        X: np.ndarray, min_n_components: int = 1, max_n_components: int = 1
):
    """
    Fits the best univariate gaussian mixture model, based on BIC
    If min_n_components == max_n_components, will lock to selected number, but
    still test all types of covariances
    """
    X = X.reshape(-1, 1)

    models, bic = [], []
    n_components_range = range(min_n_components, max_n_components + 1)
    cv_types = ["spherical", "tied", "diag", "full"]

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = sklearn.mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            models.append(gmm)
            bic.append(gmm.bic(X))

    best_gmm = models[int(np.argmin(bic))]
    print("number of components ", best_gmm.n_components)

    weights = best_gmm.weights_.ravel()
    means = best_gmm.means_.ravel()
    sigs = np.sqrt(best_gmm.covariances_.ravel())

    # Due to covariance type
    if len(sigs) != len(means):
        sigs = np.repeat(sigs, len(means))

    print("weights: ", weights)
    print("means: ", means)
    print("sigs: ", sigs)

    params = [(m, s, w) for m, s, w in zip(means, sigs, weights)]
    params = sorted(params, key=lambda tup: tup[0])

    return best_gmm, params


def fit_hmm(
        X: np.ndarray,
        fret: np.ndarray,
        lengths: List[int],
        covar_type: str,
        n_components: int,
):
    """
    Fits a Hidden Markov Model to traces. The traces are row-stacked, to provide
    a (t, c) matrix, where t is the total number of frames, and c is the
    channels
    """
    hmm_model = hmmlearn.hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covar_type,
        init_params="stmc",  # auto init all params
        algorithm="viterbi",
    )
    hmm_model.fit(X, lengths)

    states = hmm_model.predict(X, lengths)
    transmat = hmm_model.transmat_

    state_means, state_sigs = [], []
    for si in sorted(np.unique(states)):
        _, params = fit_gaussian_mixture(fret[states == si])
        for (m, s, _) in params:
            state_means.append(m)
            state_sigs.append(s)

    return states, transmat, state_means, state_sigs


def _func_double_exp(_x: np.ndarray, _lambda_1: float, _lambda_2: float, _k: float):
    if _k > 1.0:
        raise ValueError(f"_k of value {_k:.2f} is larger than 1!")
    if _k < 0:
        raise ValueError(f"_k of value {_k:.2f} is smaller than 1!")
    _exp1 = _lambda_1 * np.exp(-1 * _lambda_1 * _x)
    _exp2 = _lambda_2 * np.exp(-1 * _lambda_2 * _x)

    return _k * _exp1 + (1 - _k) * _exp2


def _func_exp(_x: np.ndarray, _lambda, ):
    return _lambda * np.exp(- _lambda * _x)


def loglik_single(x: np.ndarray, _lambda):
    """
    Returns Negative Loglikelihood for a single exponential with given param and given observations
    """
    return -1 * np.sum(np.log(_func_exp(x, _lambda)))


def loglik_double(x: np.ndarray, _lambda_1: float, _lambda_2: float, _k: float):
    """
    Returns Negative Loglikelihood for a double exponential with given params and given observations
    """
    return -1 * np.sum(np.log(_func_double_exp(x, _lambda_1, _lambda_2, _k)))


def fit_and_compare_exp_funcs(arr, x0=None, verbose=0, meth="l-bfgs-b"):
    # def lh_single(l):
    #     loglik_single(arr, l)
    if x0 is None:
        x0 = (1 / arr.mean(), 1 / (arr.mean() + 1), 0.55)

    lh_single = lambda l: loglik_single(arr, l)

    def lh_double(x):
        return loglik_double(arr, *x)

    res1 = scipy.optimize.minimize(lh_single, x0=np.array(1. / arr.mean()), method=meth,
                                   options={"disp": False},
                                   bounds=[(0., None)])
    llh_1 = - res1.fun
    bic_1 = 2 * np.log(len(arr)) * 1 - 2 * llh_1

    if verbose:
        print("Params for single exp:")
        print(res1.x)
        print(f"BIC : {bic_1:.6f}")

    res2 = scipy.optimize.minimize(lh_double, x0=np.array(x0), method=meth,
                                   options={"disp": False},
                                   bounds=[(0., None), (0., None), (0.1, .9)])

    llh_2 = - res2.fun
    bic_2 = np.log(len(arr)) * 3 * 2 - 2 * llh_2
    if verbose:
        print("Params for double exp:")
        print(res2.x)
        print(f"BIC : {bic_2:.6f}")

    out = {}
    if bic_1 < bic_2:
        out["BEST"] = "SINGLE"
    else:
        out["BEST"] = "DOUBLE"

    out["SINGLE_LLH"] = llh_1
    out["SINGLE_BIC"] = bic_1
    out["SINGLE_PARAM"] = res1.x

    out["DOUBLE_LLH"] = llh_2
    out["DOUBLE_BIC"] = bic_2
    out["DOUBLE_PARAM"] = res2.x

    return out
