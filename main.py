import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import fretti.utils
import fretti.math
import warnings

warnings.simplefilter("ignore", RuntimeWarning)


class TraceChannel:
    """
    Class for storing trace information for individual channels
    """

    def __init__(self, color):
        self.color = color  # type: str
        self.int = None  # type: Union[None, np.ndarray]
        self.bg = None  # type: Union[None, np.ndarray]
        self.bleach = None  # type: Union[None, int]


class TraceContainer:
    """
    Class for storing individual newTrace information.
    """

    ml_column_names = [
        "p_bleached",
        "p_aggegate",
        "p_noisy",
        "p_scramble",
        "p_1-state",
        "p_2-state",
        "p_3-state",
        "p_4-state",
        "p_5-state",
    ]

    def __init__(self, filename, name=None, movie=None, n=None):
        self.filename = filename  # type: str
        self.name = (
            name if name is not None else os.path.basename(filename)
        )  # type: str
        self.movie = movie  # type: str
        self.n = n  # type: str

        self.tracename = None  # type: Union[None, str]
        self.savename = None  # type: Union[None, str]

        self.load_successful = False

        self.is_checked = False  # type: bool
        self.xdata = []  # type: [int, int]

        self.grn = TraceChannel(color="green")
        self.red = TraceChannel(color="red")
        self.acc = TraceChannel(color="red")

        self.first_bleach = None  # int
        self.zerobg = None  # type: (None, np.ndarray)

        self.fret = None  # type: Union[None, np.ndarray]
        self.stoi = None  # type: Union[None, np.ndarray]
        self.hmm = None  # type: Union[None, np.ndarray]
        self.hmm_idx = None  # type: Union[None, np.ndarray]
        self.transitions = None  # type: Union[None, pd.DataFrame]
        self.y_pred = None  # type: Union[None, np.ndarray]
        self.y_class = None  # type: Union[None, np.ndarray]
        self.confidence = None  # type: Union[None, float]

        self.a_factor = np.nan  # type: float
        self.d_factor = np.nan  # type: float
        self.frames = None  # type: Union[None, int]
        self.frames_max = None  # type: Union[None, int]
        self.framerate = None  # type: Union[None, float]

        self.channels = self.grn, self.red, self.acc
        # file loading
        try:
            self.load_from_ascii()
        except (TypeError, FileNotFoundError) as e:
            try:
                self.load_from_dat()
            except (TypeError, FileNotFoundError) as e:
                warnings.warn(
                    "Warning! No data loaded for this trace!", UserWarning
                )

    def load_from_ascii(self):
        """
        Reads a trace from an ASCII text file. Several checks are included to
        include flexible compatibility with different versions of trace exports.
        Also includes support for all iSMS traces.
        """
        colnames = [
            "D-Dexc-bg.",
            "A-Dexc-bg.",
            "A-Aexc-bg.",
            "D-Dexc-rw.",
            "A-Dexc-rw.",
            "A-Aexc-rw.",
            "S",
            "E",
        ]
        if self.filename.endswith(".dat"):
            raise TypeError("Datafile is not the right type for this function!")

        with open(self.filename) as f:
            txt_header = [next(f) for _ in range(5)]

        # This is for iSMS compatibility
        if txt_header[0].split("\n")[0] == "Exported by iSMS":
            df = pd.read_csv(self.filename, skiprows=5, sep="\t", header=None)
            if len(df.columns) == colnames:
                df.columns = colnames
            else:
                try:
                    df.columns = colnames
                except ValueError:
                    colnames = colnames[3:]
                    df.columns = colnames
        # Else Fretti trace compatibility
        else:
            df = fretti.utils.csv_skip_to(
                path=self.filename, line="D-Dexc", sep="\s+"
            )
        try:
            pair_n = fretti.utils.seek_line(
                path=self.filename, line_starts="FRET pair"
            )
            self.n = int(pair_n.split("#")[-1])

            movie = fretti.utils.seek_line(
                path=self.filename, line_starts="Movie filename"
            )
            self.movie = movie.split(": ")[-1]

        except (ValueError, AttributeError):
            pass

        self.load_successful = True

        # Add flag to see if incomplete trace
        if not any(s.startswith("A-A") for s in df.columns):
            df["A-Aexc-rw"] = np.nan
            df["A-Aexc-bg"] = np.nan
            df["A-Aexc-I"] = np.nan

        if "D-Dexc_F" in df.columns:
            warnings.warn(
                "This trace is created with an older format.",
                DeprecationWarning,
            )
            self.grn.int = df["D-Dexc_F"].values
            self.acc.int = df["A-Dexc_I"].values
            self.red.int = df["A-Aexc_I"].values

            zeros = np.zeros(len(self.grn.int))
            self.grn.bg = zeros
            self.acc.bg = zeros
            self.red.bg = zeros

        else:
            if "p_bleached" in df.columns:
                colnames += self.ml_column_names
                self.y_pred = df[self.ml_column_names].values
                self.y_class, self.confidence = fretti.math.seq_probabilities(
                    self.y_pred
                )

            # This strips periods if present
            df.columns = [c.strip(".") for c in df.columns]

            self.grn.int = df["D-Dexc-rw"].values
            self.acc.int = df["A-Dexc-rw"].values
            self.red.int = df["A-Aexc-rw"].values

            try:
                self.grn.bg = df["D-Dexc-bg"].values
                self.acc.bg = df["A-Dexc-bg"].values
                self.red.bg = df["A-Aexc-bg"].values
            except KeyError:
                zeros = np.zeros(len(self.grn.int))
                self.grn.bg = zeros
                self.acc.bg = zeros
                self.red.bg = zeros

        self.calculate_fret()
        self.calculate_stoi()

        self.frames = np.arange(1, len(self.grn.int) + 1, 1)
        self.frames_max = self.frames.max()

    def load_from_dat(self):
        """
        Loading from .dat files, as supplied in the kinSoft challenge
        """
        arr = np.loadtxt(self.filename)
        l = len(arr)
        zeros = np.zeros(len(arr))

        self.grn.int = arr[:, 1]
        self.acc.int = arr[:, 2]
        self.red.int = zeros * np.nan

        self.grn.bg = zeros
        self.acc.bg = zeros
        self.red.bg = zeros * np.nan

        self.framerate = int(1 / (arr[0, 1] - arr[0, 0]))

        self.calculate_fret()
        self.calculate_stoi()

        self.frames = np.arange(1, l + 1, 1)
        self.frames_max = self.frames.max()

        self.load_successful = True

    def get_intensities(self):
        """
        Convenience function to return trace get_intensities
        """
        grn_int = self.grn.int  # type: Union[None, np.ndarray]
        grn_bg = self.grn.bg  # type: Union[None, np.ndarray]
        acc_int = self.acc.int  # type: Union[None, np.ndarray]
        acc_bg = self.acc.bg  # type: Union[None, np.ndarray]
        red_int = self.red.int  # type: Union[None, np.ndarray]
        red_bg = self.red.bg  # type: Union[None, np.ndarray]

        return grn_int, grn_bg, acc_int, acc_bg, red_int, red_bg

    def get_bleaches(self):
        """
        Convenience function to return trace bleaching times
        """
        grn_bleach = self.grn.bleach  # type: Union[None, int]
        acc_bleach = self.acc.bleach  # type: Union[None, int]
        red_bleach = self.red.bleach  # type: Union[None, int]
        return grn_bleach, acc_bleach, red_bleach

    def get_export_df(self, keep_nan_columns: Union[bool, None] = None):
        """
        Returns the DataFrame to use for export
        """
        if keep_nan_columns is None:
            keep_nan_columns = True
        dfdict = {
            "D-Dexc-bg": self.grn.bg,
            "A-Dexc-bg": self.acc.bg,
            "A-Aexc-bg": self.red.bg,
            "D-Dexc-rw": self.grn.int,
            "A-Dexc-rw": self.acc.int,
            "A-Aexc-rw": self.red.int,
            "S": self.stoi,
            "E": self.fret,
        }

        if self.y_pred is not None:
            # Add predictions column names and values
            dfdict.update(dict(zip(self.ml_column_names, self.y_pred.T)))

        df = pd.DataFrame(dfdict).round(4)

        if keep_nan_columns is False:
            df.dropna(axis=1, how="all", inplace=True)

        return df

    def get_export_txt(
            self,
            df: Union[None, pd.DataFrame] = None,
            exp_txt: Union[None, str] = None,
            date_txt: Union[None, str] = None,
            keep_nan_columns: Union[bool, None] = None,
    ):
        """
        Returns the string to use for saving the trace as a txt
        """
        if df is None:
            df = self.get_export_df(keep_nan_columns=keep_nan_columns)
        if (exp_txt is None) or (date_txt is None):
            exp_txt = "Exported by Fretti"
            date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        mov_txt = "Movie filename: {}".format(self.movie)
        id_txt = "FRET pair #{}".format(self.n)
        bl_txt = "Donor bleaches at: {} - " "Acceptor bleaches at: {}".format(
            self.grn.bleach, self.red.bleach
        )
        return (
            "{0}\n"
            "{1}\n"
            "{2}\n"
            "{3}\n"
            "{4}\n\n"
            "{5}".format(
                exp_txt,
                date_txt,
                mov_txt,
                id_txt,
                bl_txt,
                df.to_csv(index=False, sep="\t", na_rep="NaN"),
            )
        )

    def get_tracename(self) -> str:
        if self.tracename is None:
            if self.movie is None:
                name = "Trace_pair{}.txt".format(self.n)
            else:
                name = "Trace_{}_pair{}.txt".format(
                    self.movie.replace(".", "_"), self.n
                )

            # Scrub mysterious \n if they appear due to filenames
            name = "".join(name.splitlines(keepends=False))
            self.tracename = name

        return self.tracename

    def get_savename(self, dir_to_join: Union[None, str] = None):
        if self.savename is None:
            if dir_to_join is not None:
                self.savename = os.path.join(dir_to_join, self.get_tracename())
            else:
                self.savename = self.get_tracename()
        return self.savename

    def export_trace_to_txt(
            self,
            dir_to_join: Union[None, str] = None,
            keep_nan_columns: Union[bool, None] = None,
    ):
        savename = self.get_savename(dir_to_join=dir_to_join)
        with open(savename, "w") as f:
            f.write(self.get_export_txt(keep_nan_columns=keep_nan_columns))

    def calculate_fret(self):
        self.fret = fretti.math.calc_E(self.get_intensities())

    def calculate_stoi(self):
        self.stoi = fretti.math.calc_S(self.get_intensities())


data_dir = Path("/Users/mag/Documents/study/phd/deepFRET_kinSoft/expSet1")
# TODO edit these to be argparse names
n_traces = None

traces = {}
trace_paths = []

for _trace_path in data_dir.iterdir():
    if _trace_path.suffix == '.dat':
        trace_paths.append(_trace_path)

if n_traces is not None:
    trace_paths = trace_paths[:n_traces]

for trace_path in trace_paths:
    _trace = TraceContainer(filename=str(trace_path))
    _trace_name = trace_path.parent.name + "_" + trace_path.name
    traces[_trace_name] = _trace

DD, DA, AA, E, lengths = [], [], [], [], []
for tracename, trace in traces.items():
    _, I_DD, I_DA, I_AA = fretti.math.correct_DA(trace.get_intensities())
    DD.append(I_DD[: trace.first_bleach])
    DA.append(I_DA[: trace.first_bleach])
    AA.append(I_AA[: trace.first_bleach])
    E.append(trace.fret[: trace.first_bleach])
    lengths.append(len(I_DD[: trace.first_bleach]))

DD = np.concatenate(DD)
DA = np.concatenate(DA)
AA = np.concatenate(AA)
E_trace = np.concatenate(E).reshape(-1, 1)

E_dist = np.concatenate([e[0:20] for e in E]).reshape(-1, 1)
print(E_dist.shape)
print(E_trace.shape)

if fretti.math.contains_nan(AA):
    X = np.column_stack((DD, DA))
else:
    X = np.column_stack((DD, DA, AA))

print("Fitting Gaussian Mixture model ... \n")
best_mixture_model, params = fretti.math.fit_gaussian_mixture(
    E_dist, min_n_components=1, max_n_components=6
)

print("Fitting HMM ... \n")
states, transmat, state_means, state_sigs = fretti.math.fit_hmm(
    X=X,
    fret=E_trace,
    lengths=lengths,
    n_components=best_mixture_model.n_components,
    covar_type=best_mixture_model.covariance_type,
)

print("States: ", np.unique(states))
print("Transition matrix:\n", np.round(transmat, 2))
print("State means:\n", state_means)
print("State sigmas:\n", state_sigs)
print("\n")

pos = 0

for l, trace in zip(lengths, traces.values()):
    si = states[pos: pos + l]
    pos += l

    idealized, time, transitions = fretti.math.find_transitions(
        states=si, fret=trace.fret
    )
    trace.hmm = idealized
    trace.hmm_idx = time
    trace.transitions = transitions

transitions = pd.concat([trace.transitions for trace in traces.values()])

for _, t in transitions.groupby(["state", "state+1"]):
    s_before = t["state"].values[0]
    s_after = t["state+1"].values[0]

    if transmat[s_before, s_after] == 0:
        continue

    print(
        "{} -> {}".format(t["state"].values[0], t["state+1"].values[0])
    )
    print("number of datapoints: ", len(t["lifetime"]))

    try:
        max_lifetime = np.max(t["lifetime"])

        data = t["lifetime"]

        lifetime_dict = fretti.math.fit_and_compare_exp_funcs(data)
        _b = lifetime_dict["BEST"]
        names = ["LLH", "BIC", "PARAM"]
        values = []
        for name in names:
            key = _b + "_" + name

            val = lifetime_dict[key]
            try:
                val_str = '{:.4f}'.format(val)
                values.append(val_str)
            except TypeError:
                for v in val:
                    v_str = '{:.4f}'.format(v)
                    values.append(v_str)

        if len(values) > len(names):
            names.remove("PARAM")
            names.extend("PARAM_{}".format(i) for i in range(1, 4))

        print("The best fit ({} exp) returned these params: ".format(lifetime_dict["BEST"]))
        print(fretti.utils.nice_string_output(names, values, 2))
        if _b == "DOUBLE":
            print("This is a degenerate state!")
        print('\n')

    except RuntimeError:
        print("Couldn't fit. Skipping")
        continue
    print()

# do magic hmm

# get transitions

# fit w exponential and extract BIC etc.

# print and save to file
