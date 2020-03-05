import os
from typing import Union, Tuple, List

import pandas as pd


def seek_line(
        line_starts: Union[str, Tuple[str, str]], path: str, timeout: int = 10
):
    """Seeks the file until specified line start is encountered in the start of
     the line."""
    with open(path, encoding="utf-8") as f:
        n = 0
        if isinstance(line_starts, str):
            line_starts = (line_starts,)
        s = [False] * len(line_starts)
        while not any(s):
            line = f.readline()
            s = [line.startswith(ls) for ls in line_starts]
            n += 1
            if n > timeout:
                return None
        return line


def csv_skip_to(path, line, timeout=10, **kwargs):
    """Seeks the file until specified header is encountered in the start of
    the line."""
    if os.stat(path).st_size == 0:
        raise ValueError("File is empty")
    with open(path, encoding="utf-8") as f:
        n = 0
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
            n += 1
            if n > timeout:
                return None
        f.seek(pos)
        return pd.read_csv(f, **kwargs)


def nice_string_output(names: List[str], values: List[str], extra_spacing: int = 0, ):
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(name, value,
                                                   spacing=extra_spacing + max_values + max_names - len(name))
    return string[:-2]
