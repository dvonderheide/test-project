#!/usr/bin/env python3
import itertools as itt
import numpy as np
import sys

class Paramgroup(object):
    def __init__(self, pg={}, rep={}):
        self._groups = []
        self._repeats = {}
        self._product_groups = {}  # pg[0]['max'] = [50, 100, 150]
        for gr, items in pg.items():
            for name, params in items.items():
                self.add(gr, name, params)
        for gr, reps in rep.items():
            self.set_repeats(gr, reps)

        self._defaults = {}

        self._seedname = "general:seed"
        self._run_name = "run"
        self._groupname = "group"
        self._setname = "set"

        self._run_name = "general:run"
        self._groupname = "general:group"
        self._setname = "general:set"
        self.default('general:output_frequency', 1000000)

    def _finalize(self):
        for group in self.groups:
            for name, params in self._defaults.items():
                if name not in self._product_groups[group]:
                    self.add(group, name, params)

    def __iter__(self):
        self._finalize()
        header = self.header[3:]  # TODO: header
        count = 0
        for group in self.groups:
            for _ in range(self._repeats[group]):
                vals = [self._product_groups[group][key] for key in header]
                for ps_i, param_set in enumerate(itt.product(*vals), 1):
                    count += 1
                    yield (count, group, ps_i) + param_set

    @property
    def groups(self):
        return self._groups.copy()

    @property  # TODO: make this cleaner and not suck
    def header(self):
        header = self._get_header_from_group(self.groups[0])
        for group in self.groups[1:]:
            this = self._get_header_from_group(group)
            if this != header:
                msg = "Headers for group ({}) do not match:\n{}\n{}"
                msg = msg.format(group, header, this)
                raise RuntimeError(msg)
        cols_prefix = [self._run_name, self._groupname, self._setname]
        return cols_prefix + header

    def print(self, outfile=None, logfile=None, delimiter=","):
        if len(self.groups) == 0:
            raise RuntimeError("No parameters added.")

        self._finalize()
        header = self.header
        with open(outfile, "w") if outfile else sys.stdout as of:
            print("#", self, file=of)
            print(*header, self._seedname, sep=delimiter, file=of)

            porder = [header.index(name) for name in header]
            for param_set in self:
                line = [param_set[i] for i in porder]
                print(*line, np.random.randint(1e8), sep=delimiter, file=of)

        if logfile and outfile:
            with open(logfile, "a") as lf:
                from datetime import date

                print(date.today().strftime("%Y.%m.%d"), outfile, self, file=lf)

    def _get_header_from_group(self, group):
        header = list(sorted(self._product_groups[group].keys()))
        if len(header) != len(set(header)):
            msg = "One or more param names repeated: {}".format(header)
            raise RuntimeError(msg)
        return header

    @staticmethod
    def _tolist(params):
        try:
            params = params.tolist()  # numpy
        except AttributeError:
            if not isinstance(params, str) and not hasattr(params, "__iter__"):
                params = [params]
        return params

    def default(self, name, param):
        param = self._tolist(param)
        self._defaults[name] = param

    def add(self, group, name, params, override=False):
        params = self._tolist(params)

        if group not in self.groups:
            self._groups.append(group)
            self._repeats[group] = 1
            self._product_groups[group] = {}

        if not override and name in self._product_groups[group]:
            print(f"Warning: Overwriting {name} in {group}")
        self._product_groups[group][name] = params

    def set_repeats(self, n, group=None):
        if group:
            if group not in self.groups:
                raise RuntimeError("{} not in groups: {}".format(group, self.groups))
            self._repeats[group] = n
        else:
            for gr in self.groups:
                self._repeats[gr] = n

    def __repr__(self):
        return "Paramgroup(pg={}, rep={})".format(self._product_groups, self._repeats)

    def __len__(self):
        self._finalize()
        gc = {
            gr: np.prod([len(x) for x in self._product_groups[gr].values()])
            for gr in self.groups
        }
        return sum(self._repeats[gr] * gc[gr] for gr in self.groups)


def _parse_line(line):
    import re
    from ast import literal_eval

    pg = literal_eval(re.sub(".*pg=(\{.*\}), rep=.*", r"\1", line))
    rep = literal_eval(re.sub(".*pg=.*, rep=(\{.*\})\)$", r"\1", line))
    return {"pg": pg, "rep": rep}
