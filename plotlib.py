import pandas
from time import time
import numpy as np
import multiprocessing as mp
from re import sub, match
import warnings
import argparse
import os
import csv
from contextlib import contextmanager

import matplotlib
from matplotlib import cm
import matplotlib.colors
from matplotlib.colors import cnames, hex2color, rgb_to_hsv, hsv_to_rgb
import matplotlib.animation as animation
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.widgets import Slider, Button, RadioButtons

from tqdm import tqdm
from itertools import product
from functools import partial

# matplotlib.use("agg")
tqdm80 = partial(tqdm, ncols=80)
# matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

import plotnine
from plotnine import scales, geoms, labels, guides, guide_legend
from plotnine import ggplot, aes, facet_wrap, stats
from plotnine import element_text, theme
from plotnine.themes import theme_bw, theme_classic

# import vapory_git as pov
# from vapory_git import Camera, Scene, Background, LightSource, Texture, Pigment
# from vapory_git import Box, Sphere, Finish, Normal

import simbiofilm as sb

from tempfile import mkstemp


@contextmanager
def ignore_copywarn():
    pandas.options.mode.chained_assignment = None
    yield
    pandas.options.mode.chained_assignment = "warn"


def _getcorners(ix, spacing=0.02):
    ix1 = np.array(ix, dtype=float)
    ix2 = ix1 + 1 - spacing
    ix1 += spacing
    return ix1, ix2


def _getcolor(s, r, i, p, maxp, maxb):
    if p > 0:
        minv = 0.250
        # phage_factor = ((0.95 - minv) * np.log10(p)) / np.log10(maxp) + minv
        # phage_factor = ((0.95 - minv) * p / maxp) + minv
        # phage_factor = 1 - (((0.95 - minv) * p / maxp) + minv)
        phage_factor = 1 - (0.95 - minv) * np.log10(p) / np.log10(maxp) + minv
    else:
        phage_factor = 1
    total_biomass = s + r + i
    # biomass_factor = 1 - total_biomass / maxb

    # red = (s / total_biomass * (1 - phage_factor)) ** 2
    # blu = (r / total_biomass * (1 - phage_factor)) ** 2
    # grn = (i / total_biomass * (1 - phage_factor)) ** 2

    red = 0.8 * (s / total_biomass * phage_factor) ** 2
    blu = 0.8 * (r / total_biomass * phage_factor) ** 2
    grn = 0.8 * (i / total_biomass * phage_factor) ** 2

    return red, grn, blu, 1 - (total_biomass / maxb) ** 3
    # return red, grn, blu, total_biomass / maxb


def _make_base(shape):
    box = Box(
        [-1, -1, -1],
        [0, shape[1] + 1, shape[2] + 1],
        Texture(
            Pigment("color", [0.5, 0.5, 0.5]),
            Finish("ambient", 0.6),
            Normal("agate", 0.25, "scale", 0.5),
        ),
    )
    return box


def _render_3d_frame(
    file, outdir, shape, max_biomass=6.5e-12, max_phage=1000, format="png", prefix=""
):
    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    with np.load(file) as f:
        particle_names = [x for x in list(f.keys()) if x.endswith("data")]
        sus = np.zeros(shape)
        res = np.zeros(shape)
        inf = np.zeros(shape)
        phg = np.zeros(shape, dtype=int)
        biomass = {
            "Species1_data": sus,
            "Species2_data": res,
            "infected_data": inf,
            "phage_data": phg,
        }

        for name, dat in zip(particle_names, (f[x] for x in particle_names)):
            bio = biomass[name]
            for ind in dat:
                if "phage" in name:
                    bio[np.unravel_index(ind["location"], shape)] += 1
                else:
                    bio[np.unravel_index(ind["location"], shape)] += ind["mass"]

        objects = [_make_base(shape), Background("color", [0.9, 0.9, 0.9])]

        for ix in product(*map(range, shape)):
            s, r, i, p = sus[ix], res[ix], inf[ix], phg[ix]
            if s == 0 and r == 0 and i == 0:
                continue
            col = _getcolor(s, r, i, p, max_phage, max_biomass)
            ix1, ix2 = _getcorners(ix, 0.1)

            box = Box(
                ix,
                np.array(ix) + 1,
                Texture(Pigment("color", "rgbf", col)),
                Finish("ambient", 1.0, "diffuse", 1.0),
            )  # , 'no_shadow')
            objects.append(box)

            # box = Box(ix,
            #           np.array(ix) + 1, Texture(Pigment('color', col, 'filter', alpha)),
            #           Finish('ambient', 1.0, 'diffuse', 1.0))  # , 'no_shadow')
            # objects.append(box)

            # light = LightSource([shape[0], -shape[1] / 2, -shape[2] / 2], 'color', [1, 1, 1],
            #                     'spotlight', 'radius', shape[0], 'adaptive', 1, 'jitter',
            #                     'point_at', [0, shape[1] / 2, shape[2] / 2])
            # objects.append(light)

        camera = Camera(
            "location",
            # [shape[1], -shape[1] * 1 / 2, -shape[2] * 1 / 2],
            # [shape[1], -shape[1] * 1 / 2, shape[2] * 3 / 2],
            # [shape[1], shape[1] * 3 / 2, -shape[2] * 1 / 2],
            [shape[1], shape[1] * 3 / 2, shape[2] * 3 / 2],
            "sky",
            [1, 0, 0],
            "look_at",
            [0, shape[1] / 2, shape[2] / 2],
        )
        tmpfname = mkstemp(".pov", dir="tmpf")[1]
        Scene(camera, objects=objects).render(
            outname, tempfile=tmpfname, width=3000, height=3000
        )
        print(f"Rendered {outname}")


_colors = {
    k: rgb_to_hsv(hex2color(v))[0]
    for k, v in cnames.items()
    if np.all(rgb_to_hsv(hex2color(v))[1:] == 1)
}


class Frame:
    # min and max are the scale
    minb = 1e-12 / 3
    maxb = 6.31e-12
    maxp = 0
    maxp = np.log10(1000)
    # minv and maxv determine the saturation.
    pminv = 0.3
    pmaxv = 0.99
    bminv = 0.2
    bmaxv = 0.95

    def __init__(self, containers, types, colors, dl, name, time):
        """
        Color options:
           aqua        blue      chartreuse cyan        darkorange
           deepskyblue fuchsia   gold       lime        magenta
           orange      orangered red        springgreen yellow
       Or a number [0, 1]
        """
        colors = [_colors[c] if c in _colors else float(c) for c in colors]
        self.types = types
        ctc = (containers, types, colors)
        self.phage = [(c, r) for c, t, r in zip(*ctc) if t == "phage"]
        self.solute = [(c, r) for c, t, r in zip(*ctc) if t == "solute"]
        self.biomass = [(c, r) for c, t, r in zip(*ctc) if t == "biomass"]
        self.shape = containers[0].shape
        self.name = name
        self.dl = dl
        self.time = time

    def render(self, solute=None):
        # HSV -> RGB
        # Biomass color -> Hue
        # Biomass value out of max -> saturation
        # phage value out of max -> value
        # Solute ??????
        frame = np.zeros((3,) + self.shape)  # NxMx3
        hue, saturation, value = frame
        for phage, _ in self.phage:  # color ignored
            value += phage
        value[value > 0] = (
            1
            - self.pminv
            - (self.pmaxv - self.pminv) * np.log10(value[value > 0]) / self.maxp
        )
        color = np.zeros_like(hue)
        biomass = np.zeros_like(hue)
        for i, (bio, col) in enumerate(self.biomass):
            hasbio = bio > 0
            mixed = (biomass > 0) & (bio > 0) & (np.abs(color - col) <= 0.5)
            single = (biomass == 0) & (bio > 0)
            circle = (biomass > 0) & (bio > 0) & (np.abs(color - col) > 0.5)
            linear = mixed | single
            biomass += bio
            color[circle] = (
                color[circle] - (1 - col) * bio[circle] / biomass[circle] + 1
            )
            color[linear] = color[linear] + col * bio[linear] / biomass[linear]
        hue[:] = color
        value[value == 0] = 1  # Default to white
        # hue[saturation > 0] /= saturation[saturation > 0]
        saturation[biomass > 0] = self.bminv + (self.bmaxv - self.bminv) * (
            biomass[biomass > 0] - self.minb
        ) / (self.maxb - self.minb)
        frame = np.moveaxis(frame, 0, -1)
        # TODO: getting negative values sometimes -- species mixing looks bad.
        return hsv_to_rgb(frame)


def biomass_colors(n, cb_safe=True):
    # if n <= 3:
    #     yield from ["#1b9e77", "#d95f02", "#7570b3"]
    # elif n == 4:
    yield "blue"
    yield "red"
    yield "orange"
    yield "magenta"
    yield "deepskyblue"
    # yield "magenta"
    # yield "springgreen"


def to_grid(shape, locations, values):
    ret = np.zeros(shape)
    np.add.at(ret.reshape(-1), locations, values)
    return ret


def _plot_frame_matplotlib(
    file,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    style=None,
    solute_adjustment=None,
):

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # matplotlib deprecation warnings

    # outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    with np.load(file) as f:
        config = sb.parse_params(*f["config"].T)
        shape = [int(x) for x in config.space.shape]
        dl = config.space.dl * 1e6
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]
        if solute_adjustment:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)

        # TODO: generic this.
        # TODO: get names and colors from config
        containers = []
        types = []
        hues = []
        col = biomass_colors(len(biomass_names))
        for k in f.keys():
            if k.lower().endswith("_value"):
                containers.append(f[k])
                types.append("solute")
                hues.append("magenta")
            if k.lower().endswith("_data"):
                if "phage" in k.lower():
                    types.append("phage")
                    containers.append(
                        to_grid(shape, f[k]["location"], np.ones_like(f[k]["location"]))
                    )
                    hues.append(0)
                if "species" in k.lower() or "infected" in k.lower():
                    containers.append(to_grid(shape, f[k]["location"], f[k]["mass"]))
                    types.append("biomass")
                    hues.append(next(col))

        fig, ax = plt.subplots(figsize=np.array(shape) / 2)
        frame = Frame(containers, types, hues)
        pltspace = (0, shape[1] * dl, 0, shape[0] * dl)
        ax.imshow(frame.render(), extent=pltspace, origin="lower", resample=True)
        fig.savefig("test.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # return fig, ax


def _make_frame(fname, crop=None, solute_adjustment=None):
    with np.load(fname, allow_pickle=True) as f:
        config = sb.parse_params(*f["config"].T)
        shape = [int(x) for x in config.space.shape]
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]
        if solute_adjustment:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)

        # TODO: for legend, use Patch: https://bit.ly/2Mi64o5
        # TODO: Related, name containers by config. Also colors!

        # TODO: generic this.
        # TODO: get names and colors from config
        containers = []
        types = []
        hues = []
        col = biomass_colors(len(biomass_names))
        for k in f.keys():
            if k.lower().endswith("_value"):
                containers.append(f[k])
                types.append("solute")
                hues.append("magenta")
            if k.lower().endswith("_data"):
                if "phage" in k.lower():
                    types.append("phage")
                    containers.append(
                        to_grid(shape, f[k]["location"], np.ones_like(f[k]["location"]))
                    )
                    hues.append(0)
                # elif "matrix" in k.lower():
                else:
                    containers.append(to_grid(shape, f[k]["location"], f[k]["mass"]))
                    types.append("biomass")
                    hues.append(next(col))

        name = fname.split("/")[-1][:-4]  # removes npz
        return Frame(
            containers, types, hues, config.space.dl * 1e6, name, float(f["t"])
        )


@contextmanager
def _timer(msg):
    start = time()
    yield
    end = time()
    print(msg, end - start)


def make_video(
    datadir,
    format,
    crop=None,
    maxb=6.5e-12,
    maxp=1000,
    solute_adjustment=None,
    interval=20,
    name=None,
    outdir=".",
    prefix="",
    progress_bar=False,
    skip_made=False,
):
    flist = sorted(
        [
            "{}/{}".format(datadir, x)
            for x in os.listdir(datadir)
            if x.lower().endswith(".npz")
        ]
    )
    # flist = flist[360:361]
    name = name if name is not None else datadir.split("/")[-1]

    print("Making frames")
    frames = [_make_frame(f, crop, solute_adjustment) for f in flist]
    fs = frames[0]
    shape = np.array(fs.shape)[::-1]
    fig, ax = plt.subplots(figsize=shape / 25)  # TODO size args
    pltspace = (0, shape[0] * fs.dl, 0, shape[1] * fs.dl)
    images = []
    print("Plotting frames")
    if format in fig.canvas.get_supported_filetypes():
        os.makedirs(f"{outdir}/{name}_frames", exist_ok=True)
    else:
        os.makedirs(outdir, exist_ok=True)

    progress = tqdm80 if progress_bar else (lambda x: x)
    if format in fig.canvas.get_supported_filetypes():
        odir = f"{outdir}/{name}_frames"
        im = ax.imshow(np.zeros((50, 300, 3)), extent=pltspace, origin="lower")
        fig.savefig("tmp.png")
        for f in progress(frames):
            ofname = f"{odir}/{f.name}.{format}"
            if skip_made and os.path.isfile(ofname):
                continue
            print(f.name)
            im.set_data(f.render())
            timestr = "{:.2f} days".format(f.time).rjust(10, " ")
            ax.set_title("t = " + timestr)
            ax.draw_artist(ax.patch)
            ax.draw_artist(im)
            plt.savefig(ofname, dpi=150, bbox_inches="tight")

    else:
        for f in frames:
            im = ax.imshow(f.render(), extent=pltspace, origin="lower", animated=True)
            images.append([im])

        ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True)
        print("Making video")
        ani.save(f"{name}.{format}")

    plt.close()


def _plot_frame(
    file,
    outdir,
    shape,
    crop=None,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    style=None,
    solute_adjustment=None,
):

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # matplotlib deprecation warnings

    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    with np.load(file, allow_pickle=True) as f:
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]

        gridcols = np.unravel_index(range(np.prod(shape)), shape)

        # set up biomass
        biomass = [np.rec.array(f[x]) for x in biomass_names]
        biomass_totals = [np.zeros(np.prod(shape)) for x in biomass_names]
        [np.add.at(tot, c.location, c.mass) for tot, c in zip(biomass_totals, biomass)]

        def _make_df(data):
            return pandas.DataFrame({"count": data, "x": gridcols[1], "y": gridcols[0]})

        dfs = [_make_df(d) for d in biomass_totals]
        for df, species in zip(dfs, biomass_names):
            df["species"] = sub("_data", "", species)
        bio = pandas.concat(dfs)
        minv = 0.20
        minb = bio[bio["count"] > 0]["count"].min()
        bio["alpha"] = ((0.95 - minv) * (bio["count"] - minb)) / (
            max_biomass - minb
        ) + minv

        # phage
        # phage = np.rec.array(f[phage_names[0]])
        phage_total = np.zeros(np.prod(shape))
        # np.add.at(phage_total, phage.location, 1)
        phg = _make_df(phage_total)

        minv = 0.250
        if max_phage > 0:
            max_phage = np.log10(max_phage)
        minp = np.log10(phg[phg["count"] > 0]["count"].min())
        if max_phage <= minp:
            max_phage += min(0.01, max_phage * 1.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore log10(0) warnings
            phg["alpha"] = ((0.95 - minv) * (np.log10(phg["count"]) - minp)) / (
                max_phage - minp
            ) + minv

        phg["species"] = "phage"

        # combine, and clear out empty cells. Otherwise, alpha is minv

        # NORMAL:
        colors = ["red", "blue", "green", "black"]
        lims = ["Species1", "Species2", "infected", "phage"]

        if style == "single":
            colors = ["red", "orange", "green", "black"]
            lims = ["Species1", "Matrix1", "infected", "phage"]

        if style == "two":
            colors = ["red", "blue", "orange", "#aa7711", "green", "black"]
            lims = ["Species1", "Species2", "Matrix1", "Matrix2", "infected", "phage"]

        colors = ["red", "blue", "orange"]
        lims = ["Species1", "Species2", "Matrix1"]

        if solute_adjustment is not None:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)
            mins = sv[sv.nonzero()].min()
            maxs = sv[sv.nonzero()].max()
            sv = ((0.95) * (sv - mins)) / (maxs - mins)
            solute = pandas.DataFrame(
                {
                    "y": gridcols[0],
                    "x": gridcols[1],
                    "alpha": sv[gridcols[0], gridcols[1]],
                    "species": "substrate",
                    "count": f["solute_value"][gridcols[0], gridcols[1]],
                }
            )

            data = pandas.concat([bio, phg, solute], sort=False)
            # colors.append('#ffffff')
            colors.append("#3c1978")
            lims.append("substrate")
        else:
            data = pandas.concat([bio, phg], sort=False)

        data = data[data["count"] > 0]
        if crop is not None:
            data = data[(data.x >= crop[1][0]) & (data.x < crop[1][1])].copy()
            data["x"] -= crop[1][0]
            shape = (crop[0], crop[1][1] - crop[1][0])

        # # lims = ['susceptible', 'resistant', 'infected', 'phage']
        # sections = np.unique([x.split(':')[0] for x in f['config'].T[0]])
        # if any(['matrix' in x for x in sections]):
        #     colors.append('yellow')
        #     lims.append('matrix')
        #     for name in [x for x in data.species.unique() if 'matrix' in x]:
        #         data.loc[data.species == name, 'species'] = 'matrix'

        dl = 3
        # dl = int(cfg.space.dl * 1e6)
        # bio.loc[bio.species == 'Species1', 'species'] = 'susceptible'
        # bio.loc[bio.species == 'Species2', 'species'] = 'resistant'

        timestr = "{:.2f} days".format(f["t"]).rjust(10, " ")

        data["y"] += 1
        plot = ggplot()
        plot += geoms.geom_tile(
            aes(x="x", y="y", alpha="alpha", fill="factor(species)"), data=data
        )
        plot += scales.scale_alpha_continuous(range=[0, 1], limits=[0, 1], guide=False)
        plot += scales.scale_fill_manual(values=colors, limits=lims, drop=False)
        plot += labels.ggtitle("t =" + timestr)
        plot += scales.scale_y_continuous(
            limits=[0, shape[0]],
            labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
            expand=[0, 0],
        )
        plot += scales.scale_x_continuous(
            limits=[0, shape[1]],
            labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
            expand=[0, 0],
        )
        plot += guides(fill=guide_legend(title=""))
        plot += labels.xlab("") + labels.ylab("")
        plot += theme_classic()
        plot += theme(plot_title=element_text(size=8))
        wid = 5.0
        hi = wid * shape[0] / shape[1]
        plot.save(
            outname, height=hi, width=wid, dpi=150, verbose=False, limitsize=False
        )


def noframe(f, outdir):
    outname = "{}/{}.png".format(outdir, f[:-4])
    return not os.path.isfile(outname)


def plot_all_frames(
    datadir,
    outdir,
    crop=None,
    maxp=None,
    maxb=None,
    format="png",
    style=None,
    nprocesses=4,
    phage_time=False,
    solute_adjustment=None,
    **kwargs,
):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    flist = [
        "{}/{}".format(datadir, x)
        for x in os.listdir(datadir)
        if ".npz" in x and noframe(x, outdir)
    ]
    if not flist:
        return
    cfg = sb.getcfg(flist[0])
    shape = tuple(map(int, cfg.space.shape))

    if len(shape) == 2:
        maxb, maxp = 6.5e-12, 1000
        # solute_adjustment = 0.05
        if nprocesses > 1:
            if (maxb is None and maxp is None) or phage_time:
                with mp.Pool(nprocesses) as p:
                    maxes = p.starmap(_get_max, [(fname, shape) for fname in flist])
                    if maxb is None:
                        maxb = max(x[0] for x in maxes)
                    if maxp is None:
                        maxp = int(max(x[1] for x in maxes))
                    if phage_time:
                        return

            with warnings.catch_warnings(), mp.Pool(nprocesses) as p:
                warnings.simplefilter("ignore")  # ignore log10(0) warnings
                p.starmap(
                    _plot_frame,
                    [
                        (
                            f,
                            outdir,
                            shape,
                            crop,
                            maxb,
                            maxp,
                            format,
                            "",
                            style,
                            solute_adjustment,
                        )
                        for f in flist
                    ],
                )
        else:
            if maxb is None and maxp is None:
                maxes = [_get_max(fname, shape) for fname in flist]
                if maxb is None:
                    maxb = max(x[0] for x in maxes)
                if maxp is None:
                    maxp = int(max(x[1] for x in maxes))
            for f in flist:
                print(f)
                _plot_frame(
                    f,
                    outdir,
                    shape,
                    crop,
                    maxb,
                    maxp,
                    format,
                    style=style,
                    solute_adjustment=solute_adjustment,
                )
                break  # FIXME REMOVE ME -- debugging only
    if len(shape) == 3:
        maxb, maxp = 6.5e-12, 1000
        if nprocesses > 1:
            with warnings.catch_warnings(), mp.Pool(nprocesses) as p:
                warnings.simplefilter("ignore")  # ignore log10(0) warnings
                p.starmap(
                    _render_3d_frame,
                    [(f, outdir, shape, maxb, maxp, format, "") for f in flist],
                )
        else:
            for f in flist:
                _render_3d_frame(f, outdir, shape, 6.5e-12, 1000, format, "")


def _get_max(fname, shape):
    maxb, maxp = 0, 0
    total = np.zeros(np.prod(shape))
    with np.load(fname) as f:
        time = f["t"]
        for name in [x for x in f.keys() if "_data" in x]:
            total[:] = 0
            if "phage" in name:
                np.add.at(total, f[name]["location"], 1)
                maxp = max([maxp, total.max()])
            else:
                np.add.at(total, f[name]["location"], f[name]["mass"])
                maxb = max([maxb, total.max()])
    return time, maxb, maxp


def _stripnewline(thisstr):
    return thisstr[:-1] if thisstr.endswith("\n") else thisstr


def _get_end_condition(filename):
    with open(filename, "r") as f:
        last = None
        for line in f:
            second = last
            last = line
    if "Total time" not in last:
        raise RuntimeError("Run {} did not complete properly".format(filename))
    if "End time" in second:
        second = "FIN-Time"
    time = float(last.split(" ")[-1]) / 3600  # hours
    return second, time


def load_data(
    datadir,
    header=1,
    write_summaries=True,
    force_load=False,
    funcs=None,
    dtypes={},
    return_cfgs=False,
):
    """ Funcs is for additional processing before the file gets written, so it doesn't
    need to be repeated in the future. """
    print("Loading cfg...", flush=True)
    cfgs = pandas.read_csv(datadir + "/parameters.csv", header=header, dtype="category")
    # cfgs.insert(0, "general:run", range(1, len(cfgs) + 1))
    cols = [rc.split(":")[1] for rc in cfgs.columns]
    if len(cols) > len(set(cols)):
        fcols = [rc.replace(":", "_") for rc in cfgs.columns]
        cols = [f if cols.count(c) > 1 else c for f, c in zip(fcols, cols)]
    cfgs.columns = cols
    cfgs.run = cfgs.run.astype(int)
    cfgs = cfgs.set_index("run").drop(columns='output_frequency')

    print("Loading summaries...", flush=True)
    feathersummary = datadir + "/full_summary.feather"
    if not os.path.isfile(feathersummary) or force_load:
        summaries = _load_summaries(datadir, np.array(cfgs.index))

        print("Adding cfg to summaries...", flush=True)
        summaries.index = summaries.index.astype(int).astype('category')
        df = cfgs[cfgs.index.isin(summaries.index)].combine_first(summaries)
        for k, v in dtypes.items():
            df[k] = df[k].astype(v).astype("category") if k in cfgs else df[k].astype(v)

        df.iteration = df.iteration.astype(int)
        if "site" in df:
            df["site"] = df.site.astype(int).astype("category")

        try:
            for f in funcs:
                df = f(df)
        except TypeError:  # Covers not callable or iterable
            if funcs:
                df = funcs(df)

        if write_summaries:
            try:
                print("Writing summaries to {}".format(feathersummary), flush=True)
                df.reset_index().to_feather(feathersummary)
            except:
                print("Tried and failed to write summaries.")

    else:
        df = pandas.read_feather(feathersummary)
        df = df.set_index("run")

    df.index = df.index.astype("category")
    return df if not return_cfgs else (df, cfgs)


def _load_summaries(datadir, runs):

    # def _loadone(filename, run, data, failed, simn=None):
    def _loadone(filename, simn=None):
        try:
            with open(filename) as summaryfile:
                run_summary = pandas.read_csv(summaryfile)
        except FileNotFoundError:
            failed.append(run if simn is None else f"{run}.{simn}")
            return
        run_summary["run"] = int(run)
        if simn is not None:
            run_summary["site"] = simn
        data.append(run_summary.set_index("run"))

    ##
    data = []
    failed = []
    multisim = len(
        [f for f in os.listdir(f"{datadir}/data/run_{1:06}") if match("^run\d+$", f)]
    )

    if multisim > 0:
        fnstring = f"{datadir}/data/run_{{:06}}/run{{}}/summary.csv"
    else:
        fnstring = f"{datadir}/data/run_{{:06}}/summary.csv"

    for run in tqdm80(runs):
        if multisim > 0:
            for i in range(1, multisim + 1):
                _loadone(fnstring.format(int(run), i), i)
        else:
            _loadone(fnstring.format(datadir, run))

    if failed:
        print("Unable to load: {}\n{} Total".format(failed, len(failed)))
    return pandas.concat(data, sort=True)


def plot_ribbon_traj(
    data,
    facets,
    prefix="",
    traj_variable="f",
    color=None,
    format="png",
    tmin=None,
    tmax=None,
    alpha=0.9,
    trim=False,
    lw=0.5,
    title=None,
):

    # data = data.copy()

    tmin = tmin if tmin is not None else 0
    tmax = tmax if tmax is not None else 10

    plt.interactive(False)

    ntotal = np.prod([data[x].unique().size for x in facets])
    for facs, df in tqdm80(data.groupby(list(facets)), total=ntotal):
        # for facs, df in data.groupby(list(facets)):
        if trim:
            df = df[df.time >= 0]
        with warnings.catch_warnings():

            df = df[df.t_bin < 10]
            warnings.simplefilter("ignore")  # ignore matplotlib deprecation warnings

            plot = ggplot()

            plot += geoms.geom_ribbon(
                aes(
                    x="t_bin * 24",
                    ymin="CI99min",
                    ymax="CI99max",
                    group="factor(f_bin)",
                ),
                df,
                alpha=0.6,
            )

            if color in data:
                plot += geoms.geom_line(
                    aes(
                        x="t_bin * 24",
                        y="fmean",
                        group="factor(f_bin)",
                        color=f"factor({color})",
                    ),
                    df,
                    alpha=alpha,
                    size=lw,
                    show_legend=False,
                )
            else:
                plot += geoms.geom_line(
                    aes(x="t_bin * 24", y="fmean", group="factor(f_bin)"),
                    df,
                    alpha=alpha,
                    size=lw,
                    show_legend=False,
                    color=color,
                )

            plot += labels.xlab("Time after infection (hours)")
            plot += labels.ylab("Resistant strain frequency")
            plot += scales.scale_x_continuous(
                limits=(tmin if not trim else 0, 5 * 24),
                breaks=np.arange(0, 6 * 24, 12),
            )
            # plot += scales.ylim([0, 1])
            # plot += plotnine.coord_cartesian(xlim=(tmin if not trim else 0, tmax), ylim=(0, 1))
            # plot += scales.scale_y_continuous(limits=(0, 1), expand=(0, 0))
            # plot += scales.scale_x_continuous(limits=(tmin, tmax))

            if not trim:
                plot += geoms.geom_vline(aes(xintercept=0), data=df)

            # if title:
            #     plot += labels.ggtitle(title)
            plot += labels.ggtitle(f'Nbulk = {df["max"].iloc[0]}')

            plot += theme_bw()

            wid = 4.5
            hei = 2.5
            if len(facets) > 1:
                facetnames = [
                    x if isinstance(x, str) else y[0] + str(round(x, 2))
                    for x, y in zip(facs, facets)
                ]
            else:
                facetnames = facets[0] + str(round(facs, 2))
            outname = "{}{}.{}".format(prefix, "_".join(facetnames), format)
            plot.save(
                outname,
                height=(hei + 1) * 2,
                width=(wid + 1) * 2,
                verbose=False,
                limitsize=False,
            )


def plot_trajectories(
    data,
    facets,
    prefix="",
    yvar="f",
    format="png",
    tmin=None,
    tmax=None,
    colorby=None,
    alpha=0.10,
    lw=0.5,
    title=None,
):

    data = data.copy()
    # data = data[data['f2-f1'] < 0.01]
    if colorby is not None:
        colorby = "factor({})".format(colorby)

    tmin = tmin if tmin is not None else data.time.min()
    tmax = tmax if tmax is not None else data.time.max()

    plt.interactive(False)

    ntotal = np.prod([data[x].unique().size for x in facets])
    for facs, df in tqdm80(data.groupby(list(facets)), total=ntotal):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore matplotlib deprecation warnings

            plot = ggplot()
            if colorby is not None:
                plot += geoms.geom_line(
                    aes(x="time", y=yvar, group="factor(run)", color=colorby),
                    df,
                    alpha=alpha,
                    size=lw,
                )
            else:
                plot += geoms.geom_line(
                    aes(x="time", y=yvar, group="factor(run)"), df, alpha=alpha, size=lw
                )

            plot += labels.xlab("Time after infection (days)")
            plot += labels.ylab("R/(S+R)") if yvar == "f" else labels.ylab(yvar)
            plot += scales.scale_x_continuous(limits=(tmin, tmax))
            # plot += scales.scale_x_continuous(limits=(tmin, tmax))

            if title:
                plot += labels.ggtitle(title)
            plot += theme_bw()

            wid = 5.5
            hei = 2.5
            if len(facets) > 1:
                facetnames = [
                    x if isinstance(x, str) else y[0] + str(round(x, 2))
                    for x, y in zip(facs, facets)
                ]
            else:
                facetnames = facets[0] + str(round(facs, 2))
            outname = "{}{}.{}".format(prefix, "_".join(facetnames), format)
            plot.save(
                outname,
                height=(hei + 1) * 2,
                width=(wid + 1) * 2,
                verbose=False,
                limitsize=False,
            )


def plot_convergence(data, groups, name, nresamples=None, repeats=50):
    alldata = []
    nruns = min(data.groupby(groups).apply(len))
    if not nresamples:
        nresamples = nruns
    Y = np.linspace(1, nruns, nresamples, dtype=int)
    for gr, df in tqdm(data.groupby(groups)):
        _dat = np.zeros((nresamples, repeats))
        _mean = df[name].mean()
        for j, n in enumerate(Y):
            for i in range(repeats):
                _dat[j, i] = np.random.choice(df[name], size=n).mean() - _mean
        alldata.append(_dat / df[name].mean())

    dat = np.concatenate(alldata, axis=1)
    plt.plot(Y, np.nanmean(dat, axis=1))
    # __import__('ipdb').set_trace()


def plot_landscape(data, x, y, z):
    plt.interactive(False)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    melted = data.melt([x, y], z, value_name=z).groupby([x, y]).mean().reset_index()
    xs, ys = len(melted[x].unique()), len(melted[y].unique())
    X = np.array(melted[x]).reshape((xs, ys))
    Y = np.array(melted[y]).reshape((xs, ys))
    Z = np.array(melted[z]).reshape((xs, ys))

    # sz = 512
    # br = cm.get_cmap('seismic_r', sz)
    # br = np.concatenate([br(np.linspace(0, 0.5, sz // 2)), br(np.linspace(0.5, 1, sz // 2))[::4]])
    # brcm = matplotlib.colors.ListedColormap(br)

    surface = ax.plot_surface(
        X, Y, Z, cmap=cm.seismic_r, linewidth=0, antialiased=False, vmin=0, vmax=0.6
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    fig.colorbar(surface, shrink=0.5, aspect=5)
    plt.interactive(True)
    plt.show()


def plot_line(
    data,
    xcol,
    ycol,
    groupby=None,
    fmin=None,
    fmax=None,
    errorbars=False,
    prefix="",
    title=None,
    format="png",
    transform=None,
):

    axes = [xcol, ycol]
    if groupby is not None:
        axes.append(groupby)

    # data = {
    #     'mean': data[[xcol, ycol]].groupby([xcol]).mean()[ycol].rename('mean'),
    #     'std': data[[xcol, ycol]].groupby([xcol]).std()[ycol].rename('std')
    # }

    all_cols = [xcol, ycol] if groupby is None else [xcol, ycol, groupby]
    groupcols = [xcol] + ([] if groupby is None else [groupby])
    df = data[all_cols].groupby(groupcols).mean()
    df["std"] = data[all_cols].groupby(groupcols).std()[ycol]
    df["ymin"] = np.clip(df[ycol] - df["std"], 0, np.inf)
    df["ymax"] = df[ycol] + df["std"]

    plot = ggplot(df.reset_index()) + theme_bw()
    # plot += labels.xlab(xcol) + labels.ylab(ycol)

    # plot += labels.xlab('Initial frequency of resistant strain')
    # plot += labels.ylab('Final frequency')

    if fmin is not None and fmax is not None:
        plot += scales.scale_y_continuous(limits=(fmin, fmax))
    if groupby is None:
        plot += geoms.geom_line(aes(x=xcol, y=ycol))
    else:
        plot += geoms.geom_line(
            aes(x=xcol, y=ycol, group=groupby, color=f"factor({groupby})")
        )
        # plot += geoms.geom_point(aes(x=xcol, y=ycol, group=groupby, color=f'factor({groupby})'))
        # ,
        # shape=f'factor(z)'))
    if errorbars:
        plot += geoms.geom_errorbar(aes(x=xcol, ymin=f"{ycol}-std", ymax=f"{ycol}+std"))
    if title:
        plot += labels.ggtitle(title)

    outname = "{}{}-vs-{}.{}".format(prefix, ycol, xcol, format)
    plot.save(outname, height=4, width=4, verbose=False, limitsize=False)
    print(f"Saved {outname}.", flush=True)


def plot_heatmap(
    data,
    xcol,
    ycol,
    fill,
    facets=None,
    prefix="",
    format="png",
    norm_axes=True,
    fmin=None,
    fmax=None,
    xaxislabel=None,
    yaxislabel=None,
    xlabs=None,
    ylabs=None,
    summary="mean",
    alternate_color=False,
    title=None,
):

    # TODO: fix varnames.
    data = data.copy()
    data[fill] = np.clip(data[fill], fmin, fmax)
    if facets is None:
        axes = [xcol, ycol]
    else:
        axes = facets + [xcol, ycol]
    if (
        "run" in data and len(data[data.run == data.run.unique()[0]]) > 1
    ):  # if not already filtered somehow
        data = data.groupby("run").apply(lambda x: x.iloc[-1])
    if summary == "mean":
        df = data.groupby(axes).mean().reset_index()
    elif summary == "var":
        df = data.groupby(axes).var().reset_index()
    elif summary == "sum":
        df = data.groupby(axes).sum().reset_index()
    elif callable(summary):
        df = data.groupby(axes).apply(summary).reset_index()

    xlabs = (
        xlabs if xlabs is not None else ["{:0.2f}".format(x) for x in df[xcol].unique()]
    )
    ylabs = (
        ylabs if ylabs is not None else ["{:0.2f}".format(y) for y in df[ycol].unique()]
    )

    # TODO:
    if norm_axes:
        if not callable(norm_axes):
            for ax in [xcol, ycol]:
                groups = df[ax].unique()
                a = np.zeros(len(df), dtype=type(df[ax][0]))
                for i, u in enumerate(groups):
                    a[df[ax] == u] = i
                df[ax] = a
        else:
            norm_axes(df, xcol, ycol)

    if facets:
        height = round((df[(facets[0])].unique().size + 0.25) * 0.8, 1)
        if len(facets) > 1:
            width = round((df[(facets[1])].unique().size + 0.25) * 1.3, 1)
        else:
            width = height
        height, width = height * 2.5, width * 2.5
    else:
        height, width = 7.0, 7.0

    # with ignore_copywarn():
    #     df[fill][df[fill] < 0] *= 4
    plot = ggplot(data=df)
    plot += labels.xlab(xcol if xaxislabel is None else xaxislabel)
    plot += labels.ylab(ycol if yaxislabel is None else yaxislabel)
    plot += scales.scale_y_continuous(
        breaks=df[ycol].unique(), labels=ylabs, expand=(0.01, 0.01)
    )
    # plot += scales.scale_x_reverse(breaks=df[xcol].unique(), labels=xlabs)
    plot += scales.scale_x_continuous(
        breaks=df[xcol].unique(), labels=xlabs, expand=(0.005, 0.005)
    )
    if facets is not None:
        plot += facet_wrap(facets, ncol=3)

    # if len(facets) > 1:
    #     plot += facet_wrap(facets, ncol=(df[facets[-1]].unique().size))
    # else:
    #     plot += facet_wrap(facets, ncol=int(np.sqrt(df[facets[0]].unique().size)))

    plot += theme(axis_text_x=element_text(rotation=90, hjust=1))
    plot += theme_bw()

    plot += geoms.geom_tile(aes(x=xcol, y=ycol, fill=fill))
    fmax = fmax if fmax is not None else df[fill].max()
    fmin = fmin if fmin is not None else df[fill].min()
    # midpoint = (fmin + fmax) / 2
    midpoint = 3

    if title:
        plot += labels.ggtitle(title)

    # stepsize = 0.5
    # plot += scales.scale_fill_gradientn(
    #     colors=("#352a86", "#37b89d", "#f8fa0d"),
    #     values=(0, 1/2, 1),
    #     breaks=np.arange(fmin, fmax + stepsize / 10, stepsize),
    #     limits=(fmin, fmax),
    # )
    stepsize = (fmax - fmin) / 5

    fmin = -0.10
    fmax = 0.55
    stepsize = 0.1
    plot += scales.scale_fill_gradientn(
        colors=("#b2182b", "#f7f7f7", "#2166ac"),
        # values=(0, 0.1, 1),
        # breaks=np.arange(0, 100, 10),
        values=(0, fmin / (fmin - fmax), 1),
        breaks=np.arange(fmin, fmax + stepsize / 10, stepsize),
        limits=(fmin, fmax),
        midpoint=midpoint,
    )

    # plot += scales.scale_fill_gradient(limits=(fmin, fmax), low='#f5f5f5',
    #                                    high='#101088', midpoint=midpoint)

    # plot += scales.scale_fill_gradient(limits=(fmin, fmax), low='#352a86',
    #                                    high='#f8fa0d', midpoint=5.5)

    # plot += scales.scale_fill_gradient(limits=(fmin, fmax))

    # scales.scale_fill_gradientn()
    # if summary == 'var':
    #     plot += scales.scale_fill_gradient(limits=(fmin, fmax), low='#f5f5f5',
    #                                        high='#101088', midpoint=midpoint)
    # elif alternate_color:
    #     plot += scales.scale_fill_gradient2(limits=(fmin, fmax), low='#a6611a', mid='#f7f7f7',
    #                                         high='#018571', midpoint=midpoint)
    # else:
    # plot += scales.scale_fill_gradient2(limits=(fmin, fmax), midpoint=midpoint)
    outname = "{}{}.{}".format(prefix, fill, format)
    # height = 5 * height
    # plot.save(outname, height=4, width=7.5, dpi=150,
    # plot.save(outname, height=height*3, width=width*3, dpi=150,
    plot.save(
        outname,
        height=height * 1.5,
        width=width * 1.5,
        dpi=150,
        verbose=False,
        limitsize=False,
    )
    print(f"Saved {outname}.", flush=True)


def plot_fitnessvsinvestment(data, prefix="", facets=None, format="png", title=None):
    data = data.copy()
    final = data.groupby("run").apply(lambda x: x.iloc[-1])
    final["fitness"] = final.Species1_mass / (final.Species1_mass + final.Species2_mass)
    # final['fitness'] = final.Species1_mass / final.Species2_mass
    final["investment"] = np.round(final.eps_chance / (1 - final.eps_chance), 3)
    final["rhoratio"] = np.round(2e5 / final.density, 3)

    facets = ["parameter_group", "eps_time", "infection_start"]
    initials = ["" if isinstance(final[f].iloc[0], str) else f[0] for f in facets]

    tot = np.prod([len(final[f].unique()) for f in facets])
    # for _, df in tqdm80(data.groupby('run'), total=tot):
    for group, groupdata in tqdm80(final.groupby(facets), total=tot):
        # if group[1] > group[2]:
        #     continue
        plot = ggplot()

        for _, df in groupdata.groupby("rhoratio"):
            aesth = aes(
                x="investment",
                y="fitness",
                group="investment",
                color="factor(rhoratio)",
            )
            plot += geoms.geom_boxplot(aesth, df, width=0.02)
            aesth = aes(
                x="investment", y="fitness", group="rhoratio", color="factor(rhoratio)"
            )
            plot += geoms.geom_smooth(aesth, df, span=0.4)

            # aesth = aes(x='investment', y='fitness', group='rhoratio', color='factor(rhoratio)')
            # plot += geoms.geom_line(aesth, df.groupby('investment').median().reset_index())

        plot += labels.ggtitle(
            ", ".join(["{}={}".format(f, val) for f, val in zip(facets, group)])
        )

        plot += geoms.geom_hline(yintercept=[0.5])
        plot += labels.xlab("investment") + labels.ylab("eps+ / eps-")
        plot += theme_bw()

        title = "_".join(["{}{}".format(i, v) for i, v in zip(initials, group)])
        outname = "{}_{}.{}".format(prefix, title, format)
        wid = 7
        hei = 5
        plot.save(
            outname, height=hei + 1, width=wid + 1, verbose=False, limitsize=False
        )


def explorer(df, plotfunc, groups, sets=None, onupdate=None, updatef=None):
    """ Plotfunc takes an axis and dataframe and returns an updated plot. """

    def update(val):
        wd = orig[orig.group == groupB.value_selected]
        ax.cla()  # Clear axes
        if onupdate:
            onupdate(wd, ax)
        else:
            plotfunc(wd, ax)
        fig.canvas.draw_idle()

    plt.interactive(False)
    fig, ax = plt.subplots(norws=1, ncols=2)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    orig = df.copy()
    wd = orig

    plotfunc(wd, ax)

    sliders = []
    buttons = []

    groupB = RadioButtons(ax, groups)
    buttons.append(groupB)

    # [s.on_changed(update) for s in sliders]
    [b.on_clicked(update) for b in buttons]

    plt.interactive(True)
    plt.show()
    return

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2)
    ax.margins(x=0)

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()


    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        sfreq.reset()
        samp.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)

    plt.show()


def _main():
    matplotlib.use("agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video")
    parser.add_argument("-f", "--format")
    parser.add_argument("--crop", nargs=3)
    parser.add_argument("--maxb")
    parser.add_argument("--maxp")
    parser.add_argument("--solute_adjustment")
    parser.add_argument("--interval")
    parser.add_argument("--name")
    parser.add_argument("--prefix")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--jobs")
    parser.add_argument("--old", action="store_true")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    if args.video:

        if not args.format:
            raise RuntimeError("Video requires format argument (-f)")
        else:
            kwargs = {"format": args.format}

        if args.crop:
            kwargs["crop"] = [int(x) for x in args.crop]
        if args.maxb:
            kwargs["maxb"] = float(args.maxb)
        if args.maxp:
            kwargs["maxp"] = int(args.maxp)
        if args.solute_adjustment:
            kwargs["solute_adjustment"] = float(args.solute_adjustment)
        if args.interval:
            kwargs["interval"] = int(args.interval)
        if args.name:
            kwargs["name"] = args.name
        if args.prefix:
            kwargs["prefix"] = args.prefix
        if args.jobs:
            kwargs["nprocesses"] = int(args.jobs)
        if args.output:
            kwargs["outdir"] = args.output
        kwargs["progress_bar"] = args.progress
        kwargs["skip_made"] = args.skip

        # print(kwargs)
        if args.old:
            plot_all_frames(args.video, **kwargs)
        else:
            make_video(args.video, **kwargs)


if __name__ == "__main__":
    _main()
