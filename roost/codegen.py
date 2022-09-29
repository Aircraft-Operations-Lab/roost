#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from roc3.bada4 import BADA4_jet_CR as jet
from jinja2 import Template, Environment, StrictUndefined, UndefinedError
from pathlib import Path

fdir = Path(__file__).parent / 'cuda'


def prune_terms(src):
    l = ["0*__powf(M, {0})  +".format(i) for i in range(5)] + \
        ["0*__powf(M, 5)"] + ["0*__powf(CF, {0})  +".format(i) for i in range(4)] + \
        ["0*__powf(CF, 4)"]
    for term in l:
        src = src.replace(term, "")
    n_subs = 1
    while n_subs > 0:
        n_subs = 0
        for s1, s2 in [
            (r"\+[ ]*__powf\([a-zA-Z_]*, \d\)\*\([ ]*\)", ""),
            # (r"\+     powf\(delta_T_flat_MCRZ, \d\)\*\([ ]*\)", ""),
            (r"\+[ ]*\)", ")"),
            (r"[ ]*\n;", ";"),
            (r"\*__powf\([a-zA-Z_]*, 0\)", ""),
            (r"__powf\([a-zA-Z_]*, 0\)\*", ""),
            (r"__powf\(([a-zA-Z_]*), 1\)", r"\1"),
            (r"__powf\(([a-zA-Z_]*), 1.0\)", r"\1"),
            #(r"0\*__powf\(([a-zA-Z_]*), ([a-zA-Z0-9_]*)\)", r"0"),

            (r"\+[ ]*powf\([a-zA-Z_]*, \d\)\*\([ ]*\)", ""),
            (r"\*powf\([a-zA-Z_]*, 0\)", ""),
            (r"powf\([a-zA-Z_]*, 0\)\*", ""),
            (r"powf\(([a-zA-Z_]*), 1\)", r"\1"),
            (r"powf\(([a-zA-Z_]*), 1.0\)", r"\1"),
        ]:
            (src, n) = re.subn(s1, s2, src)
            n_subs += n
    return src


def get_bada_source(ac_label, **kwargs):
    apm = kwargs['apm']
    if apm is None:
        apm = jet(ac_label)
    params = apm.get_parameters()
    params['a_nonnull'] = [sum(abs(params['a'][6*i + j]) for j in range(6))  > 0 for i in range(6)]
    params['f_nonnull'] = [sum(abs(params['f'][5 * i + j]) for j in range(5)) > 0 for i in range(5)]

    for k, v in params.items():
        if type(v) == list:
            for i in range(len(v)):
                if v[i] == 0.0:
                    v[i] = 0

    def define_parameter(name):
        value = params[name]
        if type(value) == float:
            return "#define {0} {1}\n".format(name, value)

    define_text = "".join(define_parameter(n) for n in params if type(params[n]) == float)

    with open(os.path.join(fdir, 'badalib.cu'), 'r') as f:
        src = f.read()
    tpl = Template(src, trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
    params['DEFINE_PARAMETERS'] = define_text
    params['avoid_negative_thrust'] = kwargs['avoid_negative_thrust']
    try:
        src_t = tpl.render(**params)
    except UndefinedError as ue:
        for p, p_def in params.items():
            print(f"{p}: {p_def}")
        raise ue
    return prune_terms(src_t)


def get_geolib_source(n_trjs=128, n_nodes=128, n_ec=6, **kwargs):
    define_params = {
        'N_trjs': n_trjs,
        'N_nodes': n_nodes,
        'N_ec': n_ec,
    }
    with open(os.path.join(fdir, 'geolib.cu'), 'r') as f:
        src = f.read()
    tpl = Template(src, trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
    src_t = tpl.render(**define_params)
    return src_t


def get_pptp_source(**kwargs):
    bada_src = get_bada_source(kwargs['aircraft'], **kwargs)
    gl_src = get_geolib_source(**kwargs)
    modules = ['wxtex', 'accf', 'pptp']
    src_raw = ""
    for module in modules:
        with open(os.path.join(fdir, f'{module}.cu'), 'r') as f:
            src_piece = f.read()
            src_raw += src_piece
    pptp_tpl = Template(src_raw, trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
    pptp_src = pptp_tpl.render(**kwargs)
    return bada_src + gl_src + pptp_src


def get_pptp_source_and_annotated_lines(**kwargs):
    src = get_pptp_source(**kwargs)
    annotated_src = ""
    for i, line in enumerate(src.split('\n')):
        annotated_src += f"{i:04} {line}"
    return src, annotated_src

