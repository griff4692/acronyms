import numpy as np


def render_args(args):
    for arg in vars(args):
        print(arg, '-->', getattr(args, arg))


def safe_multiply(a, b):
    return np.exp(np.log(a + 1e-5) + np.log(b + 1e-5))
