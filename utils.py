def jaccard_overlap(a, b):
    a, b = set(a), set(b)
    return len(a.intersection(b)) / float(len(a.union(b)))


def render_args(args):
    for arg in vars(args):
        print(arg, '-->', getattr(args, arg))
