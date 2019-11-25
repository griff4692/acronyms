def render_args(args):
    for arg in vars(args):
        print(arg, '-->', getattr(args, arg))
