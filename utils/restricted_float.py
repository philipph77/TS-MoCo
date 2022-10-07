import argparse

def restricted_float(x):
    lower_bound=0.0
    upper_bound=1.0
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < lower_bound or x > upper_bound:
        raise argparse.ArgumentTypeError(f"{x} not in range [{lower_bound}, {upper_bound}]")
    return x