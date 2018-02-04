
from cli import *

from image_handling import parse_drawing
from context_correction import context_aware_correction


def main():
    args = arguments()

    outs = parse_drawing(args.input_file)
    orig = outs['orig']


    outs = context_aware_correction(orig,outs)

    do_outputs(orig,outs)


if __name__ == '__main__':
    main()

