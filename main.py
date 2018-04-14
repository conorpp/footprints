import numpy as np
from cli import *
from utils import Timer

from image_handling import parse_drawing
from context_correction import context_aware_correction
from inference import infer_drawing

def main():
    args = arguments()
    T = Timer()

    T.TIME()
    outs = parse_drawing(args.input_file)
    orig = outs['orig']
    PARAMS['orig'] = orig
    T.TIME()
    T.echo('parse_drawing time:')

    T.TIME()
    outs = context_aware_correction(orig,outs)
    T.TIME()
    T.echo('context correction time:')

    T.TIME()
    outs = infer_drawing(orig,outs)
    T.TIME()
    T.echo('infer drawing time:')


    do_outputs(orig,outs)


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()')
    main()

