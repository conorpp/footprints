
from cli import *

from image_handling import parse_drawing
from context_correction import context_aware_correction
from inference import infer_drawing

def main():
    args = arguments()

    t1 = TIME()
    outs = parse_drawing(args.input_file)
    orig = outs['orig']
    PARAMS['orig'] = orig
    t2 = TIME()
    print('parse_drawing time: %d' % (t2-t1))

    t1 = TIME()
    outs = context_aware_correction(orig,outs)
    t2 = TIME()
    print('context correction time: %d' % (t2-t1))

    t1 = TIME()
    outs = infer_drawing(orig,outs)
    t2 = TIME()
    print('infer drawing time: %d' % (t2-t1))


    do_outputs(orig,outs)


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()')
    main()

