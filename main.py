import numpy as np
from cli import *
from utils import Timer

from image_handling import parse_drawing
from context_correction import context_aware_correction, Output
from inference import infer_drawing

def main():
    args = arguments()
    T = Timer()

    T.TIME()
    outs = parse_drawing(args.input_file)
    orig = outs['orig']
    PARAMS['orig'] = orig
    T.TIME()
    T.print('parse_drawing time:')

    T.TIME()
    outs = context_aware_correction(orig,outs)
    T.TIME()
    T.print('context correction time:')

    orig2 = np.copy(orig)
    Output.draw_ocr_group_rects(orig2, outs['ocr_groups_horz'], outs['ocr_groups_verz'])
    #Output.draw_colinear_lines(orig2,outs['colinear_groups'])
    Output.draw_dimensions(orig2, outs['dimensions'])
    save(orig2,'output2.png')

    T.TIME()
    outs = infer_drawing(orig,outs)
    T.TIME()
    T.print('infer drawing time:')


    do_outputs(orig,outs)


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()')
    main()

