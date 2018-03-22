
from utils import *
from filters import *
from analyzers import *
from processors import *
from structures import wrap_image, Shape
import preprocessing

def init(input_file):
    arr = load_image(input_file)
    arr = remove_alpha(arr)
    arr = polarize(arr)
    orig = np.copy(color(arr))
    analyzers.init(arr,orig)

    arr = preprocessing.preprocess(arr)
    arr = wrap_image(arr)
    return arr,orig



def parse_drawing(input_file):

    arr,orig = init(input_file)
    print(arr['img'].shape)

    triangles = []
    ocr = []
    lines = []
    rectangles = []
    circles = []
    leftover = []
    irregs = []

    submaps = extract_features([arr])
    submaps = block_clipped_components(submaps)

    analyze_rectangles(submaps)

    r,l = pass_rectangles(submaps)
    rectangles += r
    leftover += l


    ## semi-rects
    semir,leftover = pass_rectangles(leftover,0)

    analyze_circles(semir)
    circles,semir= pass_circles(semir)

    analyze_semi_rects(semir)

    semir,l = pass_semi_rectangles(semir)
    make_irregular_shapes(semir)

    analyze_irregs(l)
    semir2,l = pass_irregs(l)

    outs = separate_irregs(semir2+semir)
    outs = block_dots(outs)
    analyze_rectangles(outs)
    leftover += outs

    irregs += semir
    irregs += semir2
    leftover += l
    ##

    ## OCR
    analyze_ocr(leftover)
    ocr, leftover = pass_ocr(leftover)

    # check orientations
    rotate_right(leftover)
    analyze_ocr(leftover)
    ocr2, leftover = pass_ocr(leftover)
    ocr += ocr2

    rotate_left(ocr2)
    rotate_left(leftover)
    slashes, ocr = pass_slashes(ocr)   # blocking slashes
    #
    ## OCR is pretty greedy so still consider it for everything else
    leftover += ocr
    leftover += slashes
    ##

    analyze_circles(leftover)
    c,leftover = pass_circles(leftover)
    circles += c

    outs = separate_circles(circles)
    outs = block_dots(outs)
    outs = extract_features(outs)
    analyze_rectangles(outs)
    leftover += outs

    # greedily churn out the lines
    while True:
        newleftovers = False

        while True:
            analyze_lines(leftover)
            r,leftover = pass_lines(leftover)
            lines += r

            newlines,leftover = find_line_features(leftover)
            if len(newlines) == 0:
                break
            newleftovers = True

            line_submaps = extract_features(newlines)
            assign_best_fit_lines(line_submaps)
            lines += line_submaps

            leftover = extract_features(leftover)
            leftover = block_dots(leftover)

            analyze_rectangles(leftover)

        if not newleftovers:
            break

        cutleftovers, leftover = cut_linking_lines(leftover)
        cutleftovers = extract_components(cutleftovers)
        analyze_rectangles(cutleftovers)
        leftover += cutleftovers

    analyze_triangles(leftover,arr)
    triangles,leftover = pass_triangles(leftover, arr)

    polish_rectangles(rectangles)

    outs = {'triangles': triangles,
            'ocr': ocr,
            'lines': lines,
            'rectangles': rectangles,
            'circles': circles,
            'irregs': irregs,
            'leftover': leftover,
            'orig': orig,
            'arr': arr
            }
    return outs


