from cli import *
from image_handling import parse_drawing


def main():
    args = arguments()

    outs = parse_drawing(args.input_file)
    orig = outs['orig']

    do_outputs(orig,outs)




if __name__ == '__main__':
    main()

