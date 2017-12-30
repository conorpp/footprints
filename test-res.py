import sys,os,json,argparse

from enum import Enum
import io

from google.protobuf import json_format
import google.cloud
from google.cloud import vision_v1

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw

#if len(sys.argv) != 2:
    #print 'usage: %s <input.json>' % sys.argv[0]
    #sys.exit(1)

def open_image_res(filein):
    json_inp = open(filein,'r').read()
    response = vision_v1.types.AnnotateImageResponse()

    json_format.Parse(json_inp,response)
    return response



class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image


def get_document_bounds(json_file, image_file, feature):
    """Returns document bounds given an image."""
    #client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    #response = client.document_text_detection(image=image)
    response = open_image_res(json_file)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)

                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

        if (feature == FeatureType.PAGE):
            bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds

def render_doc_text(jsonin, imin, fileout):
    image = Image.open(imin)
    #bounds = get_document_bounds(jsonin, imin, FeatureType.PAGE)
    #draw_boxes(image, bounds, 'blue')
    #bounds = get_document_bounds(jsonin, imin, FeatureType.PARA)
    #draw_boxes(image, bounds, 'red')
    #bounds = get_document_bounds(jsonin, imin, FeatureType.WORD)
    #draw_boxes(image, bounds, 'yellow')
    bounds = get_document_bounds(jsonin, imin, FeatureType.SYMBOL)
    draw_boxes(image, bounds, 'purple')

    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='The image for text detection.')
    parser.add_argument('image_file', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()

    parser = argparse.ArgumentParser()
    render_doc_text(args.json_file, args.image_file, args.out_file)
