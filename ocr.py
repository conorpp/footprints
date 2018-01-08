from tesserocr import PyTessBaseAPI, RIL, iterate_level
from PIL import Image
from utils import *
import sys

ABCs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
WHITELIST = "1234567890.,X()/\\" + ABCs
OCR_API = PyTessBaseAPI()
OCR_API.SetVariable('tessedit_pageseg_mode',"7")
OCR_API.SetVariable('tessedit_char_whitelist',WHITELIST)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: %s <input.png [...]>' % sys.argv[0])
        sys.exit(1)

    def num_from_name(x):
        nums='1234567890'
        n = ''
        for y in x:
            if y in nums:
                n+=y
        return int(n)

    with PyTessBaseAPI() as api:
        api.SetVariable('tessedit_pageseg_mode',"7")
        api.SetVariable('tessedit_char_whitelist',WHITELIST)

        for x in sorted(sys.argv[1:], key=lambda x:num_from_name(x)):
            im = load_image(x)
            api.SetImageBytes(im.tobytes(), im.shape[1], im.shape[0], 1, im.shape[1])

            text = api.GetUTF8Text()  # r == ri
            conf = api.MeanTextConf()
            if text:
                symbol = text[0]
                print(u'%s == %s, %d%%' % (x,symbol, conf),)
            else:
                print(u'%s == nothing, %d%%' % (x,conf),)
