import itertools
import string
from PIL import Image, ImageDraw, ImageFont  # version 7.0.0
import cv2                                   # version 4.2.0.32
import numpy as np                           # version 1.18.1


def create_img(ch, pos_x=3, pos_y=2):
    IMG_SHAPE = (17, 14)
    img = Image.new("L", color=0, size=(IMG_SHAPE[1], IMG_SHAPE[0]))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('DejaVuSansMono', size=12)
    draw.text(xy=(pos_x, pos_y), text=ch, fill=255, font=font, spacing=0)

    # display(img)  # for jupyter-qtconsole/notebook
    # img.save("%s.png" % ch)
    return np.array(img)


def dist(ch1, ch2):
    img1 = create_img(ch1)
    img2 = create_img(ch2)
    con1, _ = cv2.findContours(img1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    con2, _ = cv2.findContours(img2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()
    print('%s <-> %s: hd %f' % (ch1, ch2, hd.computeDistance(con1[0], con2[0])))
    print('%s <-> %s: sd %f' % (ch1, ch2, sd.computeDistance(con1[0], con2[0])))


total = 0
fail = 0
for ch1, ch2 in itertools.combinations(string.ascii_letters + string.digits, r=2):
    # Interesting cases
    if (ch1, ch2) in [('r', 'T'), ('s', 'F')]:
        continue
    total += 1
    try:
        dist(ch1, ch2)
    except:
        fail += 1

print("Fail rate %0.2f%%" % ((fail/total) * 100))