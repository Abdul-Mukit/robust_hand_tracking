# Warning: Here I have tried out RGB-BGR conversion for data augmentation. Same is implemented in image_edited.py
import random
from PIL import Image


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2):
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw =int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    flip = random.randint(1, 10000) % 2
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    # cropped.show()

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


jitter = 0.2
hue = 0.1
saturation = 1.5
exposure = 1.5

# jitter = 0
# hue = 0
# saturation = 0
# exposure = 0
shape = [320, 320]

imgpath = 'docs/egohandstrain.jpg'
img = Image.open(imgpath).convert('RGB')
if random.randint(0, 1):
    r, g, b = img.split()
    img = Image.merge('RGB',(b, g, r))
    print('BGR')

img.show()
img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
img.show()