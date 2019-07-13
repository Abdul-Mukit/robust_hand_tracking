# In this program I try to show the efficasy of hand detection using the gamma correction.
# https://www.youtube.com/watch?v=Khy8U_zXDC4
# Usage: python demo_realsense_gamma_analysis.py cfg/yolo-hands.cfg backup/hands/000570.weights

from utils import *
from darknet import Darknet
import cv2
import pyrealsense2 as rs
import collections

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    namesfile = 'data/hands.names'

    class_names = load_class_names(namesfile)

    use_cuda = 1
    if use_cuda:
        m.cuda()

    # RealSense Start
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    # Setting exposure
    s = profile.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, exposure_val)

    # Setting counter for evaluation
    movingList = collections.deque(maxlen=100)



    while True:
        # Reading image from camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())

        if gamma_correction:
            img = adjust_gamma(img, gamma=gamma_val)


        # yolo stuff
        sized = cv2.resize(img, (m.width, m.height))
        bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        print('------')
        draw_img = plot_boxes_cv2(img, bboxes, None, class_names)

        # Evaluation
        movingList.append(any(bboxes))
        print('Continuity : {}'.format(np.mean(movingList)))

        cv2.imshow(cfgfile, draw_img)
        cv2.waitKey(1)


############################################
if __name__ == '__main__':

    exposure_val = 166
    gamma_val = 2
    gamma_correction = False


    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
