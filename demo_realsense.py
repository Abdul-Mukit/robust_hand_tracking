from utils import *
from darknet import Darknet
import cv2
import pyrealsense2 as rs

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    class_names = load_class_names(namesfile)

    use_cuda = 1
    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    # RealSense Start
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    # Setting exposure
    s = profile.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, 166)



    while True:
        # res, img = cap.read()
        # Reading image from camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sized = cv2.resize(img, (m.width, m.height))
        bboxes = do_detect(m, sized, 0.6, 0.4, use_cuda)
        print('------')
        draw_img = plot_boxes_cv2(img, bboxes, None, class_names)

        cv2.imshow(cfgfile, draw_img)
        cv2.waitKey(1)


############################################
if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        namesfile = sys.argv[3]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/hands.names')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile namesfile')
        print('')
        print('    perform detection on camera')
