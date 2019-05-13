import pickle
from optparse import OptionParser
import os
import cv2
import numpy as np
import sys


parser = OptionParser()
parser.add_option("-i", "--input_file", dest="input_file", help="Path to input image file.", default = "0a0c7c1698944fa19fe72ae77e20cd7d.jpg")
parser.add_option("-o", "--output_file", dest="output_file", help="Path to output label file.", default="output.mp4")
parser.add_option("-d", "--input_dir", dest="input_dir", help="Path to input working directory.", default="/home/asprohy/data/traffic/test_data")
parser.add_option("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="/home/asprohy/data/traffic/test_data")
# parser.add_option("-r", "--frame_rate", dest="frame_rate", help="Frame rate of the output video.", default=25.0)

(options, args) = parser.parse_args()
if not options.input_file:   # if filename is not given
    parser.error('Error: path to image input_file must be specified. Pass --input-file to command line')

input_image_file = options.input_file
output_image_file = options.output_file
img_path = os.path.join(options.input_dir, '')
output_path = os.path.join(options.output_dir, '')
num_rois = 4
frame_rate = float(options.frame_rate)

def cleanup():
    print("cleaning up...")
    os.popen('rm -f ' + img_path + '*')
    os.popen('rm -f ' + output_path + '*')

def get_file_names(search_path):
    for (dirpath, _, filenames) in os.walk(search_path):
        for filename in filenames:
            yield filename#os.path.join(dirpath, filename)

def format_img(img, C):
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape

    if width <= height:
        f = img_min_side/width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side/height
        new_width = int(f * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main():

    cleanup()
    sys.setrecursionlimit(40000)
    config_output_filename = 'config.pickle'

    with open(config_output_filename, 'r') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.iteritems()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = num_rois