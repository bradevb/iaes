import cv2 as cv
import numpy as np


def load_image(img_or_path, flags=None):
    """
    Loads and returns image from path if necessary, otherwise returns the image.
    :param img_or_path: Can be a filesystem path or a pre-loaded image.
    :param flags: cv2 flags you would like the image to be loaded with.
    :return: A loaded image from cv2.
    """
    if type(img_or_path) is np.ndarray:
        return img_or_path
    img = cv.imread(img_or_path, flags)
    if img is None:
        raise Exception('Error loading image.\nPass in an image loaded by cv.imread or a valid path to an image.')
    return img


def show_result(img, topmost=False, timeout=0):
    """
    Shows an image. Primarily for debugging purposes.
    :param img: The image to show.
    :param topmost: Pass True to make the resulting window a floating one.
    :param timeout: How long to wait before closing the window. Default is forever.
    :return: None
    """
    window_name = 'TEST IMAGE'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    if topmost:
        cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
    cv.imshow(window_name, img)
    cv.waitKey(timeout)
    cv.destroyWindow(window_name)
