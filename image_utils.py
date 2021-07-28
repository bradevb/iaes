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


def show_result(img, timeout=0, win_name='TEST IMAGE', ensure_destruction=False, topmost=False):
    """
    Shows an image. Primarily for debugging purposes.
    :param img: The image to show.
    :param timeout: How long to wait before closing the window. Default is forever.
    :param win_name: The name of the window.
    :param ensure_destruction: Whether or not to ensure that window is destroyed before next cv.destroyWindow call.
    :param topmost: Pass True to make the resulting window a floating one.
    :return: None
    """
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    if topmost:
        cv.setWindowProperty(win_name, cv.WND_PROP_TOPMOST, 1)
    cv.imshow(win_name, img)
    cv.waitKey(timeout)
    cv.destroyWindow(win_name)
    if ensure_destruction:
        cv.waitKey(1)


def generate_kernels(width, dtype=np.uint8):
    kernel = np.ones((width, width), dtype)
    kernel_h = np.ones((1, width), np.uint8)
    kernel_v = np.ones((width, 1), np.uint8)
    return kernel, kernel_h, kernel_v


def fix_as_binary(img, thresh=127):
    img[img > thresh] = 255
    img[img < thresh] = 0
    return img
