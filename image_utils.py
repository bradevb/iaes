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

    if timeout:
        cv.waitKey(timeout)
    else:
        while cv.waitKey(33) == -1:  # Wait until a key is pressed, or KeyboardInterrupt
            pass

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


def get_color_mask(image, low_thresh, high_thresh):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    color_lo = np.array(low_thresh)
    color_hi = np.array(high_thresh)

    mask = cv.inRange(hsv, color_lo, color_hi)
    return mask


def check_color(image, low_thresh, high_thresh):
    mask = get_color_mask(image, low_thresh, high_thresh)
    return bool(cv.countNonZero(mask))


def replace_color(image, low_thresh, high_thresh, replacement_color):
    mask = get_color_mask(image, low_thresh, high_thresh)
    image[mask > 0] = replacement_color
    return image


def pad_match_concat(big_img, small_img, axis=1):
    """Takes a big image and small image, pads the small image, and concatenates the two. Mainly for displaying two
    different sized images side-by-side."""
    if big_img.shape == small_img.shape:
        return np.concatenate([big_img, small_img], axis=axis)

    padded = np.zeros_like(big_img)
    padded[:small_img.shape[0], :small_img.shape[1]] = small_img

    return np.concatenate([big_img, padded], axis=axis)


def show_cell_groups(img, cell_groups):
    for group in cell_groups:
        tmp = img.copy()
        for c in group:
            x, y, w, h = c.coords
            cv.rectangle(tmp, (x, y), (x + w, y + h), (255, 0, 0), 2)
        show_result(tmp)
