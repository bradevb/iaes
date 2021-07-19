import cv2 as cv

from image_utils import show_result


def convert_gray(image, output_process=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if output_process:
        show_result(gray)
    return gray


def gaussian_blur(image, kernel_size=(5, 5), sigma=0, output_process=False):
    blur = cv.GaussianBlur(image, kernel_size, sigma)
    if output_process:
        show_result(blur)
    return blur


def thresh(image, thresh1=230, thresh2=235, thresh_type=cv.THRESH_BINARY_INV, output_process=False):
    ret, threshold = cv.threshold(image, thresh1, thresh2, thresh_type)
    if output_process:
        show_result(threshold)
    return threshold
