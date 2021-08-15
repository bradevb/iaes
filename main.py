import tempfile

import numpy as np

import image_utils
import processors
import cv2 as cv
from screenshot import screencapture
from extractor import FormExtractor, CellExtractor

# These are both temporary for testing. In prod, load them in from config
APP_NAME = 'Microsoft Remote Desktop'
WINDOW_NAME = 'DRPR-RDS-CAP2'


def capture_remote(app_name, window_name):
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
        screencapture.screenshot_window(app_name, title=window_name, filename=temp_image.name)
        return image_utils.load_image(temp_image.name)


def get_form_ext(img):
    preprocessors = [
        lambda i: processors.convert_gray(i),
        lambda i: processors.gaussian_blur(i),
        lambda i: processors.thresh(i)
    ]

    return FormExtractor(img, preprocessors)


def get_captiva_form(img):
    form_ext = get_form_ext(img)
    form_ext.sort_forms()
    forms = form_ext.get_images()
    return forms[0]


def get_cell_ext(img):
    orange_low = (12, 208, 216)
    orange_high = (92, 255, 255)
    replacement = (255, 255, 255)

    preprocessors = [
        lambda i: image_utils.replace_color(i, orange_low, orange_high, replacement),
        lambda i: processors.convert_gray(i),
        lambda i: processors.thresh(i, 200, 225, cv.THRESH_BINARY)
    ]

    return CellExtractor(img, preprocessors, line_width=1, line_min_len=15, output_process=True)


def get_top_form(cells, img):
    """This needs to be reimplemented."""
    return cells[2]  # TODO: get rid of this, but first implement the red pixel check
    if len(cells[2]) == 7:  # Yikes. This needs to be better. Work on classifying cells
        return cells[2]

    for c in cells:  # This too will be replaced with classifying
        if len(c) == 7:
            return c
    else:
        raise RuntimeError('Top form could not be found.')


def get_bottom_form(cells, img):
    """Like get_top_form, this needs to be reimplemented."""
    pad = 25

    bottom_form = sorted(cells[0], key=lambda c: (c[1], c[0]))
    debug_image = img.copy()

    for x, y, w, h in bottom_form:
        cv.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    image_utils.show_result(debug_image)

    for x, y, w, h in bottom_form[::5]:
        tmp_img = img[y:y + h, x - pad:x]  # Get slice to the left of the cell
        debug_image = img.copy()
        cv.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        image_utils.show_result(debug_image)

        gray_img = cv.cvtColor(tmp_img, cv.COLOR_BGR2GRAY)
        white_pix = np.sum(gray_img == 255)
        if white_pix:
            return bottom_form
    else:
        raise RuntimeError('Bottom form could not be found.')


def main():
    image = capture_remote(APP_NAME, WINDOW_NAME)
    # image = image_utils.load_image('./tests/images/130.png')

    captiva_form = get_captiva_form(image)
    image_utils.show_result(captiva_form)
    cell_ext = get_cell_ext(captiva_form)
    cells = sorted(cell_ext.group_cells(75, 45), key=len, reverse=True)

    # top_form = get_top_form(cells, captiva_form)
    # bot_form = get_bottom_form(cells, captiva_form)
    top_form = cells[2]
    bot_form = sorted(cells[0], key=lambda c: (c[1], c[0]))

    # top_form_bounds =  # TODO: delete these boundary variables

    image_utils.show_result(top_form)
    image_utils.show_result(bot_form)


if __name__ == '__main__':
    main()
