import tempfile

import numpy as np
import pandas as pd
import pytesseract
from tqdm import tqdm

import image_utils
import processors
import cv2 as cv
from screenshot import screencapture, get_window_id
from extractor import FormExtractor, CellExtractor
from form.iaes_forms import TopForm, BottomForm, TopBottomForm

# These are both temporary for testing. In prod, load them in from config
APP_NAME = 'Microsoft Remote Desktop'
WINDOW_NAME = 'DRPR-RDS-CAP2'

TOP_COL_NAMES = ['proj_start_date',
                 'beginning_bal',
                 'proj_min_date',
                 'proj_min_bal',
                 'total_amount',
                 'pi_amount',
                 'monthly_amount']
BOT_COL_NAMES = ['to_date', 'to_amount', 'description', 'from_date', 'from_amount']

TEXT_COLOR_LOW = (0, 0, 0)
TEXT_COLOR_HIGH = (179, 255, 182)


class RemoteDesktop:
    def __init__(self, app_name, window_name):
        self.window_name = window_name
        self.app_name = app_name
        self.has_run = False
        self.win_id = None
        self._windows = None

    def screenshot_remote(self):
        self.check_win_id()

        with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
            screencapture.screenshot_window(self.app_name, title=self.window_name, filename=temp_image.name,
                                            window_selection_options='on_screen_only')

            return image_utils.load_image(temp_image.name)

    def check_win_id(self):
        self._get_windows()

        if not self._validate_win_list():
            self._select_window()
        elif self.has_run and self._windows[0] != self.win_id:
            self._select_window()

        self.has_run = True

    def _select_window(self):
        while not self._validate_win_list():
            print("There's been an error finding the correct Remote Desktop window.")
            print('The window has either changed or is not visible on screen.')
            print('Please put ONLY the window that has the IAES form on the screen, and press Enter.')
            input()
            self._get_windows()

        self.win_id = self._windows[0]

    def _validate_win_list(self):
        if len(self._windows) > 1 or not self._windows:
            return False
        return True

    def _get_windows(self):
        self._windows = list(get_window_id.gen_window_ids(self.app_name, self.window_name, options='on_screen_only'))


def cap_rem(app_name, window_name):  # This is still here for development/testing purposes. TODO: delete it.
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
        lambda i: processors.thresh(i, 200, 225, cv.THRESH_BINARY_INV)
    ]

    return CellExtractor(img, preprocessors, line_width=1, line_min_len=15)


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


def get_cell_images(orig, cells):
    images = []
    temp = orig.copy()

    for x, y, w, h in cells:
        cv.rectangle(temp, (x, y), (x + w, y + h), (255, 255, 255), 3)  # Cover up cell borders
        images.append(temp[y:y + h, x:x + w])
    return images


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_cell(cell, scale=3):
    # Check if cell has text in it
    mask = image_utils.get_color_mask(cell, TEXT_COLOR_LOW, TEXT_COLOR_HIGH)
    text_in_cell = cv.countNonZero(mask)

    if not text_in_cell:
        return None

    img = cell.copy()
    img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    text = pytesseract.image_to_string(img, config='--psm 6')
    return text.replace('\n', '').replace('\f', '').replace('\t', '')


def build_table(form_images, col_names):
    """col_names is a list of expected column names."""
    table = []
    rows = chunks(form_images, len(col_names))

    with tqdm(total=len(form_images)) as pbar:
        for r in rows:
            row = {}

            for cell, name in zip(r, col_names):
                row[name] = parse_cell(cell, scale=3)
                pbar.update()

                # Uncomment for debugging
                # tess_input = image_utils.load_image('./tessinput.tif')
                # both = image_utils.pad_match_concat(tess_input, cell)
                # both = cv.resize(both, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
                # image_utils.show_result(both)

            table.append(row)

    return table if len(table) > 1 else table[0]


def main():
    image = cap_rem(APP_NAME, WINDOW_NAME)  # TODO: change this later to use the RemoteDesktop class

    captiva_form = get_captiva_form(image)
    cell_ext = get_cell_ext(captiva_form)

    groups = cell_ext.group_cells(75, 45)
    cells = sorted(groups, key=len, reverse=True)

    if len(cells) != 3:
        raise RuntimeError("Couldn't find top and bottom forms.")

    top_form_coords = sorted(cells[2], key=lambda c: (c[1], c[0]))
    bot_form_coords = sorted(cells[0], key=lambda c: (c[1], c[0]))

    top_form_images = get_cell_images(captiva_form, top_form_coords)
    bot_form_images = get_cell_images(captiva_form, bot_form_coords)

    top_table = pd.Series(build_table(top_form_images, TOP_COL_NAMES))
    bot_table = pd.DataFrame(build_table(bot_form_images, BOT_COL_NAMES))

    print()


if __name__ == '__main__':
    main()
