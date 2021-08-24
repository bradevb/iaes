import concurrent.futures
import tempfile
import threading
import os

import numpy as np
import pytesseract
from colorama import Fore
from pynput import keyboard
from tqdm import tqdm

import image_utils
import processors
import cv2 as cv
from screenshot import screencapture, get_window_id
from extractor import FormExtractor, CellExtractor
from form.iaes_forms import TopForm, BottomForm, TopBottomForm, Cell
from validators.months.month_helpers import calc_balance
from validators import TOP_VALIDATORS, BOTTOM_VALIDATORS, TOP_BOTTOM_VALIDATORS

ENV = os.getenv('ENV')
DEV = ENV == 'DEV'

# These are both temporary for testing. In prod, load them in from config
APP_NAME = 'Microsoft Remote Desktop'
WINDOW_NAME = 'DRPR-RDS-CAP2'

TOP_COL_NAMES = ['proj_start_date',
                 'beginning_bal',
                 'proj_min_date',
                 'proj_min_bal',
                 'total_amount',
                 'pi_amount',
                 'escrow_amount']
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
        # TODO: check to see if previous window id is present. If it is, use its id to screenshot window
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


def _cap_rem(app_name, window_name):
    """Captures remote even when the remote window is in the background. Only for dev and testing purposes."""
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


def parse_cell(cell: Cell, scale=3):
    img = cell.image.copy()

    # Check if cell has text in it
    mask = image_utils.get_color_mask(img, TEXT_COLOR_LOW, TEXT_COLOR_HIGH)
    text_in_cell = cv.countNonZero(mask)

    if not text_in_cell:
        return cell
        # return None
    # TODO: find a way to remove the cursor from a cell if it's there. An easy way to do this might be to just erode
    #  the contents of the cell, and then check the cell for text color (since the text is thicker than the cursor,
    #  there should be some remnants of text color left)
    img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    text = pytesseract.image_to_string(img, config='--psm 6')
    cell.text = text.replace('\n', '').replace('\f', '').replace('\t', '')
    return cell


class StopThread(Exception):
    pass


def build_table(form_images, col_names, stop: threading.Event):
    """col_names is a list of expected column names."""
    table = []
    rows = chunks(form_images, len(col_names))

    # Prepare the table and Cell list for OCR step
    cells = []
    for row_idx, r in enumerate(rows):
        row = {}

        for cell_img, name in zip(r, col_names):
            row[name] = None
            cells.append(Cell(cell_img, row_idx, name))

        table.append(row)

    # Spawn OCR threads. These are threads instead of processes because pytesseract spawns subprocesses of tesseract,
    # and are therefore similar to an IO bound task.
    futures = []

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        for cell in cells:
            if stop.is_set():
                raise StopThread

            futures.append(executor.submit(parse_cell, cell))

        with tqdm(futures) as pbar:
            for f in concurrent.futures.as_completed(futures):
                if stop.is_set():
                    raise StopThread

                cell = f.result()
                table[cell.row_idx][cell.col_name] = cell.text
                pbar.update()

    return table if len(table) > 1 else table[0]


def get_form_bounds(img, cell_group):
    min_x = min(cell_group, key=lambda c: c[0])[0]
    min_y = min(cell_group, key=lambda c: c[1])[1]
    max_x = max(cell_group, key=lambda c: c[0] + c[2])[0]
    max_y = max(cell_group, key=lambda c: c[1] + c[3])[1]

    image = img[min_x:max_x, min_y:max_y]
    image_utils.show_result(image)


# TODO: when sorting cells, fix the bug where the selected cell (which is bigger than the others) gets put in the
#  wrong place
def parse_and_validate(stop: threading.Event, val_failed: threading.Event):
    if DEV:
        image = _cap_rem(APP_NAME, WINDOW_NAME)
    else:
        rdp = RemoteDesktop(APP_NAME, WINDOW_NAME)
        image = rdp.screenshot_remote()

    print('Parsing and validating forms...')

    captiva_form = get_captiva_form(image)
    cell_ext = get_cell_ext(captiva_form)

    groups = cell_ext.group_cells(75, 30)
    cells = sorted(groups, key=len, reverse=True)
    top_form_coords, bot_form_coords = cells[2], cells[0]
    # TODO: Check for presence of red pixels in entire captiva form. If red cells are
    #  found, prompt user to either turn off image snippets in View -> Image Snippets, or to move the selected cell
    #  to an empty one.
    # get_form_bounds(captiva_form, top_form_coords)
    # get_form_bounds(captiva_form, bot_form_coords)

    if len(cells) != 3:
        raise RuntimeError("Couldn't find top and bottom forms.")
    if len(top_form_coords) != 7:
        raise RuntimeError("Couldn't properly read top form.")
    if len(bot_form_coords) % 5 != 0:
        raise RuntimeError("Couldn't properly read bottom form.")

    top_form_coords = sorted(cells[2], key=lambda c: (c[1], c[0]))
    bot_form_coords = sorted(cells[0], key=lambda c: (c[1], c[0]))

    top_form_images = get_cell_images(captiva_form, top_form_coords)
    bot_form_images = get_cell_images(captiva_form, bot_form_coords)

    try:
        top_table = TopForm(build_table(top_form_images, TOP_COL_NAMES, stop), validators=TOP_VALIDATORS)
        bot_table = BottomForm(build_table(bot_form_images, BOT_COL_NAMES, stop), validators=BOTTOM_VALIDATORS)
    except StopThread:
        return

    top_bot_table = TopBottomForm(top_table, bot_table, validators=TOP_BOTTOM_VALIDATORS)

    try:
        top_bot_table.validate()
    except ValueError as e:

        print(f'{Fore.RED}VALIDATION ERROR:')
        print(f'{Fore.RED}{e}')
        print()

        val_failed.set()
    else:
        print(f'{Fore.GREEN}VALIDATORS PASSED!')
        print(f'{Fore.GREEN}Just check descriptions to make sure they line up.')

        # TODO: make this always return two decimal points
        print(f'{Fore.GREEN}This is the expected final balance: '
              f'{calc_balance(bot_table.df, top_table.df["beginning_bal"]).balance.iloc[-1]}')
        print(f'{Fore.GREEN}If you make any changes, just press the hotkey to begin scanning again.')
        print()

        # TODO: Put a print here that tells the user what the last month's balance should be. Remind the user that
        #  they MUST check that with the IAES document. Maybe even pause execution and wait for user to confirm it
        #  matches. If it does not match, there is something wrong, and the user needs to go through the entire form
        #  and double check everything.
        # TODO: Put a print here that checks the descriptions and prints them in red if they don't match expected
        #  descriptions. For County Property Tax(es), put them in yellow so that the user knows they're in the
        #  dictionary, but need to be checked just in case.
        val_failed.clear()  # This is to stop the hotkey listener


def main_thread(stop: threading.Event, val_failed: threading.Event):
    t = threading.Thread()
    while True:
        stop.wait()

        if t.is_alive():
            print('thread is already running, waiting for it to finish.')
            stop.set()
            t.join(10)

        stop.clear()
        t = threading.Thread(target=parse_and_validate, args=(stop, val_failed))
        t.start()


def main():
    stop = threading.Event()
    val_failed = threading.Event()
    threading.Thread(target=main_thread, args=(stop, val_failed), daemon=True).start()

    print('Waiting for hotkey...')
    print('Make sure there is a blank cell selected when you press the hotkey.')

    # TODO: add a cancel hotkey that, when pushed, runs val_failed.clear()

    def on_hotkey():
        val_failed.clear()
        stop.set()  # Sets stop to True, and triggers main_thread to begin parsing.

    def on_any_press(key):
        canon = listener.canonical(key)
        hotkey.press(canon)

    def on_any_release(key):
        canon = listener.canonical(key)
        hotkey.release(canon)

        if val_failed.is_set():
            stop.set()  # Sets stop to True, and triggers main_thread to begin parsing.

    hotkey_combo = keyboard.HotKey.parse('<alt>' if DEV else '<f4>')
    hotkey = keyboard.HotKey(hotkey_combo, on_hotkey)

    with keyboard.Listener(on_press=on_any_press, on_release=on_any_release) as listener:
        listener.join()


if __name__ == '__main__':
    main()
