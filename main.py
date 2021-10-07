import concurrent.futures
import shutil
import tempfile
import threading
import os

import numpy as np
import pytesseract
from colorama import Fore
from pynput import keyboard, mouse
from tqdm import tqdm

import image_utils
import cspace
import processors
import cv2 as cv
from screenshot import screencapture, get_window_id
from extractor import FormExtractor, CellExtractor
from form.iaes_forms import TopForm, BottomForm, TopBottomForm, Cell
from threadpool import threadpool
from validators.months.month_helpers import calc_balance
from validators import TOP_VALIDATORS, BOTTOM_VALIDATORS, TOP_BOTTOM_VALIDATORS

ENV = os.getenv('ENV')
IMG_OVERRIDE = os.getenv('IMG_OVERRIDE')
IMG_OVERRIDE = IMG_OVERRIDE.split(',') if IMG_OVERRIDE is not None else IMG_OVERRIDE
IMG_PATH_OVERRIDE = os.getenv('IMG_PATH_OVERRIDE')
DEV_HOTKEYS = os.getenv('DEV_HOTKEYS')
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

# For converting screenshots to the right color space
CSPACE_PATH = './icc/IAES_COLOR_PROFILE.icc'

# HSV values used to replace/detect colors throughout the application
TEXT_COLOR_LOW = (0, 0, 0)
TEXT_COLOR_HIGH = (179, 255, 182)
ORANGE_LOW = (12, 190, 206)
ORANGE_HIGH = (179, 255, 255)
ORANGE_REFINED_LOW = (12, 209, 246)
ORANGE_REFINED_HIGH = (20, 240, 255)
RED_LOW = (4, 165, 255)
RED_HIGH = (11, 244, 255)
QUESTION_MARK_LOW = (111, 151, 178)
QUESTION_MARK_HIGH = (179, 172, 222)
SELECTION_LOW = (97, 158, 195)
SELECTION_HIGH = (112, 177, 255)


class RemoteDesktop:
    def __init__(self, app_name, window_name, cspace_path=None):
        self.window_name = window_name
        self.app_name = app_name
        self.cspace_path = cspace_path
        self.has_run = False
        self.win_id = None
        self._windows = None

    def screenshot_remote(self):
        self.check_win_id()

        with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
            screencapture.screenshot_window(self.app_name, title=self.window_name, filename=temp_image.name,
                                            window_selection_options='on_screen_only')

            # Set screenshot's colorspace
            cspace.set_cspace(temp_image.name, self.cspace_path)

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


def _dev_cap_rem(img_path, cspace_path):
    """Simulates taking a screenshot of remote. Takes a pre-saved screenshot's path."""
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_image:
        shutil.copy2(img_path, temp_image.name)

        # Set screenshot's colorspace
        cspace.set_cspace(temp_image.name, cspace_path)

        return image_utils.load_image(temp_image.name)


def get_captiva_form(img):
    def get_form_ext(form_img):
        preprocessors = [
            lambda i: processors.convert_gray(i),
            lambda i: processors.gaussian_blur(i),
            lambda i: processors.thresh(i)
        ]

        return FormExtractor(form_img, preprocessors)

    def good_form_check(form_img):
        mask = image_utils.get_color_mask(form_img, SELECTION_LOW, SELECTION_HIGH)
        if not np.any(mask):
            return

        blue_contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        good_contours = []

        for c in blue_contours:
            _, _, w, h = cv.boundingRect(c)
            if w * h > 40:
                good_contours.append(c)

        return len(good_contours) == 1 or image_utils.check_color(form_img, QUESTION_MARK_LOW, QUESTION_MARK_HIGH)

    form_ext = get_form_ext(img)
    for f in form_ext.get_images():
        if good_form_check(f):
            return f


def get_cells(img) -> list:
    def get_cell_ext(form_img):
        replacement = (255, 255, 255)

        preprocessors = [
            lambda i: image_utils.replace_color(i, ORANGE_LOW, ORANGE_HIGH, replacement),
            lambda i: processors.draw_over_unwanted_areas(i),
            lambda i: processors.convert_gray(i),
            lambda i: processors.thresh(i, 200, 225, cv.THRESH_BINARY_INV)
        ]

        return CellExtractor(form_img, preprocessors)

    cell_ext = get_cell_ext(img)
    cell_ext.extract()

    groups = cell_ext.group_cells(150, 100)

    # Sort individual groups into rows
    groups = [sorted(g, key=lambda c: (c.coords[1], c.coords[0])) for g in groups]
    # Sort groups by descending y
    groups.sort(key=min)
    # Then sort each row in bottom form by x
    bot_form_idx = 1 if len(groups) > 1 else 0
    bot_form_chunks = chunks(groups[bot_form_idx], 5)
    bot_form = []
    for i in bot_form_chunks:
        bot_form.extend(sorted(i, key=lambda c: c.coords[0]))

    groups[bot_form_idx] = bot_form

    return groups


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_cell(cell: Cell):
    def remove_cursor(cell_img):
        """Finds and removes cursors from any cell by finding lines that are long enough to be the cursor,
        and writing over them."""
        temp_img = cell_img.copy()
        temp_img = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
        temp_img = cv.threshold(temp_img, 210, 255, cv.THRESH_BINARY)[1]

        # Adjust expected cursor height by the height of the cell. This makes the cursor removal work when larger
        # monitors are used and cells are larger
        cursor_scale = temp_img.shape[0] / 16
        cursor_height = int(11 * cursor_scale)
        cursor_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, cursor_height))

        # Get mask and invert it with ~ so that cursor's position/pixels are now white
        cursor_mask = ~cv.morphologyEx(temp_img, cv.MORPH_CLOSE, cursor_kernel, iterations=1)
        c_mask_count = cv.countNonZero(cursor_mask)

        # Check if cell has a lot of cursor pixels. If it has over 50, this cell is more than likely a selected cell,
        # and therefore needs special processing
        if c_mask_count > 50 * cursor_scale:
            temp_img = process_selected(cell_img)
            return temp_img

        # Otherwise, just check if there ARE cursor pixels
        contours, _ = cv.findContours(cursor_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            temp_cell = cell_img.copy()
            temp_cell[cursor_mask == 255] = 255
            return temp_cell

        return cell_img

    def process_selected(cell_img):
        """Special care needs to be taken for the selected cell. This processing is done to get the best results from
        tesseract. """
        temp_img = cell_img.copy()

        temp_img = ~image_utils.get_color_mask(temp_img, TEXT_COLOR_LOW, TEXT_COLOR_HIGH)
        temp_img = cv.cvtColor(temp_img, cv.COLOR_GRAY2BGR)
        temp_img = cv.resize(temp_img, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
        temp_img = cv.GaussianBlur(temp_img, (7, 7), 0)

        return temp_img

    img = cell.image.copy()
    img = remove_cursor(img)

    # NOTE: maybe make this check for number of nonzero instead of just if there are any
    if not image_utils.check_color(img, TEXT_COLOR_LOW, TEXT_COLOR_HIGH):
        return cell

    if img.shape[0] < 54:
        s = 54 / img.shape[0]
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)

    text = pytesseract.image_to_string(img, config='--psm 6')
    cell.text = text.replace('\n', '').replace('\f', '').replace('\t', '')
    return cell


class StopThread(Exception):
    pass


def cancel_futures(futures):
    for f in futures:
        f.cancel()


def build_table(cell_instances, col_names, stop: threading.Event):
    """Takes a list of Cell instances and column names, and builds a table from them. Then, it passes all cells to be
    parsed by Tesseract. Cell instances are modified in place with their new row indexes and column names."""
    table = []
    rows = chunks(cell_instances, len(col_names))

    # Prepare the table and Cell list for OCR step
    cells = []
    for row_idx, r in enumerate(rows):
        row = {}

        for c, name in zip(r, col_names):
            row[name] = None  # Initialize a spot in table for this cell
            c.row_idx, c.col_name = row_idx, name
            cells.append(c)

        table.append(row)

    # Spawn OCR threads. These are threads instead of processes because pytesseract spawns subprocesses of tesseract,
    # and are therefore similar to an IO bound task.
    futures = []

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        for cell in cells:
            if stop.is_set():
                cancel_futures(futures)
                raise StopThread

            futures.append(executor.submit(parse_cell, cell))

        with tqdm(futures) as pbar:
            for f in concurrent.futures.as_completed(futures):
                if stop.is_set():
                    cancel_futures(futures)
                    raise StopThread

                cell = f.result()
                table[cell.row_idx][cell.col_name] = cell.text
                pbar.update()

    return table if len(table) > 1 else table[0]


def trim_cell_borders(cell_groups, threshold):
    """Trims off the gray outline on cells"""
    for group in cell_groups:
        for cell in group:
            img = cell.image.copy()
            thresh = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(thresh, threshold, 255, cv.THRESH_BINARY)[1]
            _, img = image_utils.trim_recursive(thresh, img)
            cell.image = img


def pad_beginning(ls, ele, amount):
    """Pads the beginning of *ls* with *ele* inplace. Specify the amount of *ele* with *amount*."""
    for _ in range(amount):
        ls.insert(0, ele)


def replace_ele_with_ele(ls1, ls2, value):
    """Replaces elements of *ls1* with elements of *ls2* if ls1's element == value"""
    if len(ls1) != len(ls2):
        raise ValueError('Length of lists must be the same')

    return [ele2 if ele1 == value else ele1 for ele1, ele2 in zip(ls1, ls2)]


class NeedsScroll(Exception):
    def __init__(self, message, top_form):
        self.message = message
        self.top_form = top_form

    def __str__(self):
        return self.message


def check_cell_groups(cells, prev_top_cells):
    """Function that ensures that the top and bottom forms are being read correctly."""

    def should_scroll(bot_cells, rows):
        """Reads through bottom *rows* rows of table, and if there are cells containing text, return True,
        otherwise return False"""
        num_of_cells = rows * 5
        for c in bot_cells[-num_of_cells:]:
            if image_utils.check_color(c.image, TEXT_COLOR_LOW, TEXT_COLOR_HIGH):
                return True
        return False

    prev_top = prev_top_cells

    if len(cells) == 2:
        top_form, bot_form = cells
    elif len(cells) == 1:
        top_form = []
        bot_form = cells[0]
    else:
        raise RuntimeError('Unable to properly read form. Please ensure that both the top and bottom forms are '
                           'completely in view, and that there is nothing blocking either of them.\nAlso ensure '
                           'that the currently selected cell is in view.')

    # Pad beginning of top_form with None in case there are some that are out of view.
    # This is to ensure that the cells are always in the correct order even if there are some out of view.
    pad_beginning(top_form, None, 7 - len(top_form))

    if None in top_form:
        if not prev_top:
            if len(cells) != 2 or None in top_form:
                raise NeedsScroll(
                    'Unable to read entire top form. Please scroll up all the way and ensure that the top '
                    'form is completely within view. \nAfter it has been read through the first time, '
                    'you may scroll back down.', top_form)
        elif None in prev_top or len(prev_top) != 7:
            raise NeedsScroll(
                'Unable to read entire top form. Please scroll up all the way and ensure that the top '
                'form is completely within view. \nAfter it has been read through the first time, '
                'you may scroll back down.', top_form)
        else:
            top_form = replace_ele_with_ele(top_form, prev_top, None)

    if len(bot_form) % 5 != 0:
        raise RuntimeError("Couldn't properly read bottom form.")

    if should_scroll(bot_form, 2):
        raise NeedsScroll('Please scroll down so that the entire bottom table is in view.', top_form)

    return top_form, bot_form


def parse_and_validate(prev_top_cells: list, events: dict, dev_image_path=None):
    print('Parsing and validating forms...')

    stop = events['stop']
    val_failed = events['val_failed']
    scroll = events['scroll']

    if DEV:
        if not dev_image_path:
            raise RuntimeError('Image path must be passed in order to run in DEV!')
        image = _dev_cap_rem(dev_image_path, CSPACE_PATH)
    else:
        rdp = RemoteDesktop(APP_NAME, WINDOW_NAME, CSPACE_PATH)
        image = rdp.screenshot_remote()

    if stop.is_set():
        return

    captiva_form = get_captiva_form(image)
    if captiva_form is None:
        val_failed.clear()
        raise RuntimeError("Couldn't find IAES form. Please ensure Captiva is pulled up and an IAES form is on "
                           "screen.\nIf this error continues to occur, please try moving the window containing the "
                           "IAES document.")

    # Check for image snippet red box. If it's there, it could be blocking cells, so raise an error.
    if image_utils.check_color(captiva_form, RED_LOW, RED_HIGH):
        val_failed.clear()
        raise RuntimeError('Image snippet detected. Please select a blank cell and rescan.\nIt is recommended that '
                           'you turn off image snippets by going to View -> Image Snippets.')
    elif image_utils.check_color(captiva_form, ORANGE_REFINED_LOW, ORANGE_REFINED_HIGH):
        val_failed.clear()
        raise RuntimeError('A cell highlighted in orange has been found. Please fix that cell and rescan.')

    cells = get_cells(captiva_form)
    trim_cell_borders(cells, 210)

    # Verify that top and bottom cells are being detected correctly, and check if the user needs to scroll up to get
    # the top form in view
    try:
        top_form, bot_form = check_cell_groups(cells, prev_top_cells)
    except RuntimeError as e:
        val_failed.clear()
        raise e
    except NeedsScroll as e:
        scroll.set()
        print(e)
        return e.top_form

    try:
        top_table = TopForm(build_table(top_form, TOP_COL_NAMES, stop), validators=TOP_VALIDATORS)
        bot_table = BottomForm(build_table(bot_form, BOT_COL_NAMES, stop), validators=BOTTOM_VALIDATORS)
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
        return top_form
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
        return top_form
    finally:
        scroll.clear()


@threadpool
def threadpool_parse_validate(*args, **kwargs):
    res = parse_and_validate(*args, **kwargs)
    return res


def main_thread(events: dict):
    stop = events['stop']
    go = events['go']
    t = concurrent.futures.ThreadPoolExecutor().submit(lambda: None)
    prev_top_cells = []

    while True:
        try:
            res = t.result()
            if res is not None:
                prev_top_cells = res
        except Exception as e:
            print(f'error occurred\n{e}')

        go.wait()
        go.clear()
        stop.clear()
        t = threadpool_parse_validate(prev_top_cells, events)


class DelayedEvent:
    def __init__(self, wait, event_func):
        self._timer = threading.Timer(0, lambda: None)
        self._event_func = event_func
        self.wait = wait

    def trigger(self):
        self._timer.cancel()
        self._timer = threading.Timer(self.wait, self._event_func)
        self._timer.start()


def main():
    prev_top_cells = []

    stop = threading.Event()
    go = threading.Event()
    scroll = threading.Event()
    val_failed = threading.Event()
    events = {'stop': stop, 'go': go, 'scroll': scroll, 'val_failed': val_failed}

    if DEV and not DEV_HOTKEYS:
        base_path = './tests/images' if IMG_PATH_OVERRIDE is None else IMG_PATH_OVERRIDE

        if IMG_OVERRIDE is not None:
            for image_num in IMG_OVERRIDE:
                parse_and_validate(prev_top_cells, events, f'{base_path}/{image_num}')
            return

        for image_num in [f for f in os.listdir(base_path) if f.endswith('.png')]:
            parse_and_validate(prev_top_cells, events, f'{base_path}/{image_num}')
        return

    main_t = threading.Thread(target=main_thread, args=(events,), daemon=True)
    main_t.start()

    print('Waiting for hotkey...')
    print('Make sure there is a blank cell selected when you press the hotkey.')

    # TODO: add a cancel hotkey that, when pushed, runs val_failed.clear()

    def on_hotkey():
        val_failed.clear()
        go.set()  # Sets stop to True, and triggers main_thread to begin parsing.

    def on_any_press(key):
        canon = key_listener.canonical(key)
        hotkey.press(canon)

    any_release_timer = DelayedEvent(0.5, go.set)

    def on_any_release(key):
        canon = key_listener.canonical(key)
        hotkey.release(canon)

        if val_failed.is_set():
            stop.set()
            any_release_timer.trigger()

    scroll_timer = DelayedEvent(0.5, go.set)

    def on_scroll(x, y, dx, dy):
        if scroll.is_set():
            stop.set()
            scroll_timer.trigger()

    hotkey_combo = keyboard.HotKey.parse('<alt>' if DEV else '<f4>')
    hotkey = keyboard.HotKey(hotkey_combo, on_hotkey)

    key_listener = keyboard.Listener(on_press=on_any_press, on_release=on_any_release)
    key_listener.start()
    mouse_listener = mouse.Listener(on_scroll=on_scroll)
    mouse_listener.start()

    main_t.join()


if __name__ == '__main__':
    main()
