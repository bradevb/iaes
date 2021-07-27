import cv2 as cv
import imutils
import numpy as np

import image_utils


class Extractor:
    """
    Base Extractor class. Takes a list of preprocessors to run before running _extract, which is meant to be
    overwritten. It it meant to be called after being initialized.
    """

    def __init__(self, img_or_path, preprocessors, output_process=False):
        assert isinstance(preprocessors, list), "List of processors expected"
        self._image = image_utils.load_image(img_or_path)
        self._preprocessors = preprocessors
        self.output_process = output_process
        self._processed = self._apply_preprocessors()
        self._extract()

    def _extract(self):
        pass

    def _apply_preprocessors(self):
        processed = self._image
        for preprocessor in self._preprocessors:
            processed = preprocessor(processed)
        return processed


class FormExtractor(Extractor):
    """
    Class for separating the different forms and windows of Captiva.
    Recommended preprocessing steps are:
    1. cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    2. cv.GaussianBlur(image, (5, 5), 0)
    3. cv.threshold(image, 230, 235, cv.THRESH_BINARY_INV)
    """

    def __init__(self, img_or_path, preprocessors, output_process=False):
        self._form_coords = None
        super().__init__(img_or_path, preprocessors, output_process)

    def _extract(self):
        # Find and sort contours
        contours, hierarchy = cv.findContours(self._processed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        form_images = []
        form_coords = []

        for c in contours:
            box = cv.minAreaRect(c)
            box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
            box = np.array(box, dtype="int")
            area = self._processed.shape[0] * self._processed.shape[1]
            if area / 10 < cv.contourArea(box) < area * 2 / 3:
                form = self._image.copy()
                x, y, w, h = cv.boundingRect(box)
                form = form[y:y + h, x:x + w]
                form_images.append(form)
                form_coords.append((x, y, w, h))

                if self.output_process:
                    image_utils.show_result(form)

        self._form_coords = form_coords

    def _generate_form_images(self):
        form_images = []
        for x, y, w, h in self._form_coords:
            form_images.append(self._image[y:y + h, x:x + w])
        return form_images

    def get_images(self):
        return self._generate_form_images()

    def get_coords(self):
        return self._form_coords


def _check_rect_proximity(rect1, rect2, thresh_x, thresh_y):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Neighbor is on left or right
    left_check = abs(x1 - x2) <= thresh_x or abs(x2 - (x1 + w1)) <= thresh_x
    right_check = abs((x1 + w1) - (x2 + w2)) <= thresh_x or abs(x1 - (x2 + w2)) <= thresh_x
    # Neighbor is on top or bottom
    top_check = abs(y1 - y2) <= thresh_y or abs(y2 - (y1 + h1)) <= thresh_y
    bottom_check = abs((y1 + h1) - (y2 + h2)) <= thresh_y or abs(y1 - (y2 + h2)) <= thresh_y

    return (left_check or right_check) and (top_check or bottom_check)


def _merge_lists(ls):
    """
    Takes a list of lists and merges lists if they contain an identical element.
    :param ls: A list of lists.
    :return: A list of merged lists (or sets).
    """
    temp_ls = ls.copy()
    out = []
    while len(temp_ls) > 0:
        first, *rest = temp_ls
        first = set(first)

        first_len = -1
        while len(first) > first_len:
            first_len = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r))) > 0:
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        out.append(list(first))
        temp_ls = rest

    return out


class CellExtractor(Extractor):
    """
    Class for extracting cells from a table.
    Recommended preprocessing steps for Captiva's IAES form:
    1. image_utils.replace_color(image, (12, 208, 216), (92, 255, 255), (255, 255, 255))
    2. cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    3. cv.threshold(gray_scale_image, 200, 225, cv.THRESH_BINARY)
    """

    def __init__(self, img_or_path, preprocessors, line_width=1, line_min_width=20, output_process=False):
        self.line_width = line_width
        self.line_min_width = line_min_width
        self._cell_coords = None
        super().__init__(img_or_path, preprocessors, output_process)

    def _extract(self):
        _, kernel1_h, kernel1_v = image_utils.generate_kernels(self.line_width)
        _, kernel6_h, kernel6_v = image_utils.generate_kernels(self.line_min_width)

        # Bridge small gap in horizontal lines, erode everything else in horizontal direction
        img_bin_h = cv.morphologyEx(~self._processed, cv.MORPH_CLOSE, kernel1_h)
        img_bin_h = cv.morphologyEx(img_bin_h, cv.MORPH_OPEN, kernel6_h)
        # Do the above, but with vertical lines
        img_bin_v = cv.morphologyEx(~self._processed, cv.MORPH_CLOSE, kernel1_v)
        img_bin_v = cv.morphologyEx(img_bin_v, cv.MORPH_OPEN, kernel6_v)

        # Combine img_bins with bitwise or, and turn the result into a binary image
        img_bin_final = image_utils.fix_as_binary(image_utils.fix_as_binary(img_bin_h) |
                                                  image_utils.fix_as_binary(img_bin_v))

        # Dilate the final binary image a little bit to ensure all cells are connected
        final_kernel = np.ones((5, 5), np.uint8)
        img_bin_final = cv.dilate(img_bin_final, final_kernel, iterations=1)

        ret, labels, stats, centroids = cv.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv.CV_32S)

        cell_images = []
        cell_coords = []

        debug_image = self._image.copy()

        for x, y, w, h, area in stats[2:]:
            if area > 100:
                cell = self._image[y:y + h, x:x + w]
                cell_images.append(cell)
                cell_coords.append((x, y, w, h))
                cv.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.output_process:
            image_utils.show_result(debug_image)

        self._cell_coords = cell_coords

    def group_cells(self, dist_x, dist_y=None):
        if dist_y is None:
            dist_y = dist_x

        list_of_rects = sorted(self._cell_coords.copy())

        groups = []
        for rect in list_of_rects:
            neighbors = []

            # Compare each cell with rect to see if it's a neighbor
            for neighbor in list_of_rects:
                within_dist = _check_rect_proximity(rect, neighbor, dist_x, dist_y)
                if within_dist:
                    neighbors.append(neighbor)

            groups.append(neighbors)

        merged_groups = _merge_lists(groups)
        return merged_groups

    def _generate_cell_images(self):
        cell_images = []
        for x, y, w, h in self._cell_coords:
            cell_images.append(self._image[y:y + h, x:x + w])
        return cell_images

    # SPLIT THIS UP INTO A PRIVATE METHOD SO THAT I CAN GET CELL IMAGES WITHIN HERE
    def get_images(self):
        return self._generate_cell_images()

    def get_coords(self):
        return self._cell_coords
