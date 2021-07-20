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
        self._apply_preprocessors()

    def __call__(self, img_or_path):
        # Apply all preprocessors to the image
        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)

        return self._extract()

    def _apply_preprocessors(self):
        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)

    def _extract(self):
        pass


class FormExtractor(Extractor):
    """
    Class for separating the different forms and windows of Captiva.
    Recommended preprocessing steps are:
    1. cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    2. cv.GaussianBlur(image, (5, 5), 0)
    3. cv.threshold(image, 230, 235, cv.THRESH_BINARY_INV)
    """

    def __init__(self, img_or_path, preprocessors, output_process=False):
        super().__init__(img_or_path, preprocessors, output_process)

    def _extract(self):
        # Find and sort contours
        contours, hierarchy = cv.findContours(self._processed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        forms = []

        for c in contours:
            box = cv.minAreaRect(c)
            box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
            box = np.array(box, dtype="int")
            area = self._processed.shape[0] * self._processed.shape[1]
            if area / 10 < cv.contourArea(box) < area * 2 / 3:
                form = self._image.copy()
                x, y, w, h = cv.boundingRect(box)
                form = form[y:y + h, x:x + w]
                forms.append(form)

                if self.output_process:
                    image_utils.show_result(form)

        return forms


class CellExtractor(Extractor):
    """
    Class for extracting cells from a table.
    Recommended preprocessing steps for Captiva's IAES form:
    1. cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    2. cv.threshold(gray_scale_image, 200, 225, cv.THRESH_BINARY)
    """

    def __init__(self, img_or_path, preprocessors, line_width=1, line_min_width=20, output_process=False):
        super().__init__(img_or_path, preprocessors, output_process)
        self.line_width = line_width
        self.line_min_width = line_min_width

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

        cells = []
        for x, y, w, h, area in stats[2:]:
            if area > 100:
                cell = self._image[y:y + h, x:x + w]
                cells.append(cell)

        if self.output_process:
            debug_image = self._image.copy()
            for x, y, w, h, area in stats[2:]:
                cv.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_utils.show_result(debug_image)

        return cells
