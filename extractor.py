import cv2 as cv
import numpy as np
import imutils
from utils import load_image, show_result


class FormExtractor:
    def __init__(self, preprocessors, output_process=False):
        assert isinstance(preprocessors, list), "List of processors expected"
        self._preprocessors = preprocessors
        self.output_process = output_process

    def __call__(self, img_or_path):
        # Load image
        self._image = load_image(img_or_path)

        # Apply all preprocessors to the image
        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)

        return self._extract_forms()

    def _extract_forms(self):
        gray = cv.cvtColor(self._processed, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding with a threshold of 230
        ret, th = cv.threshold(gray, 230, 235, 1)

        # Find and sort contours
        contours, hierarchy = cv.findContours(th.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        boxes = []

        for c in contours:
            box = cv.minAreaRect(c)
            box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
            box = np.array(box, dtype="int")
            area = self._processed.shape[0] * self._processed.shape[1]
            if area / 10 < cv.contourArea(box) < area * 2 / 3:
                boxes.append(box)

        return boxes
