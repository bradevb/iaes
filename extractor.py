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
        pass
