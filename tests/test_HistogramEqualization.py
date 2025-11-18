import os
import numpy as np
from unittest import TestCase
from imgprocalgs.algorithms.HistogramEqualization import HistogramEqualization


class TestHistogramEqualization(TestCase):
    """Unit tests for the HistogramEqualization algorithm"""
    TEST_IMAGE = "tests/data/lena.jpg"  # Assumes this test image exists

    def setUp(self):
        """Runs before each test method - initializes test environment"""
        self.dest_path = "tests/data/"
        self.algorithm = HistogramEqualization(self.TEST_IMAGE, self.dest_path)

    def tearDown(self):
        """Runs after each test method - cleans up test artifacts"""
        # Remove the generated output file after test
        output_path = os.path.join(self.dest_path, "equalized_image.jpg")
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_histogram_generation(self):
        """Test if histogram calculation produces valid results"""
        histogram = self.algorithm._calculate_histogram(self.algorithm.grayscale_img)
        # Verify histogram has 256 bins (for intensity values 0-255)
        self.assertEqual(histogram.shape, (256,))
        # Verify total pixel count in histogram matches image dimensions
        self.assertEqual(histogram.sum(), self.algorithm.width * self.algorithm.height)

    def test_process_output(self):
        """Test if the processing method generates a valid output file"""
        output_path = self.algorithm.process()
        # Verify the output file is created
        self.assertTrue(os.path.exists(output_path))