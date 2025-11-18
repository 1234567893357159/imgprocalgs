"""Module for histogram equalization algorithm - enhances image contrast by redistributing intensity values"""
import argparse
import os
import numpy as np
from PIL import Image as PillowImage
from imgprocalgs.algorithms.utilities import Image, create_empty_image


class HistogramEqualization:
    """
    Histogram Equalization algorithm implementation.
    Enhances image contrast by transforming the intensity distribution of pixels.
    Works on grayscale images by spreading out intense pixel values.
    """
    def __init__(self, image_path: str, dest_path: str):
        """
        Initialize the histogram equalization processor
        
        :param image_path: Path to the input image file
        :param dest_path: Directory path to save the output image
        """
        self.image = Image(image_path)  # Load input image using project's Image wrapper
        self.dest_path = dest_path      # Destination path for processed image
        self.width, self.height = self.image.get_size()  # Get image dimensions
        self.grayscale_img = self._to_grayscale()  # Convert to grayscale for processing

    def _to_grayscale(self) -> PillowImage:
        """
        Convert input color image to grayscale using luminance formula.
        Maintains perceived brightness based on human visual sensitivity.
        
        :return: Grayscale version of the input image
        """
        # Create empty image with same dimensions as input
        grayscale = create_empty_image(self.width, self.height)
        pixels = grayscale.load()  # Get pixel access object
        
        # Convert each pixel to grayscale using weighted RGB formula
        for x in range(self.width):
            for y in range(self.height):
                r, g, b = self.image.pixels[x, y]
                # Luminance formula: weights correspond to human color perception
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                pixels[x, y] = (gray_value, gray_value, gray_value)  # Grayscale has equal RGB components
        
        return grayscale

    def _calculate_histogram(self, img: PillowImage) -> np.ndarray:
        """
        Calculate frequency distribution of pixel intensities (0-255)
        
        :param img: Grayscale image to analyze
        :return: 1D numpy array where index = intensity, value = pixel count
        """
        pixels = img.load()
        histogram = np.zeros(256, dtype=int)  # 256 possible intensity values (0-255)
        
        # Count occurrences of each intensity value
        for x in range(self.width):
            for y in range(self.height):
                intensity = pixels[x, y][0]  # Get intensity (R component of grayscale)
                histogram[intensity] += 1
        
        return histogram

    def _calculate_cdf(self, histogram: np.ndarray) -> np.ndarray:
        """
        Calculate Cumulative Distribution Function (CDF) from histogram.
        Normalizes to 0-255 range for intensity mapping.
        
        :param histogram: Pixel intensity frequency distribution
        :return: Normalized CDF array for intensity transformation
        """
        # Calculate cumulative sum of histogram values
        cdf = histogram.cumsum()
        
        # Find first non-zero value in CDF to avoid division issues
        cdf_min = cdf[np.nonzero(cdf)[0][0]]
        
        # Normalize CDF to 0-255 range using histogram equalization formula
        total_pixels = self.width * self.height
        cdf_normalized = ((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(np.uint8)
        
        return cdf_normalized

    def process(self) -> str:
        """
        Perform full histogram equalization process and save result.
        
        :return: File path to the processed image
        """
        # Step 1: Calculate histogram and cumulative distribution function
        histogram = self._calculate_histogram(self.grayscale_img)
        cdf = self._calculate_cdf(histogram)
        
        # Step 2: Apply equalization using CDF mapping
        output_img = create_empty_image(self.width, self.height)
        output_pixels = output_img.load()
        input_pixels = self.grayscale_img.load()
        
        # Map each pixel to new intensity using CDF
        for x in range(self.width):
            for y in range(self.height):
                original_intensity = input_pixels[x, y][0]
                equalized_intensity = cdf[original_intensity]  # Apply transformation
                output_pixels[x, y] = (equalized_intensity, equalized_intensity, equalized_intensity)
        
        # Step 3: Save and return output path
        output_path = os.path.join(self.dest_path, "equalized_image.jpg")
        output_img.save(output_path)
        return output_path


def parse_args():
    """Parse command line arguments for histogram equalization"""
    parser = argparse.ArgumentParser(description='Histogram Equalization - Enhance image contrast')
    parser.add_argument("--src", type=str, required=True, help="Source image file path")
    parser.add_argument("--dest", type=str, default='data/', help="Destination directory path")
    return parser.parse_args()


def main():
    """Main entry point for command line execution"""
    args = parse_args()
    
    # Create destination directory if it doesn't exist
    os.makedirs(args.dest, exist_ok=True)
    
    # Execute equalization
    processor = HistogramEqualization(args.src, args.dest)
    output_path = processor.process()
    print(f"Equalized image saved to: {output_path}")


if __name__ == "__main__":
    main()