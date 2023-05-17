import math

import img as img
import numpy as np
import cv2 as cv2

def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 209399294

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    return np.convolve(in_signal, k_size, 'full')


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    if in_image.size == 0:
        raise ValueError("Input image is empty.")

    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)



def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    # Define the derivative kernels
    kernel_x = np.array([1, 0, -1])
    kernel_y = np.array([1, 0, -1]).reshape(3, 1)

    # Compute derivatives in x and y directions using convolution
    derivative_x = cv2.filter2D(in_image.astype(np.float32), -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    derivative_y = cv2.filter2D(in_image.astype(np.float32), -1, kernel_y, borderType=cv2.BORDER_REPLICATE)

    # Compute magnitude and direction matrices
    magnitude = np.sqrt(derivative_x ** 2 + derivative_y ** 2)
    direction = np.arctan2(derivative_y, derivative_x)

    return direction, magnitude

def getGaussianKernel(k_size: int, sigma: float) -> np.ndarray:
    # Get the binomial coefficients
    b = np.array([1, 1])
    for i in range(k_size - 2):
        b = np.convolve(b, [1, 1])
    b = b.reshape(1, -1)

    # Compute the Gaussian kernel
    kernel = b.T @ b
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum()

    # Apply the Gaussian function
    kernel *= np.exp(-0.5 * ((np.arange(k_size) - k_size // 2) / sigma)**2)
    kernel /= kernel.sum()

    return kernel

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create the Gaussian kernel
    kernel = getGaussianKernel(k_size, k_size / 3)

    # Apply convolution
    blurred_image = conv2D(in_image.astype(np.float32), kernel)

    return blurred_image.astype(in_image.dtype)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create the Gaussian kernel
    kernel = cv2.getGaussianKernel(k_size, k_size / 3)

    # Apply Gaussian blur
    blurred_image = cv2.filter2D(in_image, -1, kernel)

    return blurred_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Laplacian operator to detect edges
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    # Convert the laplacian image to grayscale with 8-bit depth
    laplacian_8bit = cv2.convertScaleAbs(laplacian)
    # Calculate the zero-crossings in the Laplacian image
    edges = cv2.threshold(laplacian_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    zero_crossings = cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # Convert the zero-crossings image to binary
    zero_crossings = np.uint8(zero_crossings / np.max(zero_crossings) * 255)
    # Return the binary zero-crossings image
    return zero_crossings


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles [(x, y, radius), (x, y, radius), ...]
    """
    circles = []
    threshold = 2  # Threshold value for edge detection

    # Normalize image if pixel values are in the range [0, 1]
    if img.max() <= 1:
        img = (cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)).astype('uint8')

    # Determine the maximum radius based on image dimensions
    radius_limit = min(img.shape[0], img.shape[1]) // 2
    max_radius = min(radius_limit, max_radius)

    accumulator = np.zeros((len(img), len(img[0]), max_radius + 1))  # Accumulator array for circle detection

    # Calculate the gradients using Sobel operator
    x_gradient = cv2.Sobel(img, cv2.CV_64F, 1, 0, threshold)
    y_gradient = cv2.Sobel(img, cv2.CV_64F, 0, 1, threshold)

    # Calculate the direction (angle) of gradients
    gradient_direction = np.arctan2(y_gradient, x_gradient)
    gradient_direction = np.radians(gradient_direction * 180 / np.pi)

    # Adjust the step size based on the difference between min_radius and max_radius
    radius_step = int((max_radius - min_radius) / 60) + 1

    # Detect edges using Canny edge detection
    canny_edges = cv2.Canny(img, 75, 150)

    # Iterate over the image pixels
    for x in range(len(canny_edges)):
        for y in range(len(canny_edges[0])):
            # Check if the pixel is an edge
            if canny_edges[x][y] == 255:
                # Iterate over the radius values within the specified range
                for radius in range(min_radius, max_radius + 1, radius_step):
                    angle = gradient_direction[x, y] - np.pi / 2
                    x1 = (x - radius * np.cos(angle)).astype(np.int32)
                    x2 = (x + radius * np.cos(angle)).astype(np.int32)
                    y1 = (y + radius * np.sin(angle)).astype(np.int32)
                    y2 = (y - radius * np.sin(angle)).astype(np.int32)
                    if 0 < x1 < img.shape[0] and 0 < y1 < img.shape[1]:
                        accumulator[x1, y1, radius] += 1
                    if 0 < x2 < img.shape[0] and 0 < y2 < img.shape[1]:
                        accumulator[x2, y2, radius] +=1

    threshold = np.max(accumulator) * 0.5 + 1  # Update the threshold for circle detection
    x_coords, y_coords, radii = np.where(accumulator >= threshold)  # Get the circles that pass the threshold

    # Add detected circles to the circles list
    circles.extend((y_coords[i], x_coords[i], radii[i]) for i in range(len(x_coords)) if any([x_coords[i], y_coords[i], radii[i]]))
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (np.ndarray, np.ndarray):
    """
    Apply bilateral filter to an image and compare with OpenCV implementation
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    # Apply bilateral filter using OpenCV
    opencv_result = cv2.bilateralFilter(in_image, k_size, sigma_space, sigma_color)

    # Create an output image array for my implementation
    my_result = np.zeros_like(in_image)

    # Adjust kernel size
    kernel_size = int(k_size / 2)

    # Pad the input image with reflection
    padded_image = cv2.copyMakeBorder(in_image, top=kernel_size, bottom=kernel_size, left=kernel_size,
                                      right=kernel_size,
                                      borderType=cv2.BORDER_REFLECT_101).astype(int)

    # Iterate over the image pixels
    for y in range(kernel_size, padded_image.shape[0] - kernel_size):
        for x in range(kernel_size, padded_image.shape[1] - kernel_size):
            pivot_value = padded_image[y, x]

            # Extract the neighborhood of the current pixel
            neighborhood = padded_image[y - kernel_size:y + kernel_size + 1, x - kernel_size:x + kernel_size + 1]

            # Calculate the difference between the pivot value and the neighborhood
            diff = pivot_value - neighborhood

            # Calculate the Gaussian weights for the differences
            diff_gaussian = np.exp(-np.power(diff, 2) / (2 * sigma_space))

            # Calculate the Gaussian weights for the color similarity
            color_gaussian = cv2.getGaussianKernel(2 * kernel_size + 1, sigma=sigma_color)
            color_gaussian = color_gaussian.dot(color_gaussian.T)

            # Combine the Gaussian weights for color and spatial similarity
            combined_weights = color_gaussian * diff_gaussian

            # Calculate the weighted average of the neighborhood
            result = (combined_weights * neighborhood / combined_weights.sum()).sum()

            # Assign the result to the corresponding pixel in the output image
            my_result[y - kernel_size, x - kernel_size] = round(result)

    return opencv_result.astype(int), my_result.astype(int)