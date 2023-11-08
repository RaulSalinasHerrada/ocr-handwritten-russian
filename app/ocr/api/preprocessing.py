import cv2
import numpy as np
from functools import reduce
from ocr.api.base_classes import DocumentType
from time import time

def preprocess_image(
    image:np.ndarray,
    doc_type: DocumentType,
    filename: str | None = None,
    save=True) -> np.ndarray:
    """Preprocess image

    Args:
        image (np.ndarray): image
        doc_type (DocumentType): type of document
        filename (str | None, optional): filename, if none creates random.
        save (bool, optional): saves on folder. Defaults to True.

    Returns:
        np.ndarray: image processed for OCR
    """
    
    funcs = [
        resize_image,
        get_grayscale,
    ]

    if doc_type == DocumentType.scanned:
        funcs = [
            # resize_image,
            get_grayscale,
            remove_noise_simple,
        ]

    elif doc_type == DocumentType.photo:
        funcs = [
            get_grayscale,
            focus_image,
            # remove_noise_simple,
            remove_noise_deblur,
            # adaptive_thresholding,
            opening,
        ]

    elif doc_type == DocumentType.handwritten:
        funcs = [
            get_grayscale,
            # focus_image,
            # remove_noise_deblur,
            # opening,
            # adaptive_thresholding
        ]

    result = reduce(lambda res, f: f(res), funcs, image)

    if save:
        if filename is None:
            filename = int(time())
        cv2.imwrite("./image_data/{}.png".format(filename), result)

    return result


## methods for image processing

def resize_image(image:np.ndarray):
    """Resize image to A4 size (300 dpi)"""
    # length_x, width_y, _ = image.shape
    # factor = min(1, float(1024.0 / length_x))
    # size = int(factor * length_x), int(factor * width_y)
    # TODO: resize to minimize deformation
    size = (2480, 3507)  # A4 size
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# get grayscale image
def get_grayscale(image:np.ndarray):
    """To gray scale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise_simple(image:np.ndarray):
    """Simple Remove noise with medianblur (3)"""
    return cv2.medianBlur(image, 3)

def remove_noise_deblur(image:np.ndarray):
    """Remove blur"""
    blured1 = cv2.medianBlur(image, 3)
    blured2 = cv2.medianBlur(image, 51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    # th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)
    return normed

# thresholding
def thresholding(image:np.ndarray):
    """Set black and white with threshold"""
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# adaptive thresholding
def adaptive_thresholding(image:np.ndarray):
    """Set black and white with adaptive threshold"""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )


# dilation
def dilate(image:np.ndarray):
    """image dilation"""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion
def erode(image:np.ndarray):
    """image erosion"""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation
def opening(image:np.ndarray):
    """erosion followed by dilation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

# canny edge detection
def canny(image:np.ndarray):
    """Canny edge detection"""
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image:np.ndarray):
    """Skew correction"""
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian

def variance_of_laplacian(image:np.ndarray) -> float:
    return cv2.Laplacian(image, cv2.CV_64F).var()

def focus_image(image:np.ndarray, threshold=100):
    """Focus image"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    if variance_of_laplacian(image) < threshold:
        return cv2.filter2D(image, -1, kernel)
    else:
        return image
    
