from ocr.api.base_classes import OCRPost
from ocr.api.recogniser import Recogniser
import os

recogniser = Recogniser()

def _recognise(ocr_post:OCRPost) -> str:
    return recogniser.recognise(ocr_post)

def recognise(element: OCRPost| bytes| str| os.PathLike) -> str:
    """Recognise text from image"""
    
    if isinstance(element, str) or isinstance(element, os.PathLike):
        with open(element, 'rb') as f:
            bytes_file = f.read()
            ocr_post = OCRPost(bytes_image=bytes_file)
    
    elif isinstance(element, bytes):
        ocr_post = OCRPost(bytes_image=element)
    
    elif isinstance(element, OCRPost):
        ocr_post = element

    return _recognise(ocr_post)