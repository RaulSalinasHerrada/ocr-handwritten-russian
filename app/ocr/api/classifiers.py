from ocr.api.base_classes import LanguageType, DocumentType
import numpy as np
import pandas as pd
import cv2
import pytesseract
from langdetect import detect
from ocr.api.preprocessing import get_grayscale, variance_of_laplacian

LANG_CODES = {
    'en': LanguageType.eng,
    'et': LanguageType.est,
    'ru': LanguageType.rus,
    }


def classify_language(image:np.ndarray) -> LanguageType:
    """Classify which language based on image"""
    
    # simple config for detecting language
    
    custom_config = r'-l eng+est'
    
    ocr_data_first = pytesseract.image_to_data(
        image,
        config = custom_config,
        output_type=pytesseract.Output.DATAFRAME
    )
    
    ocr_data_first = ocr_data_first[ocr_data_first.conf > 0]
    
    def _detect_safe(text):
        """Detect, but no exception given if error"""
        try:
            lng = detect(text)
            
            if lng == 'fi': ## finnish is estonian in this case
                lng = 'et'
            
            return lng
        except:
            return "No language"
    
    ocr_data_first['language'] = [_detect_safe(x) for x in ocr_data_first.text]
    
    ocr_data_counts = ocr_data_first.language.value_counts()
    first_language = ocr_data_counts.index[0]
    
    if first_language in ['en', 'et']:
        return LANG_CODES.get(first_language)
        
    else:
        # If not english or estonian, then it should be russian
        return LANG_CODES.get('ru')
        

def _isgray(img:np.ndarray) -> bool:
    """Detect image is grey"""
    if len(img.shape) < 3 or img.shape[2]  == 1:
        return True
        
    b,g,r = cv2.split(img)
    tol = 0.95
    
    bg_test = (b==g).mean() > tol
    br_test = (b==r).mean() > tol
    gr_test = (g==r).mean() > tol
    
    test_grey:bool = bg_test and br_test and gr_test
    
    return test_grey

def classify_document(image:np.ndarray) -> DocumentType:
    """Classify document type"""
    
    ratios =  {
        "a4": 210/297,
        "letter": 8.5/11,
        "legal": 8.5/14,
        }
    
    if _isgray(image):
        return DocumentType.scanned
    
    else:
        shape = image.shape[:2]
        ratio_image = min(shape)/ max(shape)
        tol = 1E-2
        
        def _mape(x,y):
            return abs(x-y)/max(x,y)
        
        
        #tolerance of 1 percent
        in_ratio = any([
            _mape(ratio_image, ratio) < tol for ratio in ratios.values()])
        
        if in_ratio:
            
            threshold_lap = 100
            
            image = get_grayscale(image)        
            var_lap = variance_of_laplacian(image)
            
            if var_lap < threshold_lap:
                return DocumentType.handwritten
            else:
                return DocumentType.photo
        
        else:
            return DocumentType.handwritten
