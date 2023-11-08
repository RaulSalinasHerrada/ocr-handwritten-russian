import pytesseract

import base64
import csv
import os
import io
import numpy as np

import cv2

from ocr.api.base_classes import RecogniserType, DocumentType, LanguageType
from ocr.api.base_classes import Rectangle, word
from ocr.api.preprocessing import preprocess_image
from ocr.api.postprocessing import postprocess_text
from ocr.api.htr_model.htr_processing import init_htr_model, predict_image_htr

class Recogniser:    
    ocr_reader: dict

    def __init__(self):
        
        set_recognisers = {RecogniserType.htr}        
        self.ocr_reader = {rt: Recogniser.init_model(rt) for rt in set_recognisers}

    def recognise(self, cv2_image, trying = True) -> str:
        
        if not trying:
            return word
        
        doc_type = DocumentType.handwritten
        cv2_image = preprocess_image(cv2_image, doc_type)
        language = LanguageType.rus    
        recogniser_algo = RecogniserType.htr
        
        ocr_text: str
        psm_mode = 3
        tesseract_command = r"--oem 3 --psm {} -l {}"
        custom_config = tesseract_command.format(psm_mode, language.name)
        
        results = pytesseract.image_to_data(
            cv2_image,
            config=custom_config,
            output_type= pytesseract.Output.DICT)
        
        text_list = []
        
        
        #TODO: iterate over numpy array, not dict
        
        for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]
            
            rct = Rectangle(x,y,w,h)

            box_contain = any([
                Rectangle.contains(rct ,Rectangle(x2,y2,w2,h2)) for x2,y2,w2,h2 in zip(
                        results["left"],
                        results["top"],
                        results["width"],
                        results["height"])])
            
            box_subset = any([
                Rectangle.is_subset(rct ,Rectangle(x2,y2,w2,h2)) for x2,y2,w2,h2 in zip(
                        results["left"],
                        results["top"],
                        results["width"],
                        results["height"])])
            
            if box_subset and not box_contain:
                
                sub_img = cv2_image[y:y+h, x:x+w]
                text = predict_image_htr(
                    self.ocr_reader[recogniser_algo],
                    sub_img)
                text_list.append(text)
                
        ocr_text = " ".join(text_list)
        ocr_text = postprocess_text(ocr_text)
        
        return ocr_text
    
    @staticmethod
    def init_htr_model() -> dict:
        return init_htr_model()
    
    @staticmethod
    def init_model(reg:RecogniserType):
        
        if reg == RecogniserType.htr:
            return Recogniser.init_htr_model()
        
        elif reg == RecogniserType.tesseract:
            return dict()
        
        else:
            raise NotImplementedError("Recogniser not implemented")
