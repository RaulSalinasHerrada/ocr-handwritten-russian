from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum, auto, unique
from dataclasses import dataclass
import json 

word = "Мы все учились понемногу чему-нибудь и как-нибудь Так воспитанием, Слава Богу,У нас немудрено блеснуть"


class OCRPost(BaseModel):
    bytes_image: bytes
    # filename: Optional[str] = None
    # language: Optional[str] = None
    # doc_type: Optional[str] = None

@unique
class RecogniserType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name
    tesseract = auto()
    htr = auto()

@unique
class DocumentType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name
    scanned = auto()
    photo = auto()
    handwritten = auto()
    

@unique
class LanguageType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    eng = auto()
    rus = auto()
    est = auto()
    

LANG_ISO_639_1 = {
    LanguageType.eng: "en",
    LanguageType.rus: "ru",
    LanguageType.est: "et",
}

@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    w: int
    h: int
    
    @staticmethod
    def is_subset(rct1, rct2) -> bool:
        """Check if rct1 is subset of rct2"""
        if rct1 == rct2:
            return False
        elif (
            rct1.x > rct2.x and
            rct1.y > rct2.y and
            rct1.x + rct1.w < rct2.x + rct2.w and 
            rct1.y + rct1.h < rct2.y + rct2.h):
            return True
        else:
            return False
        
    @staticmethod
    def contains(rct1, rct2) -> bool:
        """Check if rct1 contains rct2"""
        return Rectangle.is_subset(rct2, rct1)
