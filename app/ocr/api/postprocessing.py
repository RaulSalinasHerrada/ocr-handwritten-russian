import re
from functools import reduce

def postprocess_text(ocr_text:str) -> str:
    """Apply postprocessing to OCR text."""
    
    funcs = [
        _correct_email,
        _one_line_break,
        _correct_amount,
        _correct_gibberish,
    ]
    
    result = reduce(lambda res, f: f(res), funcs, ocr_text)
    return result

def _correct_email(text:str) -> str:
    """Replace OCR mail instances that should be @ with @."""

    pseudo_at_list = [
        r'@',
        r'\(G',
        r'\(\)',
        ]
    
    
    def parenthesis(x: str) -> str:
        return '(' + x + ')'
    
    pseudo_at_list = [parenthesis(x) for x in pseudo_at_list]
    pseudo_at_class = parenthesis(r'|'.join(pseudo_at_list))
    
    look_ahead = r'(?=\w+\.\w{2,4})'
    
    regex_email = pseudo_at_class + look_ahead
    
    return re.sub(regex_email,'@',text)

def _correct_amount(text:str)-> str:
    """correct money amount if necesary"""
    return text

def _one_line_break(text:str) -> str:
    """line breaks collapsed into one"""
    return re.sub(r'\n+', '\n', text)
    

def _correct_gibberish(text:str):
    """correct gibberish if necesary"""
    return text