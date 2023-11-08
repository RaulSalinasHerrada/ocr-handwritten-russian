import torch
import cv2
from torchvision import transforms
import numpy as np
from time import time

from ocr.api.htr_model.htr_params import *
from ocr.api.htr_model.htr_utils import process_image
from ocr.api.htr_model.htr_classes import TransformerModel
from ocr.api.htr_model.htr_utils import indicies_to_text

def init_htr_model(argv = None):
    
    htr_model = TransformerModel(
    len(ALPHABET),
    hidden=HIDDEN,
    enc_layers=ENC_LAYERS,
    dec_layers=DEC_LAYERS,   
    nhead=N_HEADS,
    dropout=DROPOUT).to(DEVICE)
    
    htr_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    return htr_model

# predict one subimage
def predict_image_htr(model, img:np.ndarray, save = True, filename = None):
    
    if save:
        if filename is None:
            filename = int(time())
        cv2.imwrite("./image_data/sub_{}.png".format(filename), img)
    
    with torch.no_grad():
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = process_image(img).astype('uint8')
        img = img / img.max()
        img = np.transpose(img, (2, 0, 1))
        src = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
        src = transforms.Grayscale(CHANNELS)(src)
        out_indexes = model.predict(src)
        pred = indicies_to_text(out_indexes[0], idx2char)
        return pred
