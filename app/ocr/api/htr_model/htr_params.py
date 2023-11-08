
from torch.cuda import is_available
from torch import device
import os

def _find(name):
    """find file path on currect directory and subdirectories"""
    
    
    path = os.getcwd()
    print("PATH:", path)
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(path, root, name)

WEIGHTS_PATH =  _find("ocr_transformer_4h2l_simple_conv_64x256.pt")

HIDDEN = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
N_HEADS = 4
LENGTH = 42
ALPHABET = ['PAD', 'SOS', ' ', '!', '"', '%', '(', ')', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
            '[', ']', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И',
            'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х',
            'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е',
            'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т',
            'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            'ё', 'EOS']

BATCH_SIZE = 16
DROPOUT = 0.2
N_EPOCHS = 128
CHECKPOINT_FREQ = 10 # save checkpoint every 10 epochs
DEVICE = device('cuda' if is_available() else 'cpu') # or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
SCHUDULER_ON = True # "ReduceLROnPlateau"
PATIENCE = 5 # for ReduceLROnPlateau
OPTIMIZER_NAME = 'Adam' # or "SGD"
LR = 2e-6

### TESTING ###
CASE = False # is case taken into account or not while evaluating
PUNCT = False # are punctuation marks taken into account

### INPUT IMAGE PARAMETERS ###
WIDTH = 256
HEIGHT = 64
CHANNELS = 1 # 3 channels if model1

char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}