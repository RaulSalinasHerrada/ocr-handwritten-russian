import os
import string

import numpy as np
import torch
import cv2
import editdistance
import time

from tqdm import tqdm
from torchvision import transforms

from PIL import Image
from  ocr.api.htr_model.htr_params import HEIGHT, WIDTH, DEVICE, CHANNELS, ALPHABET, BATCH_SIZE

# convert images and labels into defined data structures
def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str
        path to directory with images
    labels_dir : str
        path to tsv file with labels
    returns
    ---
    img2label : dict
        keys are names of images and values are correspondent labels
    chars : list
        all unique chars used in data
    all_labels : list
    """

    chars = []
    img2label = dict()

    raw = open(labels_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        try:
            x = t.split('\t')
            flag = False
            for item in ignore:
                if item in x[1]:
                    flag = True
            if flag == False:
                img2label[image_dir + x[0]] = x[1]
                for char in x[1]:
                    if char not in chars:
                        chars.append(char)
        except:
            print('ValueError:', x)
            pass

    all_labels = sorted(list(set(list(img2label.values()))))
    chars.sort()
    chars = ['PAD', 'SOS'] + chars + ['EOS']

    return img2label, chars, all_labels


# TRANSLATE INDICIES TO TEXT
def indicies_to_text(indexes, idx2char):
    text = "".join([idx2char[i] for i in indexes])
    text = text.replace('EOS', '').replace('PAD', '').replace('SOS', '')
    return text


# COMPUTE CHARACTER ERROR RATE
def char_error_rate(p_seq1, p_seq2):
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str
    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(
        ''.join(c_seq1),
        ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))


# RESIZE AND NORMALIZE IMAGE
def process_image(img, save = True):
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = HEIGHT
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = WIDTH
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))
        
    if save:
        # filename = uuid().hex
        filename = str(time.time()).replace('.', '')
        cv2.imwrite("./images/htr/{}.png".format(filename), img)

    return img


# GENERATE IMAGES FROM FOLDER
def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images
    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        img = np.asarray(Image.open(path).convert('RGB'))
        try:
            img = process_image(img)
            data_images.append(img.astype('uint8'))
        except:
            print(path)
            img = process_image(img)
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, criterion, loader, case=True, punct=True):
    """
    params
    ---
    model : nn.Module
    criterion : nn.Object
    loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(loader) : float
        overall loss
    """
    model.eval()
    metrics = {'loss': 0, 'wer': 0, 'cer': 0}
    result = {'true': [], 'predicted': [], 'wer': []}
    with torch.no_grad():
        for (src, trg) in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            logits = model(src, trg[:-1, :])
            loss = criterion(logits.view(-1, logits.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            out_indexes = model.predict(src)
            
            true_phrases = [indicies_to_text(trg.T[i][1:], ALPHABET) for i in range(BATCH_SIZE)]
            pred_phrases = [indicies_to_text(out_indexes[i], ALPHABET) for i in range(BATCH_SIZE)]
            
            if not case:
                true_phrases = [phrase.lower() for phrase in true_phrases]
                pred_phrases = [phrase.lower() for phrase in pred_phrases]
            if not punct:
                true_phrases = [phrase.translate(str.maketrans('', '', string.punctuation))\
                                for phrase in true_phrases]
                pred_phrases = [phrase.translate(str.maketrans('', '', string.punctuation))\
                                for phrase in pred_phrases]
            
            metrics['loss'] += loss.item()
            metrics['cer'] += sum([char_error_rate(true_phrases[i], pred_phrases[i]) \
                        for i in range(BATCH_SIZE)])/BATCH_SIZE
            metrics['wer'] += sum([int(true_phrases[i] != pred_phrases[i]) \
                        for i in range(BATCH_SIZE)])/BATCH_SIZE

            for i in range(len(true_phrases)):
                result['true'].append(true_phrases[i])
                result['predicted'].append(pred_phrases[i])
                result['wer'].append(char_error_rate(true_phrases[i], pred_phrases[i]))

    for key in metrics.keys():
        metrics[key] /= len(loader)

    return metrics, result


# MAKE PREDICTION
def prediction(model, test_dir, char2idx, idx2char):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    char2idx : dict
        map from chars to indicies
    id2char : dict
        map from indicies to chars

    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
    """
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = Image.open(test_dir + filename).convert('RGB')

            img = process_image(np.asarray(img)).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
            if CHANNELS == 1:
                src = transforms.Grayscale(CHANNELS)(src)
            out_indexes = model.predict(src)
            pred = indicies_to_text(out_indexes[0], idx2char)
            preds[filename] = pred

    return preds


class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        self.X_type = X_type

    def __call__(self, X):
        X = X.transpose((2, 0, 1))
        X = torch.from_numpy(X)
        if self.X_type is not None:
            X = X.type(self.X_type)
        return X


def log_config(model):
    print('transformer layers: {}'.format(model.enc_layers))
    print('transformer heads: {}'.format(model.transformer.nhead))
    print('hidden dim: {}'.format(model.decoder.embedding_dim))
    print('num classes: {}'.format(model.decoder.num_embeddings))
    print('backbone: {}'.format(model.backbone_name))
    print('dropout: {}'.format(model.pos_encoder.dropout.p))
    print(f'{count_parameters(model):,} trainable parameters')


def log_metrics(metrics, path_to_logs=None):
    if path_to_logs != None:
        f = open(path_to_logs, 'a')
    if metrics['epoch'] == 1:
        if path_to_logs != None:
            f.write('Epoch\tTrain_loss\tValid_loss\tCER\tWER\tTime\n')
        print('Epoch   Train_loss   Valid_loss   CER   WER    Time    LR')
        print('-----   -----------  ----------   ---   ---    ----    ---')
    print('{:02d}       {:.2f}         {:.2f}       {:.2f}   {:.2f}   {:.2f}   {:.7f}'.format(\
        metrics['epoch'], metrics['train_loss'], metrics['loss'], metrics['cer'], \
        metrics['wer'], metrics['time'], metrics['lr']))
    if path_to_logs != None:
        f.write(str(metrics['epoch'])+'\t'+str(metrics['train_loss'])+'\t'+str(metrics['loss'])+'\t'+str(metrics['cer'])+'\t'+str(metrics['wer'])+'\t'+str(metrics['time'])+'\n')
        f.close()
        


