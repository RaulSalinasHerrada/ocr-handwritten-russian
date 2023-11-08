# OCR RAB

Project to process financial documents and extract the words from it

## Pipeline

* Receives request POST
* Loads image
* Preprocess image (greyscale, de-noising, etc.) given doc type
* OCR
* Postprocess text(context correction of characters)
* Returns JSON with ocr

## Request form

### body

Deploy the uvicorn (see deployment section) and send the `urlsafe_b64` encoded bytes of the image on the `request.body`. 

### Headers

#### Language

Use the header `language` for setting the language of the model

* `eng`: English
* `est`: Estonian
* `rus`: Russian

#### Document type

Use the header `doc_type` to set the document type of the picture. This allows to choose better the preprocessing of the image

* `scanned` for high-quality scanned documents
* `photo` for pictures or **low-quality scanned documents**
* `handwritten` for handwritten docs

## Response form

JSON response with

* `OCR`: text given from the image
* `doc_type`: type of document
* `language`: language used

## Deployment

### Local Deployment using Docker

Run this code on `./app`

```
sudo docker build . -t ocr-rab:latest
sudo docker run -p 8000:8000 --name ocr-api ocr-rab:latest
```

### Local Deployment using Docker compose

Run this code on `./app`

```
sudo docker compose up
```


### Local Deployment (not recommended)

requirements: `python >= 3.10`. Takes 15 minutes to download every package

```
python3.10 -m venv venv_ocr_rab
source venv_ocr_rab/bin/activate
pip install -r requirements.txt 
uvicorn main:app --reload
```

## Usage Example after deployment

### python

```
import base64
import requests
import os

path_to_image = "./images/Halb_est.jpg"

with open(path_to_image, "rb") as image_file:
    bytes_image = base64.urlsafe_b64encode(image_file.read())

def extract_language_from_name(name:str):
    if "est" in name:
        return "est"
    elif "rus" in name:
        return "rus"
    else:
        return "eng"

res = requests.post(
    "http://0.0.0.0:8000/ocr",
    data=bytes_image,
    headers={"language": extract_language_from_name(path_to_image)})

print(res.json().get("OCR"))
```

### bash

```
python request.py --image Keskmine_eng_1 --doc_type scanned
python request.py --all
cat results.txt
```
