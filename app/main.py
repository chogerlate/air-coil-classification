from fastapi import FastAPI,  Response
from pydantic import BaseModel
from PIL import Image
from fastapi import Response
import json
import requests
import uvicorn
import io

# IMPORT PYTORCH LIGHTNING LIBRARY APIs
import lightning.pytorch as pl
import timm
import torch.nn as nn
import torch
from torchvision.transforms import v2



app = FastAPI()

# identify device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' 

# the validation v2
transform = v2.Compose([
    v2.Resize((int)(448 * (256/224))),
    v2.CenterCrop(448),  # Adjust size as needed
    v2.RandomAutocontrast(p=1),
    v2.Grayscale(3),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

class eva02(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes
        ):
        super(eva02, self).__init__()
        self.num_classes = num_classes  # Assign num_classes to self.num_classes

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)  # Use num_classes parameter directly

        num_in_features = self.model.head.in_features  # get number of penultimate layer's output
        self.model.head = nn.Sequential(
                    nn.Dropout(0.25),  # Move dropout before batch norm for potential regularization
                    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(in_features=512, out_features=num_classes, bias=False)  # Use num_classes parameter directly
                    )  # modify classify's head

        self.save_hyperparameters()  # tell the model to save the hyperparameters into the checkpoint file



    def forward(self, x):
        # batch -> ([list of images],[list of targets])
        # this forward method can be refer to model(x)
        # print("\nbatch \n", batch)

        logits = self.model(x)
        predictions = torch.nn.functional.softmax(logits, dim=1)  # Apply softmax
        predictions = predictions.argmax(dim=1)  # Get class indices
        # print(x)
        return predictions



# Load model and set device
model_path = './bd_eva02_1_acc=0.77.ckpt'
air_coil_model = eva02.load_from_checkpoint(model_path).to(device).eval()


class image_payload(BaseModel):
    urls: list[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}


# Placeholder function for downloading and opening an image
def download_and_open_image(url):
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None
     

@app.post("/predict")
def prediction(payload: image_payload) :
    # initialize results

    results =  [] 
    for url in payload.urls:
        # download image from url 
        image = download_and_open_image(url)
        if not image: # if image is not downloaded
            return Response(json.dumps({"error": f"Failed to download image from url: {url}"}), media_type='application/json', status_code=400)
        image = transform(image).unsqueeze(0).to(device)
        pred = air_coil_model(image)
        print(pred)
        results.append(pred.item())
    
    return {"predictions": results}


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)



