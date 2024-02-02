# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from IPython.display import Image

#import required funtions from torchvision
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils

#create model
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

def create_model(num_classes=91, size=300):
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )
    # Retrieve the list of input channels.
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

if __name__ == '__main__':
    model = create_model(2, 640)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

# Load the best model and trained weights.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__','Diente-de-leon','Kikuyo','otros', 'papa','lengua de vaca'
]

NUM_CLASSES = len(CLASSES)
model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('/models/best_model.pth', map_location=DEVICE) # ruta de los pesos
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

#load de model to Autodetection
detection_model = AutoDetectionModel.from_pretrained(
    model_type='torchvision',
    model=model,
    confidence_threshold=0.35, #manejar score para filtrar
    category_mapping = {"1": "Diente-de-leon", "2": "Kikuyo","3": "otros","4": "papa","5": "lengua de vaca"},
    image_size=640,
    device="cpu", # or "cuda:0"
    load_at_init=True,
)

#predecir en imagenes
result = get_sliced_prediction(
    "/DJI_0922_JPG.rf.03fb1a894520dc166d6ee453922af9eb.jpg",
    detection_model,
    slice_height = 56, #variar alto de la ventana
    slice_width = 56, #variar ancho de la ventana
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)

#Guardar y mostrar resultado
result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")

#Mostrar con openCV
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('/content/demo_data/prediction_visual.png')
#import numpy as np
#img = np.read_image()
plt.imshow(img)
plt.show()
