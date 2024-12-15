import os
import torch
from torchvision import models, transforms
from PIL import Image
import json
import io
import torch.nn as nn
import base64

def model_fn(model_dir):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        request_body = json.loads(request_body)
        image_data = base64.b64decode(request_body['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        transform = transforms.ToTensor()
        image = transform(image).unsqueeze(0)
        return image
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data) 
        _, predicted = torch.max(outputs.data, 1)
    return predicted.numpy().tolist()

def output_fn(prediction, accept='application/json'):
    class_names = [
        b'apple', b'aquarium_fish', b'baby', b'bear', b'beaver', b'bed', b'bee', b'beetle', b'bicycle', b'bottle',
        b'bowl', b'boy', b'bridge', b'bus', b'butterfly', b'camel', b'can', b'castle', b'caterpillar', b'cattle',
        b'chair', b'chimpanzee', b'clock', b'cloud', b'cockroach', b'couch', b'crab', b'crocodile', b'cup', b'dinosaur',
        b'dolphin', b'elephant', b'flatfish', b'forest', b'fox', b'girl', b'hamster', b'house', b'kangaroo', b'keyboard',
        b'lamp', b'lawn_mower', b'leopard', b'lion', b'lizard', b'lobster', b'man', b'maple_tree', b'motorcycle', b'mountain',
        b'mouse', b'mushroom', b'oak_tree', b'orange', b'orchid', b'otter', b'palm_tree', b'pear', b'pickup_truck', b'pine_tree',
        b'plain', b'plate', b'poppy', b'porcupine', b'possum', b'rabbit', b'raccoon', b'ray', b'road', b'rocket', b'rose',
        b'sea', b'seal', b'shark', b'shrew', b'skunk', b'skyscraper', b'snail', b'snake', b'spider', b'squirrel', b'streetcar',
        b'sunflower', b'sweet_pepper', b'table', b'tank', b'telephone', b'television', b'tiger', b'tractor', b'train',
        b'trout', b'tulip', b'turtle', b'wardrobe', b'whale', b'willow_tree', b'wolf', b'woman', b'worm'
    ]
    predicted_class_index = prediction[0] if isinstance(prediction, list) else prediction
    predicted_class_name = class_names[predicted_class_index].decode('utf-8')
    result = {"prediction": predicted_class_name,"prediction_raw":prediction}
    return json.dumps(result)
