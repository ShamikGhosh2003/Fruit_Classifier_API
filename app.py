from flask import Flask,request
from flask_cors import CORS
import PIL
import matplotlib.pyplot
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms,models

# Load saved model
def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(512, 425)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(425, 3)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.load_state_dict(ckpt, strict=False)

    return model

# load model
model = load_ckpt('res18_10.pth')   

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch Tensor
    '''
    im = PIL.Image.open(image)
    return test_transforms(im)

def predict(image_path, model):
    # Predict the class of an image using a trained deep learning model.
    model.eval()
    img_pros = process_image(image_path)
    img_pros = img_pros.view(1,3,224,224)
    with torch.no_grad():
        output = model(img_pros)
    return output


app=Flask(__name__)
cors=CORS(app,resources={r"/*":{"origins":"*"}})  #when frontend and backend are deployed in different machines, this error may pop up. Thi allows for cross sharing/resource sharing


@app.route('/',methods=['GET'])
def home():
    return '<h1>The API is not running (Ou Arif eisob dekchish??), change holo?</h1>'

@app.route('/mypred',methods=['POST'])
def output():
    img=request.files['img']
    log_ps = predict(img, model)
    cls_score = int(torch.argmax(torch.exp(log_ps)))
    if cls_score == 0:
        return 'Banana'
    elif cls_score == 1:
        return 'Mango'
    else:    
        return 'Strawberry'

if __name__=='__main__':     #will only run if it is called from this method
    app.run(host='0.0.0.0',port=8000, debug=True)      #we can give host='127.0.0.1' (this is for your own device) or '0.0.0.0' (accessiblr by LAN) 

#1:16:14
