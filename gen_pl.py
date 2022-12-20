import torch
from PIL import Image
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.nn.functional as F
import os
from dataset.densepass_train_dataset import densepassDataSet
from dataset.adaption.dp13_dataset import densepass13DataSet
from models.segformer.segformer import Seg
model1 = Seg(backbone='mit_nat_b2',num_classes=13,embedding_dim=512,pretrained=False)
model_path = '/models/ptmodel/b2_best_densepass.pth'
model1.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")),strict=False)
device = 'cuda'
model1.to(device)
train_root = './'
train_DensePASS = densepass13DataSet(train_root, list_path='/adaptations/dataset/densepass_list/train.txt',crop_size=(2048, 400))
train_loader = data.DataLoader(train_DensePASS, batch_size=1, shuffle=False, pin_memory=True)

interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)
predicted_label = np.zeros((len(train_DensePASS), 400, 2048), dtype=np.int8)
predicted_prob = np.zeros((len(train_DensePASS), 400, 2048), dtype=np.float16)
image_name = []

for index, batch in enumerate(train_loader):
    if index % 100 == 0:
        print('{}/{} processed'.format(index, len(train_loader)))

    image, _, name = batch
    image_name.append(name[0])
    image = image.to(device)
    b, c, h, w = image.shape
    output_temp = torch.zeros((b, 13, h, w), dtype=image.dtype).to(device)
    scales = [0.5,0.75,1.0,1.25,1.5,1.75]
    for sc in scales:
        new_h, new_w = int(sc * h), int(sc * w)
        img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
        with torch.no_grad():
            output, _ = model1(img_tem)
            output_temp += interp(output)
    output = output_temp / len(scales)
    output = F.softmax(output, dim=1)
    output = interp(output).cpu().data[0].numpy()
    output = output.transpose(1,2,0)
    
    label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
    predicted_label[index] = label
    predicted_prob[index] = prob
thres = []
for i in range(13):
    x = predicted_prob[predicted_label==i]
    if len(x) == 0:
        thres.append(0)
        continue        
    x = np.sort(x)
    thres.append(x[np.int(np.round(len(x)*0.5))])
print(thres)
thres = np.array(thres)
thres[thres>0.9]=0.9
print(thres)
for index in range(len(train_DensePASS)):
    name = image_name[index]
    label = predicted_label[index]
    prob = predicted_prob[index]
    for i in range(13):
        label[(prob<thres[i])*(label==i)] = 255  
    output = np.asarray(label, dtype=np.uint8)
    output = Image.fromarray(output)
    name = name.replace('.jpg', '_labelTrainIds.png')
    save_path = '/syn_pl'
    save_fn = os.path.join(save_path, name)
    if not os.path.exists(os.path.dirname(save_fn)):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    output.save(save_fn)
