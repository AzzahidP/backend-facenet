import time
from PIL import Image
from keras_facenet import FaceNet
from numpy import asarray, linalg, expand_dims, ascontiguousarray, array

import numpy as np
import cv2
import torch
import copy
import dlib
import base64
import io
import face_recognition_models

predictor_5 = face_recognition_models.pose_predictor_five_point_model_location()
sp = dlib.shape_predictor(predictor_5)
MyFaceNet_new = FaceNet()

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path
from utils.torch_utils import time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
    model='yolov5s-face.pt',
    device=torch.device("cpu"),
    source='data/images/',
    project='runs/detect',
    name='exp',
    exist_ok=False,
):
    model = load_model(model, device)
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)

    # Read image
    im0s = array(Image.open(io.BytesIO(base64.b64decode(source))))
    im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
    assert im0s is not None, 'Image Not Found '

    # Padded resize
    img = letterbox(im0s, imgsz, auto=False)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    im = ascontiguousarray(img) #convert ke contiguous
    
    if len(im.shape) == 4:
        orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
    else:
        orgimg = im.transpose(1, 2, 0)
    
    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert from w,h,c to c,w,h
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    t_det_0 = time.time()
    # Inference
    pred = model(img)[0]
    
    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        
        im0 = im0s.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()
            t_det_1 = time.time()
            for j in range(det.size()[0]):
                
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                
                im0 = show_results(im0, xyxy, conf, landmarks, class_num)
                
                xyxy1=[int(a) for a in xyxy]
                x1,x2,x3,x4=xyxy1[0],xyxy1[1],xyxy1[2],xyxy1[3]
                #Menentukan lokasi wajah dengan koordinat masukan
                t_ext_0 = time.time()
                face_location = dlib.rectangle(x1,x2,x3,x4)
                faces = dlib.full_object_detections()

                #Mendapatkan landmark dari setiap wajah
                faces.append(sp(im0s, face_location))

                #Menormalisasi bentuk wajah
                image = dlib.get_face_chip(im0s, faces[0])
                
                #Konversi BGR menjadi RGB
                gb1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #Konversi format gambar OpenCV menjadi PIL
                gb1 = Image.fromarray(gb1)                       
                gb1 = gb1.resize((160,160))
                gb1 = asarray(gb1)

                #face = expand_dims(face, axis=0) #Menambah dimensi
                gb1 = expand_dims(gb1, axis=0) #Menambah dimensi

                #Mengekstrak fitur wajah menjadi vektor dengen pre-trained model
                signature = MyFaceNet_new.embeddings(gb1)
                t_ext_1 = time.time()

                return signature,xyxy, (t_det_1-t_det_0), (t_ext_1-t_ext_0)
        
        else:
            return '0', [], 0, 0
    
                    
def distance(source1, source2):
    device = torch.device("cuda")
    model = load_model('yolov5s-face.pt', device)
    vector_1,_,aa,ee = detect(model, device,source1,view_img=True)
    vector_2,_,bb,yy = detect(model, device,source2,view_img=True)
    dist = linalg.norm(vector_1 - vector_2)
    return dist      
         