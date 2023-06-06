from app import app
from app.dbmodel.ageitgeyUser import AgeitgeyUser
from app.dbmodel.facenetUser import FacenetUser
from flask import Flask, request, jsonify
from numpy import linalg, array
from PIL import Image
from datetime import datetime
import numpy as np
import io
import cv2
import base64
import time
import sys
import importlib
from app.controller import userController
from app.controller import facenetController




@app.route('/status')
def index():
    return jsonify({'status':'ok', 'message':'API is ready to use'})

@app.route('/facenet', methods=['POST'])
def facenet_verification():   
    try:
        t0 = time.time()
        # Menerima data dari frontend
        input = request.get_json(force=True)
        img_input = input['img'].replace('data:image/png;base64,','')    
        
        # Membuat vektor dan koordintar bounding box
        vector_input, xyxy, det_time, ext_time = facenetController.detect(source=img_input)
        if vector_input == 'undetected':
            return jsonify({'status' : 'failed', 'message': 'wajah tidak terdeteksi'})
        # Menggambar bounding box
        im0s = array(Image.open(io.BytesIO(base64.b64decode(img_input))))
        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0s, c1, c2, (255,0,0), lineType=cv2.LINE_AA)  

        # Konversi gambar ke byte string
        _, im_arr = cv2.imencode('.jpg', im0s) 
        im_bytes = im_arr.tobytes()
        im_output_b64 = base64.b64encode(im_bytes)
        im_output_b64_str = im_output_b64.decode("utf-8")     

        # Mencari identitas yang sesuai
        min_dist=0.8
        t_clf_0 = time.time()
        identity= userController.verify_from_db(FacenetUser, min_dist, vector_input)
    
        if identity == input['nama']:
            details = userController.get_all_details(FacenetUser, identity)
            t1 = time.time()
            det_time = round(1000*det_time,2)
            ext_time = round(1000*ext_time,2)
            clf_time = round(1000*(t1-t_clf_0),2)
            e2e_time = round(1000*(t1-t0),2)
            response = {
                'status' : 'ok',
                'nama' : details.full_name,
                'img' : im_output_b64_str,
                'det_time': det_time,
                'ext_time': ext_time,
                'clf_time': clf_time,
                'e2e_time': e2e_time
            }
            return jsonify(response)
        elif identity=='unknown09123':
            return jsonify({'status' : 'failed', 'message':'Anda tidak terdaftar'})
        else:
            return jsonify({'status' : 'failed', 'message':'Nama yang anda masukkan salah'})
    except Exception as e:
        return jsonify({"status":"failed","message": str(e)})

# @app.route('/verification/ageitgey', methods=['POST'])
# def ageitgey_verification():
    try:
        t0 = time.time()
        # Menerima data dari frontend
        input = request.get_json(force=True)
        img_input = input['img'].replace('data:image/png;base64,','')    
        
        # Membuat vektor dan koordintar bounding box
        vector_input, xyxy, det_time, ext_time = dlibController.signature(source=img_input)
        if vector_input == 'gagal':
            return jsonify({'status' : 'failed', 'message': 'wajah tidak terdeteksi'})

        # Menggambar bounding box
        im0s = array(Image.open(io.BytesIO(base64.b64decode(img_input))))
        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
        c1, c2 = (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3]))
        cv2.rectangle(im0s, c1, c2, (255,0,0), lineType=cv2.LINE_AA)  

        # Konversi gambar ke byte string
        _, im_arr = cv2.imencode('.jpg', im0s) 
        im_bytes = im_arr.tobytes()
        im_output_b64 = base64.b64encode(im_bytes)
        im_output_b64_str = im_output_b64.decode("utf-8")     
        
        # Mencari identitas yang sesuai
        min_dist=0.568
        t1 = time.time()
        identity= userController.verify_from_db(AgeitgeyUser, min_dist, vector_input)
        
        if identity==input['nama']:
            t2 = time.time()
            details = userController.get_all_details(AgeitgeyUser, identity)
            det_time = round(1000*det_time,2)
            ext_time = round(1000*ext_time,2)
            clf_time = round(1000*(t2-t1),2)
            e2e_time = round(1000*(t2-t0),2)
            response = {
                'status' : 'ok',
                'nama' : details.full_name,
                'img' : im_output_b64_str,
                'det_time': det_time,
                'ext_time': ext_time,
                'clf_time': clf_time,
                'e2e_time': e2e_time
            }
            return jsonify(response)
        elif identity=='unknown09123':
            return jsonify({'status' : 'failed', 'message':'Anda tidak terdaftar'})
        else:
            return jsonify({'status' : 'failed', 'message':'Nama yang anda masukkan salah'})
    
    except Exception as e:
        return jsonify({"status":"failed","message": str(e)})

@app.route('/importdata', methods = ['POST'])
def import_data():
    req_data = request.get_json(force=True)
    for key, val in req_data.items():
        # userController.save_to_db(AgeitgeyUser, key, val)
        userController.save_to_db(FacenetUser, key, val)
    return jsonify({'status':'ok'})
