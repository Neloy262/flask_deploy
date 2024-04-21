from flask import Flask,jsonify,request,Response
import firebase_admin
from firebase_admin import credentials, storage,firestore
from firebase_functions import https_fn
from firebase_admin import initialize_app
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import cv2
import datetime
from datetime import timedelta
import json


names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def draw_bbox(img,bbox_list,classes=None):
    color = (0,255,0)
    thickness = 2
    

    for x,bbox in enumerate(bbox_list):
        
        width = abs(int(bbox[0])-int(bbox[2]))
        height = abs(int(bbox[1])-int(bbox[3]))
        s1 = (int(bbox[0]),int(bbox[1]))
        e1 = (s1[0]+width//3,s1[1])

        s2 = (e1[0]+width//3,s1[1])
        e2 = (s2[0]+width//3,s1[1])

        e3 = (s1[0],s1[1]+height//3)

        s4 = (s1[0],e3[1]+height//3)
        e4 = (s1[0],s4[1]+height//3)
        
        cv2.putText(img, names[classes[x]], (s1[0],(s1[1]-height//10)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1)
        
        cv2.line(img, (s1[0],s1[1]-height//10), (e1[0],e1[1]-height//10), color, thickness)
        cv2.line(img, (s1[0]+10,s1[1]-height//10), (s1[0]+10,s1[1]), color, thickness)
        
        cv2.line(img, s1, e1, color, thickness) 
        cv2.line(img, s2, e2, color, thickness) 
        cv2.line(img, s1, e3, color, thickness)
        cv2.line(img, s4, e4, color, thickness)
        cv2.line(img, (s1[0],s1[1]+height),(e1[0],e1[1]+height) , color, thickness)
        cv2.line(img, (s2[0],s2[1]+height),(e2[0],e2[1]+height) , color, thickness)
        cv2.line(img, (s1[0]+width,s1[1]), (e3[0]+width,e3[1]), color, thickness)
        cv2.line(img, (s4[0]+width,s4[1]), (e4[0]+width,e4[1]), color, thickness) 

    return img


def overlay(overlayed_img,bbox_list,img,classes=None):
    color = (0,255,0)
    thickness = 2

    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2HSV)
    cutout_list = []
    for x,bbox in enumerate(bbox_list):

        width = abs(int(bbox[0])-int(bbox[2]))
        height = abs(int(bbox[1])-int(bbox[3]))
        


        overlayed_img[int(bbox[1]):int(bbox[1])+height,int(bbox[0]):int(bbox[0])+width,2] = img[int(bbox[1]):int(bbox[1])+height,int(bbox[0]):int(bbox[0])+width,2]
    
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_HSV2BGR)
 

    return overlayed_img

def base64_to_image(base64_string):
    # Decode the base64 string into bytes
    img_data = base64.b64decode(base64_string)
    
    # Open the image using PIL
    img = Image.open(BytesIO(img_data))
    
    return np.array(img)

cred = credentials.Certificate("productsense_fringe.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'productsense-b7b3c.appspot.com'})

db = firestore.client()

task_ref = db.collection("Tasks").document("03eK9VYUmTDqCwG4sRnJ")


model = YOLO("yolov8n.onnx")

# Create a Flask app
app = Flask(__name__)

# Define a route for the root URL '/'
@app.route('/',methods=['GET','POST'])
def hello_world():
    try:
        json_body = request.json

        org_image = base64_to_image(json_body["base64"])

        PAD = [0,0,0]
        org_image= cv2.copyMakeBorder(org_image.copy(),70,70,70,70,cv2.BORDER_CONSTANT,value=PAD)

        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
        image[:,:,2] = image[:,:,2] * 0.4
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        results = model.predict(org_image,imgsz=640,conf=0.3)
     
        image = overlay(image,results[0].boxes.xyxy.tolist(),org_image)
        image = draw_bbox(image,results[0].boxes.xyxy.tolist(),results[0].boxes.cls.tolist())

        image_data = Image.fromarray(image)

        # Create a BytesIO object to store the JPEG data
        jpeg_bytes = BytesIO()

        # Save the PIL Image as JPEG to the BytesIO object
        image_data.save(jpeg_bytes, format='JPEG')

        # Get the JPEG bytes from the BytesIO object
        jpeg_bytes = jpeg_bytes.getvalue()

        
        bucket = storage.bucket() # storage bucket
        blob = bucket.blob(json_body["filename"])
        blob.upload_from_string(jpeg_bytes,content_type="image/jpeg")
        today = datetime.datetime.now()

    # Calculate tomorrow's date
        expires = today + timedelta(days=1)
        
        signed_url = bucket.blob(json_body["filename"]).generate_signed_url(expiration=expires)
        res_data = {"url":signed_url}
        return jsonify(res_data),200
    except Exception as e:
        print(e)
        return str(e),500

@app.route('/addTask',methods=['GET','POST'])
def addTasks():
    try:
        json_body = request.json
        # shop = 
        task_ref.update({"Shops": firestore.ArrayUnion([json_body["id"]])})
        return "Added Task",200
    except Exception as e:
        return str(e),500

# Run the app if this script is executed
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)
