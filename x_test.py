import  base64
import numpy as np
import cv2
import requests
import os
import pickle

url = "http://localhost:8080/api/gethog"
def   img2vec(img):
      v,buffer = cv2.imencode(".jpg",img)
      img_str = base64.b64encode(buffer)
      data = "image data,"+str.split(str(img_str),"'")[1]
      response = requests.get(url,json={"img":data})

      return response.json()

img_list = []

path = 'app\\test'

carvectors = []
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path,sub)):
        img_file_name = os.path.join(path,sub)+"/"+fn
        img = cv2.imread(img_file_name)
        res = img2vec(img)
        vec = list(res["message"])
        vec.append(sub)
        carvectors.append(vec)

write_path = "carvectors_test.pkl"
pickle.dump(carvectors, open(write_path,"wb"))
print("data preparation is done")