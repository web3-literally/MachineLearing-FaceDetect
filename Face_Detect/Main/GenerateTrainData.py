import os
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import pickle

class TrainDataMaker():
    def __init__(self):
        self.face_encoding_list = []
        self.face_names = []
    def generateTrainData(self, DirPath):
        self.face_encoding_list = []
        self.face_names = []
        for f_name in os.listdir(DirPath):
            if f_name.endswith('.jpg') | f_name.endswith('.jPG') | f_name.endswith('.JPG') | f_name.endswith('.jpeg') | f_name.endswith('.JPEG') | f_name.endswith('.png') | f_name.endswith('.PNG'):
                strName = f_name[:-(len(f_name.split(".")[-1])+1)]
                person_image = face_recognition.load_image_file(DirPath + "//" + f_name)
                try:
                    person_face_encoding = face_recognition.face_encodings(person_image)[0]
                    self.face_encoding_list.append(person_face_encoding)
                    self.face_names.append(strName)
                except:
                    print(f_name + ": Can't Find Face!")
        with open('face_model.dat', 'wb') as f:
            pickle.dump([self.face_encoding_list, self.face_names], f)


