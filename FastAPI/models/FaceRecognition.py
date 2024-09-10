import os
import cv2
import face_recognition
import numpy as np

class FaceRecognition:
    def __init__(self, dataset_path):
        self.images = []
        self.classNames = []
        self.classNames1 = []
        self.classInfo = {}
        self.load_dataset(dataset_path)
        self.encoded_face_train_img = self.encode_faces()

    def load_dataset(self, path):
        for root, dirs, files in os.walk(path):
            for cl in dirs:
                class_path = os.path.join(root, cl)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_name.endswith(('png', 'jpg', 'jpeg')):
                        curImg = cv2.imread(img_path)
                        if curImg is not None:
                            self.images.append(curImg)
                            txt_path = os.path.join(class_path, cl + '.txt')
                            if os.path.exists(txt_path):
                                with open(txt_path, 'r', encoding='utf-8') as file:
                                    content = file.read().strip().splitlines()
                                    name = content[0].strip()
                                    info = '\n'.join(content[1:]).strip()
                                    self.classNames.append(cl)
                                    self.classNames1.append(name)
                                    self.classInfo[name] = info

    def encode_faces(self):
        return [face_recognition.face_encodings(img)[0] for img in self.images if len(face_recognition.face_encodings(img)) > 0]

    def recognize_face(self, img):
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        recognized_info = []

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(self.encoded_face_train_img, encode_face)
            faceDist = face_recognition.face_distance(self.encoded_face_train_img, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                name = self.classNames[matchIndex].upper()
                name1 = self.classNames1[matchIndex].upper()
                info = self.classInfo.get(self.classNames1[matchIndex])
                recognized_info.append({"name": name, "name1": name1, "info": info})

        return recognized_info
