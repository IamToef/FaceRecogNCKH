import threading
import os
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from datetime import datetime
import locale
import keyboard
import torch

# Đặt lại môi trường locale để hỗ trợ Unicode
locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')

def process_image(img: np.ndarray):
    loaded_data = torch.load('face_data_new1.pt')

    # Giải nén các biến từ từ điển
    encoded_face_train_img = loaded_data['encoded_face_train_img']
    classNames = loaded_data['classNames']
    classNames1 = loaded_data['classNames1']
    classInfo = loaded_data['classInfo']

    print("Data loaded successfully from 'face_data_new1.pt'.")

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    num_detected_faces = 0
    info = "Không nhận diện được khuôn mặt nào."
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train_img, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train_img, encode_face)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            name1 = classNames1[matchIndex].upper()

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Sử dụng ký tự ASCII trong chuỗi thời gian
            current_time = datetime.now().strftime("- Date %d-%m-%Y \n - Time %H:%M:%S")
            info = classInfo.get(classNames1[matchIndex], "Thông tin không có sẵn.")
            num_detected_faces += 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, info
