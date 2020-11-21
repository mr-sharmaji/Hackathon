import tkinter as tk
from tkinter import *
import cv2, os
# import mysql.connector
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import json

import cv2
import os
import requests
import pywhatkit as kit
from threading import Thread
import mysql.connector

mydb = mysql.connector.connect(host="localhost",user="root",passwd="root",database="attendify1")
mycursor = mydb.cursor()
def whatsapp(number,hour,min):
    #
    #data = pd.read_csv("Attendance\Attendance_" + date + ".csv")
    # data_dict = data.to_dict('list')
    # leads = data_dict['Number']
    # print (leads)
    #
    #
    #
    # text = 'hey how are you...'
    #
    # j = 0
    # for i in leads:
    #
    #
    #     mini = int(min) + 2 + j
    #     kit.sendwhatmsg("+91" + number, text, int(hour), mini)
    #     j = j + 1
    # #list

    kit.sendwhatmsg("+91" + number, "Hey your attendances has been recorded", int(hour), int(min) + 2)





root = Tk()

root.title("Iem AI Based Attendence System")

canvas = Canvas(root, width=2250, height=1280)
image = ImageTk.PhotoImage(Image.open("Images\\background.png"))

canvas.create_image(5, 5, anchor=NW, image=image)
canvas.pack()

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
id2 = tk.Label(root, text="* Additional Covid 19 Feature", font=('times', 15, ' bold '))
id2.place(x=80, y=390)

id1 = tk.Label(root, text="Enter ID", width=10, height=1, fg="black", bg="white", font=('times', 15, ' bold '))
id1.place(x=375, y=250)

txt1 = tk.Entry(root, width=30, bg="white", fg="black", font=('times', 12, '  '))
txt1.place(x=400, y=290)

name1 = tk.Label(root, text="Enter Name", width=10, height=1, fg="black", bg="white", font=('times', 15, ' bold '))
name1.place(x=390, y=330)

txt2 = tk.Entry(root, width=30, bg="white", fg="black", font=('times', 12, '  '))
txt2.place(x=400, y=370)

ph1 = tk.Label(root, text="Parent's Phone Number", width=20, height=1, fg="black", bg="white",
               font=('times', 15, 'bold'))
ph1.place(x=380, y=410)

txt3 = tk.Entry(root, width=30, bg="white", fg="black", font=('times', 12, '  '))
txt3.place(x=400, y=450)

message2 = tk.Label(root, text="", fg="black", bg="white", font=('times', 15, ' bold '))
message2.place(x=700, y=650)


def maskdetor():
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("w"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def clear():
    txt1.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear3():
    txt3.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (txt1.get())
    name = (txt2.get())
    phone_number = (txt3.get())


    if is_number(Id) and name.isalpha() and is_number(phone_number):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name + " Parent's Phone Number : " + phone_number
        row = [Id, name, phone_number]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
        sql = "Insert into tbl_people (id,firstname,emailaddress) values(%s,%s,%s)"
        val = (Id, name, phone_number)
        mycursor.execute(sql, val)
        sql1="insert into tbl_company_data (reference,company,department,jobposition,idno) values (%s,%s,%s,%s,%s)"
        val1=(Id,"Institute of Engineering & Management","Bca","Student",Id)
        mycursor.execute(sql1,val1)
        mydb.commit()
    else:
        # print(is_number(Id))
        # print(name.isalpha())
        # print(is_number(phone_number))
        if (is_number(Id) and name.isalpha()):
            res="Enter Phone Number"
            message.configure(text=res)

        elif is_number(Id) and is_number(phone_number):
                res="Enter Alphabetical Name"
                message.configure(text=res)
        elif (name.isalpha() and is_number(phone_number)):
                res="Enter Numeric Id"
                message.configure(text=res)
        elif is_number(Id):
            res="Enter Phone Number \n Enter Alphabetical Name"
            message.configure(text=res)
        elif name.isalpha():
            res="Enter Numeric Id \n Enter Phone Number"
            message.configure(text=res)
        elif is_number(phone_number):
            res="Enter Alphabetical Name \n Enter Numeric Id"
            message.configure(text=res)



def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in Id)
    message.configure(text=res)


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time','Phone Number']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            print(conf)
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                pp = df.loc[df['Id'] == Id]['Phone Number'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp,pp]



            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")

    print("My name is ", aa[0])
    print("My id is", Id)
    print("my phone is ",pp[0])
    number = str(pp[0])
    print(type(number))
    print(type(aa[0]))
    print(type(Id))
    # try:
    #     t = Thread(target=whatsapp(number,Hour,Minute))
    #     t.deamon=true
    #     t.start()
    # except:
    #     print("Unable to send text")


    #kit.sendwhatmsg("+91"+number,  "Hey your attendances has been recorded" , int(Hour) , int(Minute)+2)
   # sql = "INSERT INTO testyash (idno,employee) VALUES (%s,&s)"
    #val = ("100", "Yash Kanoria")
    # mycursor.execute(sql, val)
    #
    # mydb.commit()
    #
    # url = "https://attendify-api.herokuapp.com/"
    # response = requests.get(url)
    print("both", aa[0], Id)
    sql = "Insert into tbl_people_attendance(idno,date,employee,timein) values (%s,%s,%s,%s)"
    val=(10,date,aa[0],timeStamp)
    mycursor.execute(sql,val)

    mydb.commit()

    # payload = "{\r\n    \"rollNo\": \"1\",\r\n    \"name\": \"Yash Kanoria\"\r\n}"
    # PARAMS = {'rollNo': '12345' ,'name': 'test23456'}

    # headers = {
    #     'Content-Type': 'application/json'
    # }
    # PARAMS = {'rollNo': Id, 'name': aa[0]}
    # print(PARAMS)
    # response = requests.post(url, data=json.dumps(PARAMS), headers=headers)

    # print(response.text.encode('utf8'))
    fileName = "Attendance\Attendance_" + date + ".csv"


    # fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"

    # print("1 record inserted, ID:", mycursor.lastrowid)
    attendance.to_csv(fileName, mode='a', index=False)
    #attendance.to_csv(fileName, mode='a', index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)
    res = attendance
    message2.configure(text=res)


loadimageclear1 = tk.PhotoImage(file="Images\\Clear.png")
roundedbuttonclear1 = tk.Button(root, image=loadimageclear1, command=clear)
roundedbuttonclear1["bg"] = "white"
roundedbuttonclear1["border"] = "0"
roundedbuttonclear1.place(x=670, y=285)

roundedbuttonclear2 = tk.Button(root, image=loadimageclear1, command=clear2)
roundedbuttonclear2["bg"] = "white"
roundedbuttonclear2["border"] = "0"
roundedbuttonclear2.place(x=670, y=365)

roundedbuttonclear3 = tk.Button(root, image=loadimageclear1, command=clear3)
roundedbuttonclear3["bg"] = "white"
roundedbuttonclear3["border"] = "0"
roundedbuttonclear3.place(x=670, y=445)

notification = tk.Label(root, text="Notification : ", width=12, height=1, fg="black", bg="white",
                        font=('times', 15, ' bold '))
notification.place(x=390, y=490)

message = tk.Label(root, text=" ", width=30, height=2, fg="black", font=('times', 15, ' bold '))
message.place(x=540, y=490)

take = tk.PhotoImage(file="Images\\Take_Images.png")
take_images = tk.Button(root, image=take, command=TakeImages)
take_images["bg"] = "white"
take_images["border"] = "0"
take_images.place(x=380, y=550)

training = tk.PhotoImage(file="Images\\Train_Images.png")
train_images = tk.Button(root, image=training, command=TrainImages)
train_images["bg"] = "white"
train_images["border"] = "0"
train_images.place(x=530, y=550)

track = tk.PhotoImage(file="Images\\Take_Attendence.png")
track_images = tk.Button(root, image=track, command=TrackImages)
track_images["bg"] = "white"
track_images["border"] = "0"
track_images.place(x=680, y=550)

q = tk.PhotoImage(file="Images\\Quit.png")
qt = tk.Button(root, image=q, command=root.destroy)
qt["bg"] = "white"
qt["border"] = "0"
qt.place(x=830, y=550)

q1 = tk.PhotoImage(file="Images\\Detect_Mask.png")
qt1 = tk.Button(root, image=q1, command=maskdetor)
qt1["bg"] = "white"
qt1["border"] = "0"
qt1.place(x=130, y=320)

root.mainloop()
