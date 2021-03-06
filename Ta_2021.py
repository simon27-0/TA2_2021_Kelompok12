# USAGE
# python3 detect_mask_webcam.py

from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
import subprocess

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # package untuk memproses gambar input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model # package load model maskNet
from imutils.video import VideoStream
from email.mime.text import MIMEText
import numpy as np
import argparse
import imutils # package untuk metode pengubahan ukuran gambar
import time
import cv2 # import openCV untuk memanipulasi tampilan dan gambar
import os

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
j = sensor.get_object_1()


#def say(text_tts):
#    os.system('pico2wave -l en-US -w  say.wav "%s" ' % text_tts)
#    os.system("aplay say.wav")
#    os.system("rm say.wav")

def execute_unix(inputcommand):
   p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
   (output, err) = p.communicate()
   return output

# Function untuk inilisasi pembacaan code pemrosesan bingkai
# Fungsi ini mendeteksi wajah dan kemudian menerapkan pengklasifikasi masker wajah kami ke setiap ROI wajah
def detect_and_predict_mask(frame, faceNet, maskNet): # proses pembuatan bingkai, mendeteksi dimana wajah berada, mendeteksi masker dalam wajah
        # grab the dimensions of the frame and then construct a blob
        # from it
        # proses ini untuk mmembuat dimensi bingkai dan kemudian membuat blob dari dimensi tersebut
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = [] # menemukan wajah
        locs = []  # menemukan lokasi wajah
        preds = [] # menemukan masker di dalam wajah

        # loop over the detections
        for i in range(0, detections.shape[2]): # ini berfungsi untuk menyaring bagian deteksi wajah yang lemah
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # ini akan berfungsi untuk kotak pembatas frame dan memastikan frame tersebut tidak berada di luar batas gambar
                        (startX, startY, endX, endY) = box.astype("int")

                        # ensure the bounding boxes fall within the dimensions of
                        # the frame
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


                        # extract the face ROI, convert it from BGR to RGB channel
                        # ordering, resize it to 224x224, and preprocess it
                        # fungsi ini untuk extract ROI wajah dan pra pemrosesan, kemudian akan menambahkan hasil extract tersebut
                        face = frame[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # gambar yang dibaca oleh opencv merupakan urutannya warnanya adalah blue, green, red. itu harus diubah ke RGB jika ingin menggunakan fitur opencv
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)

                        # add the face and bounding boxes to their respective
                        # lists
                        faces.append(face)
                        locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        # condition ini memastikan untuk mendeteksi adanya wajah, jika tidak mendeteksi maka akan mengembalikan nilai 0
        if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32") # untuk mengubah wajah menjadi array NumPy floating point 32-bit.
                preds = maskNet.predict(faces, batch_size=32) # untuk menginisialisasi pendeteksian masker sehingga akan melakukan pendeteksian di dalam batch

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
# pada bagian script di bawah ini, system akan memuat model yang sudah di train untuk dibandingkan dengan wajah yang akan di deteksi secara real time
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
        frame = imutils.resize(frame, width=1500)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet) # ini fungsi untuk mendeteksi apakah menggunakan masker atau tidak

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                # Fungsi untuk membuat kotak pembatas wajah dan memprediksi masker
                (startX, startY, endX, endY) = box 
                (mask, withoutMask) = pred
                
                # HTML untuk menampilkan hasil deteksi temperature suhu pada frame
                body = MIMEText('''
                <html>

                        <body>
                          <h1>Alert</h1>
                          <h2>A new has Person entered the Premises</h2>
                          <h2>Body Temperature: {}</h2>
                </body>
                </html>'''.format(j), 'html', 'utf-8')
               
                # determine the class label and color we'll use to draw
                # the bounding box and text
                if mask > withoutMask:
                        label = "Terima kasih sudah menggunakan masker"
                        color = (0, 255, 0)
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        a="Terima kasih sudah menggunakan masker"
                        c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a #mengeksekusi value dari "a"
                        execute_unix(c)
                        time.sleep(1)
                        
                        img_name = "database/"+ name +"/image_{}.png".format(img_counter)
                        pyautogui.screenshot("database/"+ name +"/image_{}.png".format(img_counter))
                        print("{} written!".format(img_name))
                        img_counter += 1

                        bus = SMBus(1) #memperkenalkan library mlx90614 kepada sistem
                        sensor = MLX90614(bus, address=0x5A) #memperkenalkan pin yang digunakan untuk mlx90614
                        j = sensor.get_object_1() #inisialisasi hasil deteksi temperature suhu menjadi "j"
                        cv2.putText(frame, 'Suhu Tubuh anda: {} C'.format(j), (50, 100), #memodifikasi tampilan menjadi berwarna RGB hijau
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                        if  j < 37 :
                                cv2.putText(frame, 'Suhu tubuh anda normal', (50, 200),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                                a="Suhu tubuh anda adalah"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a= (j)
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a="celcius"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a="anda dapat memasuki ruangan"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)

                                time.sleep(1)
                        else :
                                cv2.putText(frame, 'Suhu Tubuh anda tidak normal ', (50, 200),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 5)
                                a= "Suhu anda"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a= (j)
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a="celcius"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                a="Maaf anda tidak dapat memasuki ruangan"
                                c = 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                                execute_unix(c)
                                time.sleep(2.5)
                        bus.close()
                else :
                        label = "Anda belum menggunakan masker"
                        color = (0, 0, 255)
                        a= "Anda belum menggunakan masker"
                        c= 'espeak -vid+f3 -k5 -s130 --punct="<characters>" "%s" 2>>/dev/null' % a
                        execute_unix(c)
                        time.sleep(2.5)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX-50, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        # show the output frame
        cv2.imshow("Face Mask Detector", frame) #Fitur opencv untuk mengeluarkan hasil program terhadap frame
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

