#Import All the Required Libraries

# Load the OpenCV library so we can use all its functions
import cv2     

# From ultralytics library import YOLO10 model. A deep learning object detection model that detect the license plate
from ultralytics import YOLOv10


# This is a Python library used for calculations
import numpy as np

import math

# It is a module used to  clean text that are extracted from OCR.
import re

# Import OS module to interact with the operating system
import os

import json
from datetime import datetime
import sqlite3

# Python library used to extract text from images
from paddleocr import PaddleOCR


# Allow duplicate OpenMP libraries to prevent crashes. OpenMP enables multithreading for faster CPU execution.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Use OpenCV to process the video frame by frame
cap = cv2.VideoCapture("data/carLicence1.mp4")


# Load YOLO model weights from best.pt for license plate detection
model = YOLOv10("weights/best.pt")

# This Variable is used to count the number of the frame from the video
count = 0


# Create a list having element "License" and this basically used to label the detected plate as "License"
className = ["License"]


# This line is used to initialize the OCR model from the PaddleOCR library 
# use_angle_cls = True means allow model to detect from rotated image 
# use_gpu=False means use CPU 
ocr = PaddleOCR(use_angle_cls = True, use_gpu = False)


# Define a function to extract text from a given region of the frame
def paddle_ocr(frame, x1, y1, x2, y2):

    frame = frame[y1:y2, x1: x2]           # Crops the image using NumPy slicing:
    

    # Calls the PaddleOCR object that we created earlier .Here ocr will extract the text.
    # det=False: skip detection, we already have the region
    # rec=True: perform text recognition
    # cls=False: skip angle classification
    result = ocr.ocr(frame, det=False, rec = True, cls = False)   

    text = ""                              # This variable will hold our final result
    for r in result:
        scores = r[0][1]                   # This is basically what is confidence of model for this
        if np.isnan(scores):               # Check for NAN (Not a Number)
            scores = 0
        else:
            scores = int(scores * 100)     # Converted in percentage
        if scores > 60:                    # We only take that result where confidence is more than 60 
            text = r[0][0]
    pattern = re.compile('[\W]')           # Removes all non-alphanumeric characters from the text.


# Some additional replacement
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)




# Function to Save Detected License Plates to JSON Files
# This function saves the set of license plates detected during a 10-second interval
def save_json(license_plates, startTime, endTime):
    
    # Create a dictionary with the interval data

    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }

    # Save to a new uniquely named file in the json folder
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent = 2)

    # Path to the cumulative file
    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        # If file doesn't exist, start with empty list
        existing_data = []

    # Append current interval data to cumulative list
    existing_data.append(interval_data)

    # Save the updated cumulative data back to the same file
    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent = 2)

    #Save data to SQL database
    save_to_database(license_plates, startTime, endTime)


def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()

# Record the starting time of the 10-second interval.
startTime = datetime.now()

# Create an empty set to store unique license plate numbers detected during
license_plates = set()




# Main loop to process video frames
while True:

    # cap.read() read the next frame from video.
    # If it successful read then ret=true and we go ahead 
    # If ret=false means no frame left, we comes out of the loop 
    # frame will store the frame read from the video
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1                                              # Increase the frame number
        print(f"Frame Number: {count}")

        # Predict objects in the current frame with confidence > 45% 
        # Results stored in results
        results = model.predict(frame, conf = 0.45)            
        for result in results:
            boxes = result.boxes                                # Position of detected object (Bounded object) 
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]                    # Extract coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw a rectangle around the detected object in the video frame at extracted coordinates 
                # color is (255,0,0) and thickness is 2 px
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)    

                # box.cls[0] contains the class ID of the detected object  then converted it into integer value;
                classNameInt = int(box.cls[0])         
                              
                clsName = className[classNameInt]                     # Here we map this classId with class name that is "License"
                conf = math.ceil(box.conf[0]*100)/100                 # Here it calculate how much the model is confidence about detected object
               
                label = paddle_ocr(frame, x1, y1, x2, y2)             # Call the function to extract text from the frame

                if label:
                    license_plates.add(label)
                # Here we  calculate how much space the text will take when drawn.
                # This helps in drawing a background rectangle behind the text so it’s visible.
                # It return a tuple like this ((width, height), baseline) we require only width and height.
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]   


                # Here we simply do : c2=x1+width,y1=height-3(for padding purpose) 
                # Here c2 is (x2,y2) of that rectangle
                c2 = x1 + textSize[0], y1 - textSize[1] - 3    

                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)       #  Draw a rectangle behind the text to enhance visibility of text

                # It basically put text on the rectangle
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)    
        
        # Check if 10 seconds have passed since the last save
        if (currentTime - startTime).seconds >= 10:
            endTime = currentTime                              # Mark the end time of this interval

            # Save the collected license plates along with the start and end times to JSON
            save_json(license_plates, startTime, endTime)

            startTime = currentTime                            # Reset the timer for the next interval
            license_plates.clear()                             # Clear the set to start collecting new license plates

        cv2.imshow("Video", frame)                             # It basically show the updated frame in that video
        if cv2.waitKey(1) & 0xFF == ord('1'):                  #  If want to stop the video
            break
    else:                                                      # Break if no more frames
        break                                                  # If ret = false


    
cap.release()                                                  # These both lines are used to free up the resources
cv2.destroyAllWindows()
