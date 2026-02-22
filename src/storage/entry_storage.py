import cv2
import csv
from pathlib import Path
import time

from datetime import datetime

#given relevant data for a safety violation and the camera snapshot of the incident, saves the screenshots as jpgs.
#Logs ALL incidents under a single csv file
#creates directory and files if they do not exist 
def store_entry(entry, savedFrame):
    

    demoFolder=Path("storage_Demo") 
    demoFolder.mkdir(exist_ok=True)

    current_datetime = datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    #cv2.imshow('Original', savedFrame) #Debug only. 
    cv2.imwrite(demoFolder/f"Frame_{timestamp_str}.jpg", savedFrame)


    extraFilename=f"safety_violation_logs.csv"
    extraFile_directory=demoFolder/extraFilename
            
    entry[0]['timestamp']=timestamp_str

    with open(extraFile_directory, 'a', newline='') as csvfile:
        fieldnames = ['location', 'worker name', 'confidence', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell()==0: 
            writer.writeheader() 
        writer.writerows(entry) #Incoming data MUST provide values that complies with columns
