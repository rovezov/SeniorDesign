import cv2
import csv
from pathlib import Path
import time
import os
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

    #pulls from text file to find unique ID to attach to image and row of data. increments ID back to text file
    idFilename=f"uniqueID.txt"
    id_directory=demoFolder/idFilename
    currentID=0
 
    if os.path.exists(id_directory):
        with open(id_directory, "r") as f:
            content = f.read().strip()
            currentID = int(content) + 1 if content else 1

    # Write the new ID back
    with open(id_directory, "w") as f:
        f.write(str(currentID))



    extraFilename=f"safety_violation_logs.csv"
    extraFile_directory=demoFolder/extraFilename
    entry[0]['timestamp']=timestamp_str
    entry[0]['uniqueID']=currentID


    cv2.imwrite(demoFolder/f"Frame_{currentID}.jpg", savedFrame)

    with open(extraFile_directory, 'a', newline='') as csvfile:
        fieldnames = ['uniqueID','location', 'worker name', 'confidence', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell()==0: 
            writer.writeheader() 
        writer.writerows(entry) #Incoming data MUST provide values that complies with columns

