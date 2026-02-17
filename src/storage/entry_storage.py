import cv2
import csv
from pathlib import Path
import time

from datetime import datetime

def store_entry(entry, savedFrame):
    

    demoFolder=Path("storage_Demo") #creates folder named in (), left side=Path object name you use object for several functions
    demoFolder.mkdir(exist_ok=True) #creates directory if it doesn't exist. does nothing if it exists

    current_datetime = datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    cv2.imshow('Original', savedFrame) # show the current frame. proof of camera's functionality
    cv2.imwrite(demoFolder/f"Frame_{timestamp_str}.jpg", savedFrame)

    day_str=current_datetime.strftime("%Y-%m-%d")
    hours_str=current_datetime.strftime("%H:%M:%S")
    extraFilename=f"{day_str}.csv"
    extraFile_directory=demoFolder/extraFilename
            
    entry[0]['timestamp']=hours_str #it should in theory always update the timestamp per entry in a csv file

    with open(extraFile_directory, 'a', newline='') as csvfile: #changed w to a to append data per csv file which represents a day
        fieldnames = ['location', 'warning type', 'worker name', 'confidence', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell()==0: #checks if csv file is empty, writes the header only once(at least in theory)
            writer.writeheader() 
        writer.writerows(entry) #Incoming data MUST provide values that make sense in columns
