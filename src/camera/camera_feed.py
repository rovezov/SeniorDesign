# Camera Feed Module
# Connects to camera and streams frames


import cv2
import csv
from pathlib import Path
import time

from datetime import datetime

'''

note: this code should really be moved in the same file that handles safety violation detection since that
one needs to also save information for every instance of safety violation

'''

#Daniel: this is just a modification of Ben's risk reduction code. you can cut whatever code is not needed
#in final project, but you may need the code that helps store locally on rubikPi

def get_camera_feed():
    

    cap = cv2.VideoCapture(0) # Open the camera
    last_capture_time = 0
    demoFolder=Path("storage_Demo") #creates folder named in (), left side=Path object name you use object for several functions
    demoFolder.mkdir(exist_ok=True) #creates directory if it doesn't exist. does nothing if it exists

    #dummy data served as point of sending extra data alongside images
    location="Location=-1"
    warningType="PPE Missing=[X,X,X,X,X]"
    workerName="Responsible Worker= John Smith"
    confidenceNum="Confidence= XX.XX%" 

    
    data = [
    {'location': 'not_enabled', 'warning type': '-1', 'worker name': 'unnamed', 'timestamp': 111},
    ]

    while True:
        current_time = time.time() # Get the current time
        ret, frame = cap.read() # Read the captured frame
        #side note: to my understanding, frame is not like a JPG or PNG. it is an image, but it is a numpy array  
        if((current_time - last_capture_time) >= 2): # Executes once every ___ seconds
            last_capture_time = current_time
            
            current_datetime = datetime.now() # Get the current local date and time as a datetime object
            timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


            #note: cv2 also has an .imread function that can display image files

            cv2.imshow('Original', frame) # show the current frame. proof of camera's functionality
            #cv2.imwrite(demoFolder/f"Frame_{current_time}.jpg", frame) #<--only necessary for external data
            cv2.imwrite(demoFolder/f"Frame_{timestamp_str}.jpg", frame)
            #append extra information

            extraFilename=f"{timestamp_str}.csv"
            extraFile_directory=demoFolder/extraFilename
            
            data[0]['timestamp']=timestamp_str #it should in theory always update the timestamp per entry in a csv file

            with open(extraFile_directory, 'a', newline='') as csvfile: #changed w to a to append data per csv file which represents a day
                fieldnames = ['location', 'warning type', 'worker name', 'confidence', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if csvfile.tell()==0: #checks if csv file is empty, writes the header only once(at least in theory)
                    writer.writeheader() 
                writer.writerows(data)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if cv2.waitKey(1) & 0xFF==ord('z'): //could send other information and convert into zip file

    cap.release() # Release the camera and close all windows

'''
Notes of Changes

-Added support to contain external data in its own directory

-each csv file represents a day. each csv file logs all safety violations of that day

-this code demonstrates functionality of saving data but has no support being able to take information
from other modules, only saving dummy data for now

'''
