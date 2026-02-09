# Camera Feed Module
# Connects to camera and streams frames


import cv2
from pathlib import Path
import time

from datetime import datetime



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

            extraFilename=f"{timestamp_str}.txt"
            extraFile_directory=demoFolder/extraFilename
            
    
            with extraFile_directory.open("w") as file_handler:
                file_handler.write(location+"\n")
                file_handler.write(warningType+"\n")
                file_handler.write("proof of time"+timestamp_str)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if cv2.waitKey(1) & 0xFF==ord('z'): //could send other information and convert into zip file

    cap.release() # Release the camera and close all windows

'''
Notes of Changes

-Added Support to save to any kind of folder name
    Disclaimer: 
        only implemented such that the saved items are a folder below this code's current path
        it depends on where the main

        only a hypothesis, but I think the saving and eventual retrieval of external data depends on where
        the main python program is location compared to its sub-units

        if necessary, you will probably need something like
            "base=Path.cwd() <--current directory
            parent=base.parent <--returns parent directory

-Added a somewhat rudimentary way of saving information. generates and writes to new text files
each line containing some kind of external information

I *have* considered the possibility of ZIP files, but with the way information can be broken down into jpgs and 
text files containing everything else, I don't think compression is strictly necessary
    I am aware of risks that can arise of converting output into text files

For me, I still have questions about information storage.
    how does old data get periodically cleared? will this be unecessary when we're just going to upload to some kind of database?

Getting suggestions for .csv format, pickle?(Python thing he said)...

'''
