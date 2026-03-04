#given relevant data for a safety violation and the camera snapshot of the incident, saves the screenshots as jpgs.
#Logs ALL incidents under a single csv file
#creates directory and files if they do not exist 

#If you want to see the data yourself, run hdfReader.py
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
#import cv2 #debug only



class Violation_Entry:


    def __add_entry(self, savedFrame):
    #def __add_entry(self, savedFrame, bodyCrop, faceCrop): #uncomment when ready to include other images
        #cv2.imshow('Original', savedFrame) #Debug only.
     
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        with h5py.File(self.h5_path, "a") as f:

            #retrieves attribute named next_ID, autoincrements
            #1. tries to fetch an attribute called "next_id", returns 0 if attribute does not exist
            #2. writes back to attribute "next_id" as incremented
            currentID= f.attrs.get("next_id", 0) 
            f.attrs["next_id"] =currentID + 1

            #If dataset 'logs' does not exist, create dataset.
            if "logs" not in f:
                dt = np.dtype([
                    ("uniqueID", "i8"), #integer type
                    ("location", h5py.string_dtype()),
                    ("worker_name", h5py.string_dtype()),
                    ("warning_type", h5py.string_dtype() ),
                    ("timestamp", h5py.string_dtype()),
                ])
                f.create_dataset("logs", shape=(0,), maxshape=(None,), dtype=dt) #dataset has no maximum shape/limit

            logs_ds = f["logs"]

            #Appends data via resizing and inserting data to last element
            logs_ds.resize((logs_ds.shape[0] + 1,))
            logs_ds[-1] = (
                currentID,
                self.location,
                self.workerName,
                self.warning,
                timestamp_str,
            )


            #each image-based group contains a dataset storing an ID and image
            camFrames_group = f.require_group("camera_frames")
            camFrames_group.create_dataset(
                str(currentID),
                data=savedFrame,
                compression="gzip" 
            )
            '''
            #re-enable for integration
            body_images= f.require_group("body_images")
            body_images.create_dataset(
                str(currentID),
                data=bodyCrop,
                compression="gzip"
            )

            face_images=f.require_group("body_images")
            face_images.create_dataset(
                str(currentID),
                data=faceCrop,
                compression="gzip" 
            )
            '''
    
    def __init__(self, savedFrame, location="unknown", workerName="unidentified", warning="not given"):
    #reenable 2nd initializer for saving other image types
    #def __init__(self, savedFrame, bodyCrop,faceCrop, location="unknown", workerName="unidentified", warning="not given")
        self.location=location
        self.workerName=workerName
        self.warning=warning
        
        self.demoFolder=Path("storage_Demo")
        self.demoFolder.mkdir(exist_ok=True)
        self.h5_path = self.demoFolder / "safety_logs.h5"
        self.__add_entry(savedFrame)
