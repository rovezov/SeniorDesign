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
    
    def __init__(self, location="unknown", workerName="unidentified", warning="not given"):
        self.location=location
        self.workerName=workerName
        self.warning=warning


    def add_entry(self, savedFrame):
    #def add_entry(self, savedFrame, bodyCrop, faceCrop):
        #cv2.imshow('Original', savedFrame) #Debug only.
     
        demoFolder = Path("storage_Demo")
        demoFolder.mkdir(exist_ok=True)

        h5_path = demoFolder / "safety_logs.h5"
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        with h5py.File(h5_path, "a") as f:
            next_id = f.attrs.get("next_id", 0) #retrieves attribute named next_ID, autoincrements
            f.attrs["next_id"] = next_id + 1
            currentID = next_id

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
                compression="gzip" #<--tells how to compress images
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
