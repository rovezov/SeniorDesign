#creates objects for reading hdf5 data
import h5py
import numpy as np
from pathlib import Path

import cv2 #used if function to display images are necessary

class hdfReader:
    
    def __init__(self, folderName="storage_Demo",fileName="safety_logs.h5"):

        self.demoFolder=Path(folderName)
        self.demoFolder.mkdir(exist_ok=True)
        self.h5_path = self.demoFolder / fileName

    def get_logs(self): #accesses dataset 'logs', returns data as numpy nd arrays
        with h5py.File(self.h5_path, "r") as f:
            logs = f["logs"][:]
        return logs
        

    def print_all_logs(self): #accesses dataset 'logs' and prints all entires. does NOT return anything
        with h5py.File(self.h5_path, "r") as f:
            logs = f["logs"][:] 
            #row['name'] MUST be identical to column names in the 'logs' dataset
            for row in logs:
                print(
                    f"ID: {row['uniqueID']} | " 
                    f"Location: {row['location']} | "
                    f"Worker: {row['worker_name']} | "
                    f"Confidence: {row['warning_type']} | "
                    f"Time: {row['timestamp']}"
                )
            

    def get_all_cam_frames(self): #accesses group camera frames, returns dictionary where key=ID, value=image stored as numpy nd array
        cam_img_dict={}
        with h5py.File(self.h5_path, "r") as f:
            images_group = f["camera_frames"]
            ids = sorted(images_group.keys(), key=lambda x: int(x)) #.keys() returns names of datasets. IDs are embedded in dataset names
            for uid in ids:
                img = images_group[uid][:]
                cam_img_dict[uid]=img

        return cam_img_dict

    def show_all_cam_frames(self,delay=500): #shows all camera frames, specify delay in ms
        with h5py.File(self.h5_path, "r") as f:
            if "camera_frames" not in f:
                print("Group named camera_frames not found.")
                return

            images_group = f["camera_frames"]
            # Sort IDs numerically
            ids = sorted(images_group.keys(), key=lambda x: int(x))
            for uid in ids:
                img = images_group[uid][:]

                cv2.imshow("Saved Frames", img)
                key = cv2.waitKey(delay)
                # Press ESC to exit early
                if key == 27:
                    break

        cv2.destroyAllWindows()


    def show_entry(self, uid): #opens file, shows log and camera frame given ID number uid
        with h5py.File(self.h5_path, "r") as f:
            log = f["logs"][uid]
            image = f["camera_frames"][str(uid)][:]

        print("===============      Showing Specific Entry      =====================")
        print(log)
        cv2.imshow("Frame", image)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


    
