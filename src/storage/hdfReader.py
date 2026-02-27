#required for debugging and testing phases. Need special way to read hdf5 files and 
import h5py
from pathlib import Path
import cv2

def get_logs(h5_path): #dataset logs returned as numpy nd arrays
    with h5py.File(h5_path, "r") as f:
        logs = f["logs"][:]
    print(type(logs))
    return logs

def print_all_logs(h5_path):
    with h5py.File(h5_path, "r") as f:
        logs = f["logs"][:] #<--gives all data for dataset named 'logs'

        #row['name'] MUST be identical to column names in the 'logs' dataset
        for row in logs:
            print(
                f"ID: {row['uniqueID']} | " 
                f"Location: {row['location']} | "
                f"Worker: {row['worker_name']} | "
                f"Confidence: {row['warning_type']} | "
                f"Time: {row['timestamp']}"
            )

def show_entry(h5_path, uid): #takes entry from both datasets 'logs' and 'images' corresponding to the same ID
    with h5py.File(h5_path, "r") as f:
        log = f["logs"][uid]
        image = f["camera_frames"][str(uid)][:]

    print("===============      Showing Specific Entry      =====================")
    print(log)
    cv2.imshow("Frame", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_all_frames(h5_path,delay=500): #only shows whole frames from the camera
    with h5py.File(h5_path, "r") as f:
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


#uncomment lines for specific data you want to verify
if __name__ == "__main__":

    #get_logs("storage_demo/safety_logs.h5")
    print_all_logs("storage_demo/safety_logs.h5")
    show_all_frames("storage_demo/safety_logs.h5")

    show_entry("storage_demo/safety_logs.h5", 7)

    pass