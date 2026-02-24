import cv2
import time
from entry_storage import store_entry


def run_demo(): #purpose, demonstrate a "main" module calling on entry_storage whenever necessary


    #1? Remove "Old data"


    cap = cv2.VideoCapture(0) # Open the camera
    last_capture_time = 0
    dummyData= [{'uniqueID':0,'location': 'not_enabled',  'worker name': 'unnamed', 'confidence':'-1', 'timestamp': 111},
    ]


    
    while True:
        current_time = time.time() # Get the current time
        ret, frame = cap.read() # Read the captured frame
        if((current_time - last_capture_time) >= 2): # Executes once every ___ seconds
            last_capture_time = current_time
            store_entry(dummyData,frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release() # Release the camera and close all windows
if __name__ == "__main__":
    run_demo()

