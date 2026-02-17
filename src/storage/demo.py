import cv2
import time
from entry_storage import store_entry


def run_demo(): #purpose, demonstrate a "main" module calling on entry_storage whenever necessary

    cap = cv2.VideoCapture(0) # Open the camera
    last_capture_time = 0
    dummyData= [{'location': 'not_enabled', 'warning type': '-1', 'worker name': 'unnamed', 'confidence':'-1', 'timestamp': 111},
    ]


    
    while True:
        current_time = time.time() # Get the current time
        ret, frame = cap.read() # Read the captured frame
        #side note: to my understanding, frame is not like a JPG or PNG. it is an image, but it is a numpy array  
        if((current_time - last_capture_time) >= 2): # Executes once every ___ seconds
            last_capture_time = current_time
            store_entry(dummyData,frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if cv2.waitKey(1) & 0xFF==ord('z'): //could send other information and convert into zip file

    cap.release() # Release the camera and close all windows
if __name__ == "__main__":
    run_demo()

