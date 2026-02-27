import cv2
import time
#from entry_storage import store_entry


'''
changes:
1. remove 'timestamp', readjust code to account for this
2. make the code object oriented...?

'''

from entry_storage import Violation_Entry

def run_demo(): #purpose, demonstrate a "main" module calling on entry_storage whenever necessary



    cap = cv2.VideoCapture(0) # Open the camera
    last_capture_time = 0
    


    
    while True:
        current_time = time.time() # Get the current time
        ret, frame = cap.read() # Read the captured frame
        if((current_time - last_capture_time) >= 2): # Executes once every ___ seconds
            last_capture_time = current_time

            #store_call=Violation_Entry("place 1","bob","demo warning") #tests calls to entry_storage with filled inputs
            store_call=Violation_Entry()
            store_call.add_entry(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release() # Release the camera and close all windows
if __name__ == "__main__":
    run_demo()

