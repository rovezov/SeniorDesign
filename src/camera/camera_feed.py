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
    pass #TODO