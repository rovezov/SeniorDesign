Functions of entry_storage.py:
        When called, this function will save a screenshot from the camera that contains a safety violation.
        It will also save a csv file logging each instance of safety violations for a certain day.
        The function creates a directory to save entries if it does not exist

        As of now, integration would need configuring the csv column names in entry_storage.py and data passed into it to make sense

Calling store_entry(entry, savedFrame)
        "entry": a list of dictionaries. For example: let entry= [{'location': 'not_enabled', 'warning type': '-1', 'worker name': 'unnamed', 'confidence':'-1', 'timestamp': 111},]

        "savedFrame": must be a cv2 object created from using cv2.imread(). 
                for example: frame=cap.imread()