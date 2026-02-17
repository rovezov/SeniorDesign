Functions of entry_storage.py:
        When called, this function will save a screenshot from the camera that contains a safety violation.
        It will also save a csv file logging each instance of safety violations for a certain day.
        The function creates a directory to save entries if it does not exist

        As of now, integration would need configuring the csv column names in entry_storage.py and data passed into it to make sense