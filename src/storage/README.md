REQUIRES installation of h5py. Can be achieved with "pip install h5py"

Functions of entry_storage.py:
        Saves text logs and images into an HDF5 (.h5) file.

        When called, this function will save a screenshot from the camera that contains a safety violation.
        It will also save a csv file logging each instance of safety violations for a certain day.
        The function creates a directory to save entries if it does not exist


Notes on the HDF5 format:
        The HDF5 format distinguishes 2 main types of objects:
                1. groups: holds datasets
                2. datasets: capable of storing tables, graphs, images and even documents

        supported arguments for  h5py.File( FILE_NAME, "?")
                r=readonly, file must exist to work
                r+= read and write, file must exist to work
                w=writeonly, truncates if it exists
                w- or x= fails to write if the file doesn't exist
                a= read or write to a file, creates the file if it doesn't exist

Current implementation of the text-only logs uses a variable called 'dt', representing data types. In the form of:
        dt = np.dtype([
                        ("uniqueID", "i8"),
                        ("location", h5py.string_dtype()),
                        ("worker_name", h5py.string_dtype()),
                        ("confidence", "f4"),
                        ("timestamp", h5py.string_dtype()),
                ])

        each element of this array must have 1. a field name, 
        2. data type. Strings use string_dtype(), numbers can be represented as integers=i, floats=f. Number after integers or floats indicate bit-limit of those types.

Explanation of this line .create_dataset() from h5py,
        Example:        f.create_dataset("logs", shape=(0,), maxshape=(None,), dtype=dt)
        Argument 1:  name of dataset
        Argument 2: 'size' of the dataset. (0,) indicates an empty dataset. 
        Argument 3:  maxshape=(), specifies limit of dataset's size but can also be set to None
        Argument 4: datatype. can pass a np.dtype variable type here

Demo version of store_entry(entry, savedFrame)
        "entry": a list of dictionaries. For example: let entry= [{'uniqueID':0,'location': 'not_enabled',  'worker name': 'unnamed', 'confidence':'-1', 'timestamp': 111},
        ]

        "savedFrame": numpy.ndarray created from cv2
        Demo does not have other object recognition models supported and can thus only support frames from the camera
        
Integration-ready version of store_entry (entry,camFrame,bodyCrop, faceCrop)
        "entry": a list of dictionaries. For example: [{'uniqueID':0,'location': 'not_enabled',  'worker name': 'unnamed', 'confidence':'-1', 'timestamp': 111},
        ]

        "savedFrame", "bodyCrop", "faceCrop": expects objects of type numpy.ndarray corresponding to the
        camera frame, the crop of just the body, and the crop of face resepctively