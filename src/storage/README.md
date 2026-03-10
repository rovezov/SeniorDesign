REQUIRES installation of h5py. Can be achieved with "pip install h5py"

Functions of entry_storage.py:
        Saves text logs and images into an HDF5 (.h5) file. Through class called Violation_Entry.
        
        In the demo version, the object created from Violation_Entry() has 4 parameters: the camera frame, location, name and warning_type. If location, name and warning type are not specified, they will be written with default values. The file path for the hdf5 log is also initalized

        For the integrated build, the first 3 parameters should be: camera frame, body image and face image. Then the next 3 paramaters are location, name and warning_type.


        Example: store_call=Violation_Entry(camFrame, location, name,warning_type)

        When constructing the Violation_Entry object, add_entry() is privately called, This function creates or opens a hdf5 file, creates or appends a dataset containing the text portions of the entries named 'logs', and creates or appends to groups hosting different types of images. The logs dataset and image groups share an ID that is automatically incremented, allowing for direct association.

        Note: .add_entry()has commented-out code for add_entry()'s parameters and the code to save other images in preparation for integration with the main demo.
 
        All images sent to .add_entry() must be a numpy.ndarray data type


logsReader.py functions:
        constructor, __init__(foldername, filename). Uses default folder and file names if not specified upon creating an object
        
        get_logs(): only returns text-based logs from dataset 'logs'. Indexes saved as integers. Data returned shows all column values for a given index number
        
        print_all_logs(): opens hdf5 file to print all text-based logs
        
        get_all_cam_frames(): returns dictionary where key=ID number, value=numpy array representation of an image
                NOTE: keys are returned as strings, NOT integers

        show_all_came_frames(delay): opens hdf5 file to display all images in cam_frames.
                delay specifies how long a image stays before moving to the next one
        show_entry(uid):
                opens hdf5 file, prints text log and display camera frame given ID number uid

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

        Note: i8, f4 are analogous to numpy's integer and float types, where the numbers specify bit size

        each element of this array must have 1. a field name, 
        2. data type. Strings use string_dtype(), numbers can be represented as integers=i, floats=f. Number after integers or floats indicate bit-limit of those types.

Explanation of this line .create_dataset() from h5py,
        Example:        f.create_dataset("logs", shape=(0,), maxshape=(None,), dtype=dt)
        Argument 1:  name of dataset
        Argument 2: 'size' of the dataset. (0,) indicates an empty dataset. 
        Argument 3:  maxshape=(), specifies limit of dataset's size but can also be set to None
        Argument 4: datatype. can pass a np.dtype variable type here
