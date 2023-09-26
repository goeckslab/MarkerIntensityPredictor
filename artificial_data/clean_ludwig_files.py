import os
from pathlib import Path

directory = '.'

# search path for .json and hdf5 files
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and '.meta.json' in f or '.hdf5' in f:
        print(f)
        # delete file
        os.unlink(f)
