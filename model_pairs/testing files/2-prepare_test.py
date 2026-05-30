import json
import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import splitfolders as sf

base_path = '.\\video\\'
faces_path = '.\\faces\\test'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

allfiles = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

for filename in allfiles:
    print(filename)
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):   
        print('Copying to :' + faces_path)
        copy_tree(tmp_path, faces_path)
