import os
import re

def setup_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

# get in-digits in input string(ips)
get_in_digits = lambda s: int(re.sub('\D', '', s))

def listdir(dir, file_ext):
    # list files ending with {file_ext}
    files = [f for f in os.listdir(dir) if f.endswith(file_ext)]
    # sort files as per in-digits
    files.sort(key=get_in_digits)
    return files