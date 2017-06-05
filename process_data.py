import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
files = os.listdir('data')
files.sort()
classes = []
examples = []
prev = files[0].split('_')[0]
for f in files:
    cur_id = f.split('_')[0]
    cur_pic = misc.imresize(misc.imread('data/' + f),[28,28])
    cur_pic = (np.float32(cur_pic)/255).flatten()
    if prev == cur_id:
        examples.append(cur_pic)
    else:
        classes.append(examples)
        examples = [cur_pic]
        prev = cur_id
np.save('data',np.asarray(classes))
