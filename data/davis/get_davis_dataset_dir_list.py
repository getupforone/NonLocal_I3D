import os
import numpy as np
print("thank you")

#dirs = ['/Users/giilkwon/WorkSpace/datasets/DAVIS/JPEGImages/480p']
dirs = ['/data3/DAVIS/JPEGImages/480p']

file_write_obj = open('davis_seqs_list.txt','w')
for dir in dirs:
    seqs = np.sort(os.listdir(dir))
    for ind, seq in enumerate(seqs):
        seq_path = os.path.join(dir,seq)
        line = "{}  {}".format(seq_path, ind)
        print(line)
        file_write_obj.writelines(line)
        #file_write_obj.writelines(seq)
        file_write_obj.write('\n')

file_write_obj.close()