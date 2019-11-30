import os
import numpy as np
import csv
print("thank you")

root_dir_path = '/Users/giilkwon/WorkSpace/datasets/DAVIS/'

file_imagesets_info_path = os.path.join(root_dir_path,'ImageSets/480p')
print(file_imagesets_info_path)
#file_images_path = os.path.join(root_dir_path,'JPEGImages/480p')
imageset_types = ['train','trainval','val']
train_info_path = ''
trainval_info_path = ''
val_info_path =''
train_info_file_name = ''
trainval_info_file_name = ''
val_info_file_name =''
imageset_info_names = {'train':train_info_path,'trainval':trainval_info_path,'val':val_info_path}
imageset_info_file_names = {'train':train_info_file_name,'trainval':trainval_info_file_name,'val':val_info_file_name}
# file_train_write_obj = None
# file_trainval_write_obj = None
# file_val_write_obj = None
# file_write_objs = {'train':file_train_write_obj, 'trainval': file_trainval_write_obj,'val':file_val_write_obj}
#file_write_obj = open('davis_seqs_list.txt','w')

# import csv    
# f = open('output.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# wr.writerow([1, "김정수", False])
# wr.writerow([2, "박상미", True])
# f.close()

for t in imageset_types:
    info_file_name = t + '.txt'
    imageset_info_names[t] = os.path.join(file_imagesets_info_path,info_file_name) 
    #print(imageset_info_names[t])
    imageset_info_file_names[t] = "{}_list.csv".format(t)
    #print(imageset_info_file_names[t]) 
    file_write_obj = open(imageset_info_file_names[t],'w', encoding='utf-8', newline='')
    csv_write = csv.writer(file_write_obj)
    with open(imageset_info_names[t], "r") as fr:
        for idx, path_img_annot in enumerate(fr.read().splitlines()):
            #print(idx, ' ',path_img_annot)
            img_path, annot_path = path_img_annot.split()
            img_path = img_path[1:]

            img_file_path = os.path.join(root_dir_path,img_path)
            print(img_file_path)
            
            csv_write.writerow([img_file_path])
            # file_write_obj.writelines(img_file_path)
            # file_write_obj.write('\n')
    file_write_obj.close()



# file_write_obj = open('TV_seqs_list.txt','w')

# for dir in dirs:
#     seqs = np.sort(os.listdir(dir))
#     for seq in seqs:
#         seq_path = os.path.join(dir,seq)
#         file_write_obj.writelines(seq_path)
#         #file_write_obj.writelines(seq)
#         file_write_obj.write('\n')

# file_write_obj.close()