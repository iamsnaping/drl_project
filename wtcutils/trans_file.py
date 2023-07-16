import os
import numpy as np
import re
import shutil



new_path='D:\\download\\new_data_framework\\'
old_path='D:\\download\\SLD_Framework_dataset\\SLD_Framework_dataset\\'

def change_name(file_dir, old_ext, new_ext):
    list_file = os.listdir(file_dir)  # 返回指定目录
    for file in list_file:
        ext = os.path.splitext(file)  # 返回文件名和后缀
        if old_ext == ext[1]:  # ext[1]是.doc,ext[0]是1
            newfile = ext[0] + new_ext
            os.rename(os.path.join(file_dir, file),
                      os.path.join(file_dir, newfile))


def search_file(old_path):
    list_file=os.listdir(old_path)
    for file_p in list_file:
        p=os.path.join(old_path,file_p)
        if os.path.isdir(p):
            new_p=new_path+p[56:]+'\\'
            if os.path.exists(new_p)==False:
                os.mkdir(new_p)
            search_file(p[:]+'\\')
        else:
            ext=os.path.splitext(file_p)
            if ext[1]=='.features':
                new_file=ext[0]+'.txt'
                new_file_path=new_path+p[56:]
                shutil.copyfile(p,new_file_path)
                chan_pa=new_path+old_path[56:]+'\\'
                change_path=os.path.join(chan_pa,new_file)
                os.rename(new_file_path,change_path)




search_file(old_path)
print(os.path.isdir('D:\\download\\SLD_Framework_dataset\\SLD_Framework_dataset\\s5'))