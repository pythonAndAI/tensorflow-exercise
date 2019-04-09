import os, shutil

#删除目录底下所有的文件
def remove_file(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                try:
                    os.remove(os.path.join(path, file))
                except Exception:
                    print("remove##", file, "##error!")
    else:
        print(path, "does not exist!")

#删除目录底下所有的文件夹
def remove_folder(path):
    if os.path.exists(path):
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                try:
                    #os.rmdir()只能删除空的文件夹，需要用到shutil.rmtree删除文件夹所有
                    shutil.rmtree(os.path.join(path, folder))
                except Exception:
                    print("remove##", folder, "##error!")
    else:
        print(path, "does not exist!")

#删除整个目录，并创建
def remove_all(path, is_create=True):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception:
            print("remove##", path, "##error!")
        else:
            if is_create:
                os.mkdir(path)
    else:
        print(path, "does not exist!")

#获取路径，不存在创建
def get_path(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path