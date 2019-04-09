import subprocess
import os

def run_tensorboard(path="E:\\Alls\\software\\tensorboard", port=None):
    cmd = "tensorboard --logdir=" + path
    if port != None:
        cmd = cmd + " --port=" + str(port)
    print(cmd)
    os.system(cmd)
    # print(subprocess.check_call("tensorboard --logdir=E:\\Alls\\software\\tensorboard", shell=True))
    # print(subprocess.call("tensorboard --logdir=E:\\Alls\\software\\tensorboard", shell=True))
    # print(subprocess.check_output("tensorboard --logdir=E:\\Alls\\software\\tensorboard", shell=True))
