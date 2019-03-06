import logging

class LOG:
    def __init__(self, name, isFile=True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(name)s- %(levelname)s - %(message)s')
        filePath = ""
        f = logging.FileHandler(filePath)
        f.setLevel(logging.DEBUG)
        f.format(formatter)

        self.logger.addHandler(f)

