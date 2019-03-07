import logging
import os
import time

class LOG:
    def __init__(self, name, errorLog=False, allLog=False):
        # 创建一个logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 设置日志存放路径，日志文件名
        # 获取本地时间，转换为设置的格式
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        # 定义日志输出格式
        # 以时间-日志器名称-日志级别-日志内容的形式展示
        all_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if allLog:
            # 设置所有日志存放路径
            all_log_path = os.path.join(os.path.dirname(os.getcwd()), 'logging/All_Logs/')
            # 设置日志文件名
            all_log_name = all_log_path + rq + '.log'
            # 创建handler
            # 创建一个handler写入所有日志
            fh = logging.FileHandler(all_log_name)
            # 设置日志级别
            fh.setLevel(logging.INFO)
            # 给logger添加handler
            fh.setFormatter(all_log_formatter)
            # 给logger添加handler
            self.logger.addHandler(fh)

        if errorLog:
            #设置错误日志的存放路径
            error_log_path = os.path.join(os.path.dirname(os.getcwd()), 'logging/Error_Logs/')
            # 设置日志文件名
            error_log_name = error_log_path + rq + '.log'
            # 创建handler
            # 创建一个handler写入错误日志
            eh = logging.FileHandler(error_log_name)
            eh.setLevel(logging.ERROR)
            # 以时间-日志器名称-日志级别-文件名-函数行号-错误内容
            error_log_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s  - %(lineno)s - %(message)s')
            eh.setFormatter(error_log_formatter)
            # 给logger添加handler
            self.logger.addHandler(eh)


        # 创建一个handler输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # 将定义好的输出形式添加到handler
        ch.setFormatter(all_log_formatter)
        # 给logger添加handler
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger