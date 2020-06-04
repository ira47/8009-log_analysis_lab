import csv
import os

class hdfs_deeplog_preprocessor:
    ANOMALY_LABEL = '../Data/log/hdfs/anomaly_label.csv'
    LOG_FILE = '../Data/log/hdfs/HDFS_40w'
    MOFIFIED_LOG_FILE = '../Data/log/hdfs/modified_HDFS_40w'
    WORD_VECTOR_FILE = '../Data/log/hdfs/word2vec_HDFS_40w'
    LOGKEY_DIR = '../Data/FTTreeResult-HDFS/clusters/'
    SEQUENTIAL_OUTPUT_DIR = '../Data/log_preprocessor/logkey/'
    VARIABLE_OUTPUT_DIR = '../Data/log_preprocessor/logvalue/'
    OUTPUT_FILES = ['abnormal','normal']
    TRAIN_PARAMETER = ['train/', 0.0, 0.5]
    VALIDATE_PARAMETER = ['validate/', 0.5, 0.75]
    TEST_PARAMETER = ['test/', 0.75, 1.0]
    LOG_LINE = 400000
    NUM_OF_LOGKEY = 26
    VECTOR_DIMENSION = 10

    is_block_normal = {}
    block_to_line = {}
    line_to_logkey = []
    block_to_logkey = {}

    word_to_vector = {}
    modified_logs = []

    '''
    -----------------------------------------------
    以下是处理sequential部分
    -----------------------------------------------
    '''

    def load_normal_info(self):
        NORMAL_WORD = 'Normal'
        FIRST_LINE_BLOCK_NAME = 'BlockId'

        with open(self.ANOMALY_LABEL,'r') as f:
            lines = csv.reader(f)
            for line in lines:
                block = line[0]
                normal_word = line[1]
                if normal_word == NORMAL_WORD:
                    normal_info = True
                else:
                    normal_info = False
                if block != FIRST_LINE_BLOCK_NAME:
                    self.is_block_normal[block] = normal_info
                    self.block_to_line[block] = []

    def get_blockid(self, line):
        words = line.strip().split(' ')
        for word in words:
            if len(word)>4 and word[:4] == 'blk_':
                return word
        print('无法找到block_id')
        print(line)
        exit(1)

    def load_line_info(self):
        with open(self.LOG_FILE,'r') as f:
            for line_index in range(self.LOG_LINE):
                line = f.readline()
                block = self.get_blockid(line)
                self.block_to_line[block].append(line_index)
        # print(self.block_to_line['blk_-1608999687919862906'])

    def load_logkey_info(self):
        self.line_to_logkey = [0 for i in range(self.LOG_LINE)]
        for logkey in range(1,self.NUM_OF_LOGKEY+1):
            with open(self.LOGKEY_DIR+str(logkey),'r') as f:
                f.readline()
                lines = f.readline().strip().split(' ')
                for line in lines:
                    line_index = int(line)
                    if line_index>=self.LOG_LINE:
                        print('cluster文件中某行的行数过大')
                        print(line)
                        exit(2)
                    self.line_to_logkey[line_index] = logkey

    def generate_block_to_logkey(self):
        for block,lines in self.block_to_line.items():
            logkeys = []
            for line in lines:
                logkey = self.line_to_logkey[line]
                logkeys.append(logkey)
                self.block_to_logkey[block] = logkeys

    def output_sequential(self,parameter):
        dir_suffix = parameter[0]
        left_rate = parameter[1]
        right_rate = parameter[2]
        left_range = int(len(self.block_to_line) * left_rate)
        right_range = int(len(self.block_to_line) * right_rate)

        if not os.path.exists(self.SEQUENTIAL_OUTPUT_DIR + dir_suffix):
            os.makedirs(self.SEQUENTIAL_OUTPUT_DIR + dir_suffix)
        for file_index in range(len(self.OUTPUT_FILES)):
            with open(self.SEQUENTIAL_OUTPUT_DIR + dir_suffix + self.OUTPUT_FILES[file_index],'w') as f:
                for block_id,block in enumerate(self.block_to_logkey):
                    if block_id not in range(left_range, right_range):
                        continue
                    if self.is_block_normal[block]:
                        file_select = 1
                    else:
                        file_select = 0
                    if file_select == file_index:
                        logkeys = self.block_to_logkey[block]
                        f.write(' '.join(str(logkey) for logkey in logkeys)+'\n')

    '''
    -----------------------------------------------
    以下是处理variable部分
    -----------------------------------------------
    '''


    def load_word_vector(self):
        with open(self.WORD_VECTOR_FILE, 'r') as r:
            for line in r.readlines():
                list_line = line.split(' ')
                value = list(map(float, list_line[1:]))
                key = list_line[0]
                self.word_to_vector[key] = value

    def load_modified_log(self):
        with open(self.MOFIFIED_LOG_FILE, 'r') as file:
            content_list = file.readlines()
            self.modified_logs = [x.strip() for x in content_list]

    def get_sentence_vector(self, sentence):
        words = sentence.split(' ')
        old_vector = [0.0 for i in range(self.VECTOR_DIMENSION)]
        for word in words:
            # print(word)
            if word not in self.word_to_vector.keys():
                another_vector = [0.0 for i in range(self.VECTOR_DIMENSION)]
            else:
                another_vector = self.word_to_vector[word]
            new_vector = []
            for i, j in zip(old_vector, another_vector):
                new_vector.append(i + j)
            old_vector = new_vector

        word_count = len(words)
        for idx, value in enumerate(old_vector):
            old_vector[idx] = value / word_count
        vector_str = list(map(str, old_vector))
        output = ','.join(vector_str)
        return output

    def output_variable(self,parameter):
        dir_suffix = parameter[0]
        left_rate = parameter[1]
        right_rate = parameter[2]
        left_range = int(len(self.block_to_line) * left_rate)
        right_range = int(len(self.block_to_line) * right_rate)
        if not os.path.exists(self.VARIABLE_OUTPUT_DIR + dir_suffix+ 'normal/'):
            os.makedirs(self.VARIABLE_OUTPUT_DIR + dir_suffix+ 'normal/')
        if not os.path.exists(self.VARIABLE_OUTPUT_DIR + dir_suffix+ 'abnormal/'):
            os.makedirs(self.VARIABLE_OUTPUT_DIR + dir_suffix+ 'abnormal/')
        logkey_to_normal_writelist = [[] for i in range(self.NUM_OF_LOGKEY+1)]
        logkey_to_abnormal_writelist = [[] for i in range(self.NUM_OF_LOGKEY+1)]

        for block_id,block in enumerate(self.block_to_line):
            if block_id not in range(left_range,right_range):
                continue
            lines = self.block_to_line[block]
            logkey_to_variables = [[] for i in range(self.NUM_OF_LOGKEY+1)]
            for line in lines:
                log = self.modified_logs[line]
                vector = self.get_sentence_vector(log)
                logkey = self.line_to_logkey[line]
                logkey_to_variables[logkey].append(vector)
            for logkey in range(1,self.NUM_OF_LOGKEY+1):
                if len(logkey_to_variables[logkey]) == 0:
                    output_line = '-1'
                else:
                    output_line = ' '.join(logkey_to_variables[logkey])
                if self.is_block_normal[block]:
                    logkey_to_normal_writelist[logkey].append(output_line+'\n')
                else:
                    logkey_to_abnormal_writelist[logkey].append(output_line+'\n')

        for logkey in range(1,self.NUM_OF_LOGKEY+1):
            with open(self.VARIABLE_OUTPUT_DIR + dir_suffix + 'normal/' + str(logkey),'w') as f:
                f.writelines(logkey_to_normal_writelist[logkey])
            with open(self.VARIABLE_OUTPUT_DIR + dir_suffix + 'abnormal/' + str(logkey),'w') as f:
                f.writelines(logkey_to_abnormal_writelist[logkey])

    '''
    -----------------------------------------------
    以下是处理main函数部分
    -----------------------------------------------
    '''


    def generate_sequential(self):
        self.load_normal_info()
        self.load_line_info()
        self.load_logkey_info()
        self.generate_block_to_logkey()
        self.output_sequential(self.TRAIN_PARAMETER)
        self.output_sequential(self.VALIDATE_PARAMETER)
        self.output_sequential(self.TEST_PARAMETER)

    def generate_variable(self):
        self.load_normal_info()
        self.load_line_info()
        self.load_logkey_info()
        self.load_word_vector()
        self.load_modified_log()
        self.output_variable(self.TRAIN_PARAMETER)
        self.output_variable(self.VALIDATE_PARAMETER)
        self.output_variable(self.TEST_PARAMETER)

    def __init__(self):
        self.generate_sequential()
        self.generate_variable()

hdfs_deeplog_preprocessor()
