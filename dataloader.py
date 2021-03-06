import numpy as np
import cPickle
import yaml
import random
##########################################################
##这块的主要差距，就在于因为是词，不是以句子为单位，所以以token为单位，进行batch的拆开，打乱，重组。
##########################################################

with open("SeqGAN.yaml") as stream:#这个没变化
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class Gen_Data_loader(): #主要是操作batch 
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file): #创建一个batch
        self.token_stream = [] #token的英文是记号，这里应该是记号流，也就是那些音乐符号的流 吧
        with open(data_file, 'rb') as f: #读取数据
            # load pickle data
            data = cPickle.load(f)
            for line in data:
                parse_line = [int(x) for x in line]
                self.token_stream.extend(parse_line) #这个是新加入的，下面两行是筛选的
                # if len(parse_line) == config['SEQ_LENGTH']:  
                #     self.token_stream.append(parse_line)
        if len(self.token_stream) % (self.batch_size * config['SEQ_LENGTH']) != 0:
            self.token_stream = self.token_stream[:-(len(self.token_stream) % (self.batch_size * config['SEQ_LENGTH']))]
        self.num_batch = int(len(self.token_stream) / (self.batch_size * config['SEQ_LENGTH'])) #batch的总数
        ##这个和原版不太一样。因为它是传入的符号，以符号为单位，所以符号数目/batch大小*每一句的总长度
        self.token_stream = np.reshape(self.token_stream, (-1, config['SEQ_LENGTH'])) 
        
        np.random.shuffle(self.token_stream)# 打乱    token_stream和sequnece_batch都是个啥鸭
        
        self.sequence_batch = np.array(np.split(self.token_stream, self.num_batch, 0)) #先分割再变成数组
        np.random.shuffle(self.sequence_batch)#打乱                      
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
        # flatten and shuffle the data
        sequence = np.reshape(self.sequence_batch, (-1, config['SEQ_LENGTH']))
        np.random.shuffle(sequence)
        self.sequence_batch = np.array(np.split(sequence, self.num_batch, 0))


class Dis_realdataloader(): ## 分为两个，是不是为了让读取更有效率，不需要同时读。
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # self.sentences = np.array([])
        # self.labels = np.array([])

    def load_train_data(self, positive_file):
        # Load data
        positive_examples = []
       # negative_examples = []
        with open(positive_file, 'rb')as fin:
            data = cPickle.load(fin)
            for line in data:
                parse_line = [int(x) for x in line]
                positive_examples.extend(parse_line)
            if len(positive_examples) % (self.batch_size * config['SEQ_LENGTH']) != 0:
                positive_examples = positive_examples[:-(len(positive_examples) % (self.batch_size * config['SEQ_LENGTH']))]


            self.num_batch = int(len(positive_examples) / (self.batch_size * config['SEQ_LENGTH']))
            positive_examples = np.reshape(positive_examples, (-1, config['SEQ_LENGTH'])).tolist()
            # Generate labels
            positive_labels = [[0, 1] for _ in positive_examples]
            self.sentences_batches = np.array(np.split(np.array(positive_examples), self.num_batch, 0))
            self.labels_batches = np.array(np.split(np.array(positive_labels), self.num_batch, 0))

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

        # flatten and shuffle
        # shuffle then sort the label for batch discrimination
        sentences = np.reshape(self.sentences_batches, (-1, config['SEQ_LENGTH']))
        labels = np.reshape(self.labels_batches, (-1, 2))
	
	#先完全打乱
        shuffler = np.arange(sentences.shape[0])
        np.random.shuffle(shuffler)
        sentences = sentences[shuffler]
        labels = labels[shuffler]

        #再按照label，也就是类别分开，和基数排序差不多
	sorter = np.argsort(labels[:, 1], kind='mergesort')
        sentences = sentences[sorter]
        labels = labels[sorter]

        self.sentences_batches = np.array(np.split(sentences, self.num_batch, 0))
        self.labels_batches = np.array(np.split(labels, self.num_batch, 0))

        shuffler_batch = np.arange(self.sentences_batches.shape[0])
        np.random.shuffle(shuffler_batch)
        self.sentences_batches = self.sentences_batches[shuffler_batch]
        self.labels_batches = self.labels_batches[shuffler_batch]


class Dis_fakedataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, negative_file):
        # Load data
        negative_examples = []
        with open(negative_file, 'rb')as fin:
            data = cPickle.load(fin)
            for line in data:
                parse_line = [int(x) for x in line]
                negative_examples.extend(parse_line)
            if len(negative_examples) % (self.batch_size * config['SEQ_LENGTH']) != 0:
                negative_examples = negative_examples[:-(len(negative_examples) % (self.batch_size * config['SEQ_LENGTH']))]


            self.num_batch = int(len(negative_examples) / (self.batch_size * config['SEQ_LENGTH']))
            negative_examples = np.reshape(negative_examples, (-1, config['SEQ_LENGTH'])).tolist()
            # Generate labels
            negative_labels = [[1, 0] for _ in negative_examples]
            #negative_labels = np.reshape(negative_labels, (-1, config['SEQ_LENGTH'])).tolist()
            #sequence_batch = np.split(np.array(negative_examples), num_batch, 0)
            self.sentences_batches = np.split(np.array(negative_examples), self.num_batch, 0)
            self.labels_batches = np.split(np.array(negative_labels), self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

        # flatten and shuffle
        # shuffle then sort the label for batch discrimination
        sentences = np.reshape(self.sentences_batches, (-1, config['SEQ_LENGTH']))
        labels = np.reshape(self.labels_batches, (-1, 2))

        shuffler = np.arange(sentences.shape[0])
        np.random.shuffle(shuffler)
        sentences = sentences[shuffler]
        labels = labels[shuffler]

        sorter = np.argsort(labels[:, 1], kind='mergesort')
        sentences = sentences[sorter]
        labels = labels[sorter]

        self.sentences_batches = np.array(np.split(sentences, self.num_batch, 0))
        self.labels_batches = np.array(np.split(labels, self.num_batch, 0))

        shuffler_batch = np.arange(self.sentences_batches.shape[0])
        np.random.shuffle(shuffler_batch)
        self.sentences_batches = self.sentences_batches[shuffler_batch]
        self.labels_batches = self.labels_batches[shuffler_batch]


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file, 'rb')as fin:
            data = cPickle.load(fin)
            for line in data:
                parse_line = [int(x) for x in line]
                # if len(parse_line) != config['SEQ_LENGTH']:
                #     continue
                # if len(parse_line) == config['SEQ_LENGTH']:
                positive_examples.extend(parse_line)
        with open(negative_file, 'rb')as fin:
            data = cPickle.load(fin)
            for line in data:
                parse_line = [int(x) for x in line]
                # if len(parse_line) != config['SEQ_LENGTH']:
                #     continue
                # if len(parse_line) == config['SEQ_LENGTH']:
                negative_examples.extend(parse_line)
        if len(positive_examples) % (self.batch_size * config['SEQ_LENGTH']) != 0:
            positive_examples = positive_examples[:-(len(positive_examples) % (self.batch_size * config['SEQ_LENGTH']))]
        if len(negative_examples) % (self.batch_size * config['SEQ_LENGTH']) != 0:
            negative_examples = negative_examples[:-(len(negative_examples) % (self.batch_size * config['SEQ_LENGTH']))]


        positive_examples = np.reshape(positive_examples, (-1, config['SEQ_LENGTH'])).tolist()
        negative_examples = np.reshape(negative_examples, (-1, config['SEQ_LENGTH'])).tolist()
        np.random.shuffle(positive_examples)
        np.random.shuffle(negative_examples)
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        #positive_labels = np.reshape(positive_labels, (-1, config['SEQ_LENGTH'])).tolist()
        #negative_labels = np.reshape(negative_labels, (-1, config['SEQ_LENGTH'])).tolist()

        pos_num_batch = int(len(positive_examples) / self.batch_size)
        neg_num_batch = int(len(negative_examples) / self.batch_size)
        self.num_batch = pos_num_batch + neg_num_batch


        # not using minibatch discrimination
        # TODO: check if this is right
        sentences = np.concatenate([np.array(positive_examples), np.array(negative_examples)])
        labels = np.concatenate([np.array(positive_labels), np.array(negative_labels)])
        shuffler = np.arange(sentences.shape[0])
        np.random.shuffle(shuffler)
        sentences = sentences[shuffler]
        labels = labels[shuffler]
        self.sentences_batches = np.array(np.split(sentences, self.num_batch, 0))
        self.labels_batches = np.array(np.split(labels, self.num_batch, 0))

        # # follow minibatch discrimination technique: real or fake-only minibatch
        # pos_sentences_batches = np.split(np.array(positive_examples), pos_num_batch, 0)
        # pos_labels_batches = np.split(np.array(positive_labels), pos_num_batch, 0)
        # neg_sentences_batches = np.split(np.array(negative_examples), neg_num_batch, 0)
        # neg_labels_batches = np.split(np.array(negative_labels), neg_num_batch, 0)
        # self.sentences_batches = np.concatenate([pos_sentences_batches, neg_sentences_batches])
        # self.labels_batches = np.concatenate([pos_labels_batches, neg_labels_batches])
        # shuffler = np.arange(self.sentences_batches.shape[0])
        # np.random.shuffle(shuffler)
        # self.sentences_batches = self.sentences_batches[shuffler]
        # self.labels_batches = self.labels_batches[shuffler]

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

        # flatten and shuffle
        # shuffle then sort the label for batch discrimination
        sentences = np.reshape(self.sentences_batches, (-1, config['SEQ_LENGTH']))
        labels = np.reshape(self.labels_batches, (-1, 2))

        shuffler = np.arange(sentences.shape[0])
        np.random.shuffle(shuffler)
        sentences = sentences[shuffler]
        labels = labels[shuffler]

        # # if using batch discrimination, activate this code
        # sorter = np.argsort(labels[:, 1], kind='mergesort')
        # sentences = sentences[sorter]
        # labels = labels[sorter]

        self.sentences_batches = np.array(np.split(sentences, self.num_batch, 0))
        self.labels_batches = np.array(np.split(labels, self.num_batch, 0))

        shuffler_batch = np.arange(self.sentences_batches.shape[0])
        np.random.shuffle(shuffler_batch)
        self.sentences_batches = self.sentences_batches[shuffler_batch]
        self.labels_batches = self.labels_batches[shuffler_batch]


