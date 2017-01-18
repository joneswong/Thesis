import os.path

class FBDataset(object):

    def __init__(self, train_path, valid_path=None, test_path=None):
        self.entity2id = dict()
        self.relation2id = dict()
        self.train = []
        self.valid = []
        self.test = []

        with open(train_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                if columns[0] not in self.entity2id:
                    self.entity2id[columns[0]] = len(self.entity2id)
                if columns[1] not in self.relation2id:
                    self.relation2id[columns[1]] = len(self.relation2id)
                if columns[2] not in self.entity2id:
                    self.entity2id[columns[2]] = len(self.entity2id)
                self.train.append((columns[0],columns[1],columns[2]))

        if valid_path:
            with open(valid_path, 'r') as f:
                for line in f:
                    columns = line.strip().split('\t')
                    if columns[0] not in self.entity2id:
                        self.entity2id[columns[0]] = len(self.entity2id)
                    if columns[1] not in self.relation2id:
                        self.relation2id[columns[1]] = len(self.relation2id)
                    if columns[2] not in self.entity2id:
                        self.entity2id[columns[2]] = len(self.entity2id)
                    self.valid.append((columns[0], columns[1], columns[2]))

        if test_path:
            with open(test_path, 'r') as f:
                for line in f:
                    columns = line.strip().split('\t')
                    if columns[0] not in self.entity2id:
                        self.entity2id[columns[0]] = len(self.entity2id)
                    if columns[1] not in self.relation2id:
                        self.relation2id[columns[1]] = len(self.relation2id)
                    if columns[2] not in self.entity2id:
                        self.entity2id[columns[2]] = len(self.entity2id)
                    self.test.append((columns[0], columns[1], columns[2]))

        print("%d training triples"%(len(self.train)))
        print("%d valid triples"%(len(self.valid)))
        print("%d test triples"%(len(self.test)))
        print("%d entities"%(len(self.entity2id)))
        print("%d relations"%(len(self.relation2id)))