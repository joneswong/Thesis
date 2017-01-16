import os.path

class NYTDataset(object):

    def __init__(self, input_fold, word2id):
        relation_vocabulary_path = os.path.join(input_fold, "relation2id.txt")
        self.relation2id = dict()
        with open(relation_vocabulary_path, 'r') as f:
            for line in f:
                columns = line.strip().split(' ')
                if len(columns) != 2:
                    break
                self.relation2id[columns[0]] = int(columns[1])
        print("%d relation categories."%(len(self.relation2id)))

        train_path = os.path.join(input_fold, "train.txt")
        self.train_triple2mentions = dict()
        self.train_labels = []
        self.train_mentions = []
        self.train_mention_lengths = []
        self.train_head_positions = []
        self.train_tail_positions = []
        with open(train_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')

                triple = columns[0] + '\t' + columns[1] + '\t' + columns[4]
                if triple not in self.train_triple2mentions:
                    self.train_triple2mentions[triple] = []
                self.train_triple2mentions[triple].append(len(self.train_labels))

                if columns[2] in word2id:
                    head = word2id[columns[2]]
                else:
                    head = 0
                if columns[3] in word2id:
                    tail = word2id[columns[3]]
                else:
                    tail = 0

                if columns[4] in self.relation2id:
                    label = self.relation2id[columns[4]]
                else:
                    label = self.relation2id["NA"]
                self.train_labels.append(label)

                tokens = columns[5].split(' ')
                tk_ids = []
                head_idx = 0
                tail_idx = 0
                for i in range(len(tokens)):
                    if tokens[i] == "###END###":
                        break
                    if tokens[i] == columns[2]:
                        head_idx = i
                    if tokens[i] == columns[3]:
                        tail_idx = i

                    if tokens[i] in word2id:
                        id = word2id[tokens[i]]
                    else:
                        id = 0

                    tk_ids.append(id)

                #assert head_idx != -1 and tail_idx != -1, "%d\t%d\t%s\t%s"%(head_idx, tail_idx, columns[2], columns[3])

                self.train_mentions.append(tk_ids)
                self.train_mention_lengths.append(len(tk_ids))
                self.train_head_positions.append(head_idx)
                self.train_tail_positions.append(tail_idx)

        print("Max distance between head and tail is %d."%(max([abs(self.train_head_positions[i]-self.train_tail_positions[i]) for i in range(len(self.train_head_positions))])))
        print("Max length of training mentions is %d."%(max(self.train_mention_lengths)))
        print("Max number of training triple mentions is %d."%(max([len(v) for k,v in self.train_triple2mentions.items()])))

        test_path = os.path.join(input_fold, "test.txt")
        self.test_triple2mentions = dict()
        self.test_labels = []
        self.test_mentions = []
        self.test_mention_lengths = []
        self.test_head_positions = []
        self.test_tail_positions = []
        with open(test_path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')

                #different from the key of training set, here we consider multi-label classification problem
                triple = columns[0] + '\t' + columns[1]
                if triple not in self.test_triple2mentions:
                    self.test_triple2mentions[triple] = []
                self.test_triple2mentions[triple].append(len(self.test_labels))

                if columns[2] in word2id:
                    head = word2id[columns[2]]
                else:
                    head = 0
                if columns[3] in word2id:
                    tail = word2id[columns[3]]
                else:
                    tail = 0

                if columns[4] in self.relation2id:
                    label = self.relation2id[columns[4]]
                else:
                    label = self.relation2id["NA"]
                self.test_labels.append(label)

                tokens = columns[5].split(' ')
                tk_ids = []
                head_idx = 0
                tail_idx = 0
                for i in range(len(tokens)):
                    if tokens[i] == "###END###":
                        break
                    if tokens[i] == columns[2]:
                        head_idx = i
                    if tokens[i] == columns[3]:
                        tail_idx = i

                    if tokens[i] in word2id:
                        id = word2id[tokens[i]]
                    else:
                        id = 0

                    tk_ids.append(id)

                #assert head_idx != -1 and tail_idx != -1

                self.test_mentions.append(tk_ids)
                self.test_mention_lengths.append(len(tk_ids))
                self.test_head_positions.append(head_idx)
                self.test_tail_positions.append(tail_idx)

        print("Max distance between head and tail is %d." % (max(
            [abs(self.test_head_positions[i] - self.test_tail_positions[i]) for i in
             range(len(self.test_head_positions))])))
        print("Max length of test mentions is %d." % (max(self.test_mention_lengths)))
        print("Max number of test triple mentions is %d." % (
        max([len(v) for k, v in self.test_triple2mentions.items()])))
