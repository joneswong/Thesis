import math
import struct

class EmbeddingModel(object):

    def __init__(self, input_path, output_path, apply_normalization=True):
        self.words = []
        self.words.append("UNK")
        self.embeddings = []

        with open(input_path, "rb") as f:
            first_line = f.readline()
            columns = first_line.strip().split(' ')
            self.num_of_words = int(columns[0])
            self.dimension = int(columns[1])
            print("%d\t%d"%(self.num_of_words, self.dimension))
            #set the embedding of UNK as zero vector
            self.embeddings.append([float(0)] * self.dimension)

            while True:
                chs = []
                ch = f.read(1)
                while ch != ' ':
                    if ch == "":
                        break
                    if ch != '\n' and ch != '\c' and ch != '\r':
                        chs.append(ch)
                    ch = f.read(1)
                if len(chs) == 0:
                    break
                word = ''.join(chs)
                self.words.append(word)

                vs = f.read(4 * self.dimension)
                embedding = [struct.unpack('f', vs[4*i:4*i+4])[0] for i in range(self.dimension)]
                if apply_normalization == True:
                    l2n = math.sqrt(sum([v*v for v in embedding]))
                    embedding = [v / l2n for v in embedding]
                self.embeddings.append(embedding)

        assert len(self.words) == len(self.embeddings)
        assert 1 + self.num_of_words == len(self.words)

        self.word2id = dict()
        fout = open(output_path, 'w')

        for i in range(len(self.words)):
            self.word2id[self.words[i]] = i
            fout.write(self.words[i])

            for j in range(self.dimension):
                fout.write('\t')
                fout.write(str(self.embeddings[i][j]))
            fout.write('\n')

        fout.close()

        assert len(self.word2id) == len(self.words)

if __name__=="__main__":
    ob = EmbeddingModel("../thu_baselines/NRE/data/vec.bin", "../thu_baselines/NRE/data/vector2.txt")
