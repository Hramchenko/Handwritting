import cv2
import scipy as sp
from math import floor
import torch
import tarfile

class IAMWords:
    
    def __init__(self, dataset_type, IAM_PATH, batch_size=50):
        self.dataset_type = dataset_type
        self.scale = 0.5
        self.line_height = 128
        self.line_width = 400
        self.dx = 20
        self.datasetPath = IAM_PATH
        self.batch_size = batch_size
        self.lines = self.read_lines()
        self.words_list = self.read_words_list()
        self.tmp = sp.ones([self.batch_size, self.line_height, self.line_width], dtype="uint8")
        self.tmpText = ""
        self.current = 0
        self.codes = {}
        self.inv_codes = {}
        self.word_size = 30
        self.word_images = {}
        images_file = IAM_PATH + "words." + dataset_type +".pkl"
        import os.path
        if not os.path.isfile(images_file):
            print("Reading files...")
            cnt = len(self.words_list)
            for idx in range(0, cnt):
                l = self.words_list[idx]
                if True:
                    f_name = self.word_file(l)
                    f = open(f_name, "rb")
                    c = f.read()
                    name = l[0]
                    self.word_images[name] = c
                    if idx%1000 == 0:
                        print (str(idx) + " of " + str(cnt) + " finished.")
            import pickle
            f = open(images_file, "wb")
            pickle.dump(self.word_images, f)
            f.close()
        else:
            import pickle
            f = open(images_file, "rb")
            self.word_images = pickle.load(f)
            f.close()

        print("Reading finished")
    
    def read_lines(self):
        f_name = self.datasetPath + "ascii/lines.txt"
        f = open(f_name, "r")
        doc_lines={}
        for l in f.readlines():
            l = self.check_line(l)
            if l is None:
                continue
            b = [int(l[4]), int(l[5]), int(l[6]), int(l[7])]
            doc_lines[l[0]] = b
        return doc_lines
        
    def read_words_list(self):
        f_name = self.datasetPath + "ascii/words.txt"
        f = open(f_name, "r")
        lines_orig = f.readlines()
        lines = []
        for l in lines_orig:
            l = self.check_line(l)
            if l is None:
                continue
            lines.append(l)
        import random
        from math import floor
        s = random.getstate()
        random.seed(1)
        random.shuffle(lines)
        random.setstate(s)
        part = floor(len(lines)/10)
        if self.dataset_type == "test":
            lines = lines[0: part]
        elif self.dataset_type == "valid":
            lines = lines[part: part*2]
        elif self.dataset_type == "train":
            lines = lines[part*2:]
        else:
            raise Exception()
        return lines
    
    def check_line(self, l):
        if l.startswith("#"):
            return None
        l = l.strip()
        if l == "":
            return None
        l = l.split(" ")
        if (l[1] != "ok"):
            return None
        return l
   
    def word_file(self, l):
        word_id = l[0].split("-")
        words_folder = self.datasetPath + "words/"    
        f_name = words_folder + word_id[0] + "/" + word_id[0] + "-" + word_id[1] + "/" + l[0] + ".png"   
        return f_name
    
    def fill_image(self, i, l):
        text = l[-1]
        if len(text) > self.word_size:
            return None
        gray = int(l[2])
        box = [int(l[3]), int(l[4]), int(l[5]), int(l[6])]
        word_id = l[0].split("-")
        line_id="-".join(word_id[:3])

        line_box = self.lines[line_id]
        word_dy = box[1]-line_box[1]

        if word_dy < 0:
            return None

        new_height = floor(box[3]*self.scale)

        if (new_height>self.line_height):
            return None
        new_width = floor(box[2]*self.scale)

        if new_width < 2 or new_height < 2:
            return None
        f_name = self.word_file(l)
        data = self.word_images[l[0]]
        data = sp.frombuffer(data, sp.uint8)
        try: 
            img = cv2.imdecode(data, 0)
            _, img = cv2.threshold(img,gray,255,cv2.THRESH_BINARY)
            img=cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)

            word_dy = 128 - floor(word_dy*self.scale) - new_height
            max_x = self.dx+new_width
            if max_x > self.line_width:
                new_width -= max_x - self.line_width
            self.tmp[i, word_dy:word_dy+new_height, 0+self.dx:self.dx+new_width]=img
        except:
           return None
        w = l[-1]
        W = self.encode_word(w)
        return W
    
    def encode_word(self, w):
        w = w + (" "*(self.word_size - len(w)))
        W = []
        for a in w:
            if a not in self.codes:
                idx = len(self.codes)
                self.codes[a] = idx
                self.inv_codes[idx] = a
            W.append(self.codes[a])
        W = torch.LongTensor(W)
        return W
    
    def decode_word(self, W):
        w = ""
        for i in range(0, self.word_size):
            idx = W[i].item()
            w += self.inv_codes[idx]
        return w
    
    def to_start(self):
        self.current = 0
    
    def make_batch(self):
        self.tmp.fill(255)
        img_idx = 0
        images = []
        texts = []
        while img_idx < self.batch_size:
            if self.current >= len(self.words_list):
                return None
            l = self.words_list[self.current]
            self.current += 1
            status = self.fill_image(img_idx, l)
            if status is None:
                continue
            texts.append(status)
            img_idx += 1
        t = torch.stack(texts)
        i = torch.as_tensor(self.tmp)
        i = i.type(torch.FloatTensor)
        return (i, t)