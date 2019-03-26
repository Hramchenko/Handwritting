import cv2
import scipy as sp
from math import floor
import torch
import random

class IAMWords:
    
    def __init__(self, dataset_type, IAM_PATH, batch_size=50, line_height = 128, line_width = 400, scale=0.5, rand_x=30, rand_y=5):
        self.local_rng_state = None
        self.сapture_rng()
        random.seed(1)
        self.free_rng()
        self.dataset_type = dataset_type
        self.scale = scale
        self.line_height = line_height
        self.line_width = line_width
        self.dx = 20
        self.rand_x = rand_x
        self.rand_y = rand_y
        self.datasetPath = IAM_PATH
        self.batch_size = batch_size
        self.lines = self.read_lines()
        self.words_list = self.read_words_list()
        self.tmp = sp.ones([self.batch_size, self.line_height, self.line_width], dtype="uint8")
        self.tmpText = ""
        self.codes = {}
        self.inv_codes = {}
        self.alphabet = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        self.codes = {c:i for i,c in enumerate(self.alphabet)}
        self.inv_codes = {i:c for i,c in enumerate(self.alphabet)}
        self.start, self.start_code = self.register_symbol("<START>") 
        self.stop, self.stop_code = self.register_symbol("<STOP>") 
        self.word_size = 30
        self.word_images = {}
        images_file = IAM_PATH + "words." + dataset_type +".pkl"
        import os.path
        if not os.path.isfile(images_file):
            print(images_file + " not exist.")
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
            print("Reading " + images_file + "...")
            import pickle
            f = open(images_file, "rb")
            self.word_images = pickle.load(f)
            f.close()
        self.group_words()
        self.to_start(1000)
        print("Reading finished")
        
    def register_symbol(self, s):
        l = len(self.codes)
        self.codes[s] = l
        self.inv_codes[l] = s
        return (s, l)
        
    def сapture_rng(self):
        self.global_rng_state = random.getstate() 
        if self.local_rng_state != None:
            random.setstate(self.local_rng_state)
        
    def free_rng(self):
        self.local_rng_state = random.getstate() 
        random.setstate(self.global_rng_state)
        
    def group_words(self):
        self.grouped_words = []
        
        for idx in range(0, len(self.words_list)):
            w = self.words_list[idx][-1]
            l = len(w)
            if l == 1:
                if w not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                    continue
            while l >= len(self.grouped_words):
                self.grouped_words.append([])
            self.grouped_words[l].append(idx)  
            
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
        from math import floor
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
    
    def fill_image(self, i, l, use_binarization=True, equalize=False):
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
        scale = self.scale*random.uniform(0.9, 1.1)
        new_height = floor(box[3]*scale)

        if (new_height>self.line_height):
            return None
        new_width = floor(box[2]*scale)

        if new_width < 2 or new_height < 2:
            return None
        f_name = self.word_file(l)
        data = self.word_images[l[0]]
        data = sp.frombuffer(data, sp.uint8)
        try: 
            img = cv2.imdecode(data, 0)
            if equalize:
                img = cv2.equalizeHist(img,gray)
            elif use_binarization:
                _, img = cv2.threshold(img,gray,255,cv2.THRESH_BINARY)
            img=cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)

            word_dy = 128 - floor(word_dy*self.scale) - new_height + floor(random.uniform(-self.rand_y, self.rand_y))
            dx = self.dx + floor(random.uniform(0, self.rand_x))
            max_x = dx+new_width
            if max_x > self.line_width:
                new_width -= max_x - self.line_width
            self.tmp[i, word_dy:word_dy+new_height, 0+dx:dx+new_width]=img
        except:
            return None
        w = l[-1]
        W = self.encode_word(w)
        return W
    
    def encode_word(self, w):
        w = w #+ (" "*(self.word_size - len(w)))
        W = []
        for a in w:
            W.append(self.codes[a])
        W = torch.LongTensor(W)
        return W
    
    def decode_word(self, W):
        w = ""
        for i in range(0, W.shape[0]):
            idx = W[i].item()
            w += self.inv_codes[idx]
        return w
    
    def to_start(self, max_size):
        self.сapture_rng()
        self.total = 0
        self.group_samples = []
        self.last_indexes = []
        groups = []
        for i in range(0, len(self.grouped_words)):
            if i >= max_size:
                break
            random.shuffle(self.grouped_words[i])
            cnt = len(self.grouped_words[i])
            self.group_samples.append(cnt)
            self.last_indexes.append(0)
            self.total += cnt
        self.free_rng()
        
        
        
    def make_batch_(self, use_binarization=True, equalize=False):
        self.сapture_rng()
        Y = random.random()*self.total
        self.free_rng()
        for i in range(0, len(self.grouped_words)):
            Y -= self.group_samples[i]
            if Y <= 0:
                group_index = i
                break
        self.tmp.fill(255)
        img_idx = 0
        images = []
        texts = []
        current_word = self.last_indexes[group_index]
        group = self.grouped_words[group_index]
        while img_idx < self.batch_size:
            if current_word >= len(group):
                return None
            
            word_idx = group[current_word]
            current_word += 1           
            self.last_indexes[group_index] += 1
            self.group_samples[group_index] -= 1
            self.total -= 1
            l = self.words_list[word_idx]
            status = self.fill_image(img_idx, l, use_binarization, equalize)
            if status is None:
                continue
            texts.append(status)
            img_idx += 1
        t = torch.stack(texts)
        i = torch.as_tensor(self.tmp)
        i = i.type(torch.FloatTensor)
        return (i, t)
    
    def make_batch(self, use_binarization=True, equalize=False):
        while True:
            if self.total == 0:
                return None
            result = self.make_batch_(use_binarization, equalize) 
            if result is not None:
                return result           
        
