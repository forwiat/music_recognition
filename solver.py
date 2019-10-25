import urllib.request
from hparams import hyperparams as hp
import codecs
from tqdm import tqdm
import os
class solver:
    def __init__(self):
        pass

    def solve(self, txt, mode='labels'):
        self.txt = txt
        self.mode = mode
        if self.mode == 'music':
            self.solve_music()
        elif self.mode == 'labels':
            self.solve_labels()
        else:
            print('no supported mode in solver, please check ...')
            exit(0)

    def read_file(self):
        lines = codecs.open(self.txt, 'r').readlines()
        if self.mode == 'music':
            count = 0
            self.mdic = {}
            for i in lines[1:]:
                mid, mname, mrename, murl, _, lab = i.strip().split('\t')
                mposl = []
                labl = lab.strip('，').split('，')
                mname = mname.replace(' ', '_')
                mposl.append(mname)
                mposl.append(murl)
                mposl.append(labl)
                self.mdic[count] = mposl
                count += 1
        elif self.mode == 'labels':
            count = 0
            cnt = 0
            self.ldic = {}
            self.label_vacab_dic = {}
            for i in lines[1:]:
                lid, ltype, rever, _ = i.strip().split('\t')[:4]
                lposl = []
                lposl.append(ltype)
                lreverl = rever.strip('，').split('，')
                lposl.append(lreverl)
                self.ldic[count] = lposl
                count += 1
                if ltype not in self.label_vacab_dic.keys():
                    self.label_vacab_dic[ltype] = cnt
                    cnt += 1
                for j in lreverl:
                    if j not in self.label_vacab_dic.keys():
                        self.label_vacab_dic[j] = cnt
                        cnt += 1

    def download(self):
        print('start downloading ...')
        for _, mposl in tqdm(self.mdic.items()):
            mname, murl, _ = mposl[:3]
            fpath = os.path.join(hp.orimp3_dir, mname)
            try:
                urllib.request.urlretrieve(murl, fpath)
            except urllib.request.ContentTooShortError:
                count = 1
                while count <= 5:
                    try:
                        urllib.request.urlretrieve(murl, fpath)
                        break
                    except urllib.request.ContentTooShortError:
                        count += 1
                    if count > 5:
                        print('downloading file {} from url {} failure, check your net ...'.format(fpath, murl))

    def write_file(self):
        if self.mode == 'music':
            file = open(hp.music_fname, 'w')
            file.write('编号' + '\t' + '名称' + '\t' + '类型' + '\n')
            for mid, mposl in self.mdic.items():
                mrename, _, mlab = mposl[:3]
                line = str(mid) + '\t' + mrename[:-4] + '\t' + '|'.join(mlab) + '\n'
                file.write(line)
            file.close()
        elif self.mode == 'labels':
            file1 = open(hp.labs_vacab, 'w')
            file2 = open(hp.label_reverse_fname, 'w')
            file1.write('类型' + '\t' + '编号' + '\n')
            for label, lid in self.label_vacab_dic.items():
                file1.write(label + '\t' + str(lid) + '\n')
            file2.write('编号' + '\t' + '类型' + '\t' + '互斥类型' + '\n')
            for lid, lposl in self.ldic.items():
                ltype, lreverse = lposl[:2]
                file2.write(str(lid) + '\t' + ltype + '\t' + '|'.join(lreverse) + '\n')
            file1.close()
            file2.close()

    def solve_music(self):
        self.read_file()
        self.write_file()
        self.download()
        print('solve music done.')

    def solve_labels(self):
        self.read_file()
        self.write_file()
        print('solve labels done.')

if __name__ == '__main__':
    s1 = solver()
    s1.solve(txt=hp.orilabel_txt, mode='labels')
    s1.solve(txt=hp.orimusic_txt, mode='music')
