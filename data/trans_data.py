import os
import codecs
from tqdm import tqdm
total = 1000
lines = codecs.open('./music.csv', 'r').readlines()
wavs_path = './music/wav'
small_wavs_path = './music/small_wav_data'
for i in tqdm(range(1, total + 1)):
    _, music_name, _ = lines[i].strip().split('\t')
    wav_path = os.path.join(wavs_path, music_name)
    new_wav_path = os.path.join(small_wavs_path, music_name)
    os.system('cp -r {} {}'.format(wav_path, new_wav_path))

