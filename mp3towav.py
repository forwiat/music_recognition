from pydub import AudioSegment
import os
from tqdm import tqdm
import multiprocessing as mp
mp3_dir = ''
wav_dir = ''

def process(args):
    (mp3_files, mode) = args
    for mp3_name in tqdm(mp3_files):
        wav_name = mp3_name[:-4] + '.wav'
        mp3_path = os.path.join(mp3_dir, mp3_name)
        wav_path = os.path.join(wav_dir, wav_name)
        if mode == 'pydub':
            song = AudioSegment.from_mp3(mp3_path)
            song.export(wav_path, format='wav')
        elif mode == 'ffmpeg':
            os.system('ffmpeg -i {} -ac 1 -ar 22050 {}'.format(mp3_path, wav_path))
        else:
            print('no supported handle wave type {}'.format(mode))
            exit(0)

def handle(mode, multi_cpu=False):
    if not os.path.isdir(mp3_dir):
        print('{} is not a dir.'.format(mp3_dir))
        exit(0)
    if not os.path.isdir(wav_dir):
        os.makedirs(wav_dir)
    mp3_files = os.listdir(mp3_dir)
    if multi_cpu:
        cpu_nums = mp.cpu_count()
        pool = mp.Pool(cpu_nums)
        splits = [(mp3_files[i::cpu_nums],
                   mode)
                  for i in range(cpu_nums)]
        pool.map(process, splits)
        pool.close()
        pool.join()
    else:
        process((mp3_files, mode))

if __name__ == '__main__':
    # only support pydub and ffmpeg at present
    handle(mode='ffmpeg', multi_cpu=True)
