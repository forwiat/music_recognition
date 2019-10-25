import codecs
import os
lines = codecs.open('./total_music.csv', 'r').readlines()
new_lines = []                                                                                                          
music_dic = {}
new_lines.append('编号\t名称\t类型')
for wav_name in os.listdir('./music/wav'):
    music_dic[wav_name[:-4]] = 1
for line in lines:
    music_id, music_name, music_type = line.split('\t')
    if music_name[:-4] in music_dic.keys():
        new_lines.append('{}\t{}\t{}'.format(music_id.strip(), music_name[:-4] + '.wav', music_type.strip()))
with open('./music.csv', 'w') as f:
    for i in new_lines:
        f.write(i + '\n')
print('done')
