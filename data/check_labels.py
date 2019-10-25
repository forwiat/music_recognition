import codecs
wav_label_list = codecs.open('music.csv', 'r').readlines()[1:]
labels_list = codecs.open('ori_labels.csv', 'r').readlines()[1:]
label_dic = {}
for line in labels_list:
    _, label_name, _ = line.strip().split('\t')
    label_dic[label_name] = 1
cnt = 38
new_labels = codecs.open('ori_labels.csv', 'r').readlines()
for line in wav_label_list:
    _, _, labels = line.strip().split('\t')
    labels = labels.split('|')
    for label in labels:
        if label not in label_dic.keys():
            label_dic[label] = 1
            new_labels.append('{}\t{}\tNone'.format(cnt, label))
            cnt += 1
with open('labels.csv', 'w') as f:
    for line in new_labels:
        f.write(line.strip() + '\n')
print('done')
