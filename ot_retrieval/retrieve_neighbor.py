import os
import torch
from glob import glob
import heapq
import json

with open('emb_c16_sort.txt', 'r') as f:
    split_data = f.read()
    emb_files = split_data.split('\n')
    
# get loss matrix
print(len(emb_files))
loss_matrix = torch.zeros(len(emb_files), len(emb_files))
loss_all = torch.load('c16_c16_matrix.pt')
for i in range(len(emb_files)):
    for j in range(i+1, len(emb_files)):
        loss_matrix[i][j] = loss_all[i][j-i-1].item()   
for i in range(len(emb_files)):
    for j in range(i):
        loss_matrix[i][j] = loss_matrix[j][i]

# record the index of training data
index_list = []
for i in range(10):
    index_line = []
    feat_path = os.path.join('c16_slide', str(i),'train')
    files = glob(os.path.join(feat_path, '*.pt'))
    for k in range(len(files)):
        for j in range(len(emb_files)):
            slide_id = os.path.basename(files[k])
            if slide_id in emb_files[j]:
                index_line.append(j)
    
    index_list.append(index_line)

# retrieve nearest neighbors
for q in range(10):
    idex = index_list[q]
    print(len(idex))
    loss_dict = {}
    for i in range(len(emb_files)):
        line_loss = loss_matrix[i]
        line_loss_ori = loss_matrix[i].tolist()
        line_loss = [line_loss[k] for k in idex]
        name = os.path.basename(emb_files[i])
        low = heapq.nsmallest(4, line_loss)
        index = []
        for l in low:
            index.append(line_loss_ori.index(l))
        
        neighbor = [emb_files[i] for i in index]
        slide_id = name.replace('.pt','')
        if name == neighbor[0]:
            loss_dict[slide_id] = neighbor[1:4]
        else:
            loss_dict[slide_id] = neighbor[:3]
    print(loss_dict)
    f = open('loss_matrix_1616_20att_' + str(q) +'.json', 'w')
    tmp = json.dumps(loss_dict)
    f.write(tmp)