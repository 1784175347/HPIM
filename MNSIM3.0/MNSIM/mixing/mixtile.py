import sys
import os
import configparser as cp

work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
import numpy as np
import pandas as pd
from MNSIM.Interface.interface import *
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Tile_latency import tile_latency_analysis
from MNSIM.Latency_Model.Pooling_latency import pooling_latency_analysis
from MNSIM.NoC.interconnect_estimation import interconnect_estimation
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Tile import tile
from IPython import embed

def generate_normal_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(row):
        for j in range(column):
            matrix[i][j] = start
            pos[start][0]=i
            pos[start][1]=j
            start += 1
    return matrix,pos

def generate_snake_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(row):
        for j in range(column):
            if i % 2:
                matrix[i][column - j - 1] = start
                pos[start][0]=i
                pos[start][1]=column - j - 1
            else:
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
            start += 1
    return matrix,pos

def generate_hui_matrix(row, column):
    matrix = np.zeros([row, column])
    state = 0
    stride = 1
    step = 0
    start = 0
    dl = 0
    ru = 0
    i = 0
    j = 0
    pos=np.zeros([row*column,2])
    for x in range(row * column):
        if x == 0:
            matrix[i][j] = start
            pos[start][0]=i
            pos[start][1]=j
        else:
            if state == 0:
                j += 1
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
                state = 1
            elif state == 1:
                if dl == 0:
                    i += 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    step += 1
                    if step == stride:
                        dl = 1
                        step = 0
                elif dl == 1:
                    j -= 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    step += 1
                    if step == stride:
                        dl = 0
                        step = 0
                        stride += 1
                        state = 2
            elif state == 2:
                i += 1
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
                state = 3
            elif state == 3:
                if ru == 0:
                    j += 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    step += 1
                    if step == stride:
                        ru = 1
                        step = 0
                elif ru == 1:
                    i -= 1
                    matrix[i][j] = start
                    pos[start][0]=i
                    pos[start][1]=j
                    step += 1
                    if step == stride:
                        ru = 0
                        step = 0
                        stride += 1
                        state = 0
        start += 1
    return matrix,pos

def generate_zigzag_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(row * column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                if j < column - 1:
                    j += 1
                    matrix[i][j] = start
                else:
                    i += 1
                    matrix[i][j] = start
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                matrix[i][j] = start
                step += 1
                if i == row - 1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < row - 1:
                    i += 1
                    matrix[i][j] = start
                else:
                    j += 1
                    matrix[i][j] = start
                state = 3
            elif state == 3:
                j += 1
                i -= 1
                matrix[i][j] = start
                step += 1
                if j == column - 1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        pos[start][0]=i
        pos[start][1]=j
        start += 1
    return matrix,pos

def generate_normal_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_normal_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(int(row/c)):
        for j in range(int(column/c)):
            for m in range(c):
                for n in range(c):
                    matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                    pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                    pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
            start += 1
    return matrix,pos

def generate_snake_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_snake_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(int(row/c)):
        for j in range(int(column/c)):
            for m in range(c):
                for n in range(c):
                    if i % 2:
                        matrix[i*c+m][(int(column/c) - j - 1)*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=(int(column/c) - j - 1)*c+n
                    else:
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
            start += 1
    return matrix,pos

def generate_zigzag_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_zigzag_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(int(row/c) * int(column/c)):
        if x == 0:
            for m in range(c):
                for n in range(c):
                    matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                    pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                    pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
        else:
            if state == 0:
                if j < int(column/c) - 1:
                    j += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                else:
                    i += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                step += 1
                if i == int(row/c) - 1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < int(row/c) - 1:
                    i += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                else:
                    j += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                state = 3
            elif state == 3:
                j += 1
                i -= 1
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                step += 1
                if j == int(column/c) - 1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        start += 1
    return matrix,pos

class mixtile():
    def __init__(self,mix_tile_path):
        tile_config = cp.ConfigParser()
        tile_config.read(mix_tile_path, encoding='UTF-8')
        self.tile_num=list(map(int,tile_config.get('tile','tile_num').split(',')))
        self.topology=int(tile_config.get('tile','topology'))
        self.c=int(tile_config.get('tile','c'))
        self.device_type = [tile_config.get('tile', f'device_type{i}').split(',') for i in range(self.tile_num[0])]
        self.PE_num=[tile_config.get('tile', f'PE_num{i}').split(',') for i in range(self.tile_num[0])]
        for i in range(len(self.PE_num)):
            for j in range(len(self.PE_num[i])):
                self.PE_num[i][j]=int(self.PE_num[i][j])
        self.xbar_size=[tile_config.get('tile', f'xbar_size{i}').split(',') for i in range(self.tile_num[0])]
        for i in range(len(self.xbar_size)):
            for j in range(len(self.xbar_size[i])):
                self.xbar_size[i][j]=int(self.xbar_size[i][j])
        self.PE_group=int(tile_config.get('tile','PE_group'))
        self.layer_mapping=[tile_config.get('tile', f'layer_map_mix{i}').split(',') for i in range(self.tile_num[0])]
        self.auto_layer_mapping=int(tile_config.get('tile',f'auto_layer_mapping'))
        
        #print(self.layer_mapping[5][3])
        #assert 0
    
    def read_tileinfo(self, SimConfig_path=None, TCG_mapping=None,mix_mode=1):
        if mix_mode==2:
            #self.mapping_matrix_gen()  
            assert TCG_mapping!=None,f'should have tileinfo'
            print("jiedian1")
            TCG_mapping.layer_tileinfo[0]['tile_num_mix']=self.tile_num
            TCG_mapping.layer_tileinfo[0]['device_type_mix']=self.device_type
            TCG_mapping.layer_tileinfo[0]['PE_num_mix']=self.PE_num
            TCG_mapping.layer_tileinfo[0]['xbar_size_mix']=self.xbar_size
            TCG_mapping.layer_tileinfo[0]['PE_group_mix']=self.PE_group
            TCG_mapping.layer_tileinfo[0]['layer_mapping_mix']=self.layer_mapping
            count=0
            for layer_id in range(TCG_mapping.layer_num):
                self.max_row_mix=[]
                self.max_column_mix=[]
                
                for i in range(TCG_mapping.layer_tileinfo[0]['tile_num_mix'][0]):
                    for j in range(TCG_mapping.layer_tileinfo[0]['tile_num_mix'][1]):
                        if self.auto_layer_mapping==0:
                            if TCG_mapping.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                                pass
                            elif int(TCG_mapping.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                                TCG_mapping.mapping_result[i][j]=layer_id
                                if TCG_mapping.net[layer_id][0][0]['type']=='conv':
                                    temp_max_row=min((int(self.xbar_size[i][j]) // (int(TCG_mapping.net[layer_id][0][0]['Kernelsize']) ** 2)),
                                int(TCG_mapping.net[layer_id][0][0]['Inputchannel'])) * (int(TCG_mapping.net[layer_id][0][0]['Kernelsize']) ** 2)
                                    temp_max_column=min(int(TCG_mapping.net[layer_id][0][0]['Outputchannel']), int(self.xbar_size[i][j]))
                                elif TCG_mapping.net[layer_id][0][0]['type']=='fc':
                                    temp_max_row=min(int(TCG_mapping.net[layer_id][0][0]['Infeature']), int(self.xbar_size[i][j]))
                                    temp_max_column=min(int(TCG_mapping.net[layer_id][0][0]['Outfeature']), int(self.xbar_size[i][j]))
                                else:
                                    temp_max_row=0
                                    temp_max_column=0
                                self.max_row_mix.append(temp_max_row)
                                self.max_column_mix.append(temp_max_column)
                        else:
                            if TCG_mapping.mapping_result[i][j]==layer_id:
                                if TCG_mapping.net[layer_id][0][0]['type']=='conv':
                                    temp_max_row=(int(self.xbar_size[i][j]))
                                    temp_max_column=(int(self.xbar_size[i][j]))
                                elif TCG_mapping.net[layer_id][0][0]['type']=='fc':
                                    temp_max_row=(int(self.xbar_size[i][j]))
                                    temp_max_column=(int(self.xbar_size[i][j]))
                                else:
                                    temp_max_row=0
                                    temp_max_column=0
                                self.max_row_mix.append(temp_max_row)
                                self.max_column_mix.append(temp_max_column)    
                TCG_mapping.layer_tileinfo[layer_id]["max_row_mix"]=self.max_row_mix
                TCG_mapping.layer_tileinfo[layer_id]["max_column_mix"]=self.max_column_mix
            print("节点1.5")
            #add tile array[][]
            TCG_mapping.tile_list_mix=[]
            TCG_mapping.tile_list_mix = [[[] for _ in range(int(self.tile_num[0]))] for _ in range(int(self.tile_num[0]))]
            TCG_mapping.ADC_num_mix=[]
            TCG_mapping.DAC_num_mix=[]
            for i in range(int(self.tile_num[0])):
                row = [[]] * int(self.tile_num[0])
                
                TCG_mapping.ADC_num_mix.append(row)
                TCG_mapping.DAC_num_mix.append(row)
            for i in range(int(self.tile_num[0])):
                for j in range(int(self.tile_num[0])):
                    #print("每一个都挺慢的",i*self.tile_num[0]+j)
                    temp_tile=tile(SimConfig_path,device_type=self.device_type[i][j],xbar_size=[int(self.xbar_size[i][j]),int(self.xbar_size[i][j])],PE_num=self.PE_num[i][j],mix_mode=2,easy=TCG_mapping.LUT_use)
                    TCG_mapping.tile_list_mix[i][j]=temp_tile
                    TCG_mapping.ADC_num_mix[i][j]=math.ceil(int(self.xbar_size[i][j])/8)
                    TCG_mapping.DAC_num_mix[i][j]=math.ceil(int(self.xbar_size[i][j])/8)
        
        return TCG_mapping
    def mapping_matrix_gen(self):
        if self.topology == 0:
            if self.tile_connection == 0:
                [self.mapping_order,self.pos_mapping_order] = generate_normal_matrix(self.tile_num[0], self.tile_num[1])
            elif self.tile_connection == 1:
                [self.mapping_order,self.pos_mapping_order] = generate_snake_matrix(self.tile_num[0], self.tile_num[1])
            elif self.tile_connection == 2:
                [self.mapping_order,self.pos_mapping_order] = generate_hui_matrix(self.tile_num[0], self.tile_num[1])
            elif self.tile_connection == 3:
                [self.mapping_order,self.pos_mapping_order] = generate_zigzag_matrix(self.tile_num[0], self.tile_num[1])
        elif self.topology == 1:
            if self.tile_connection == 0:
                [self.mapping_order,self.pos_mapping_order] = generate_normal_matrix_cmesh(self.tile_num[0], self.tile_num[1], self.c)
            elif self.tile_connection == 1:
                [self.mapping_order,self.pos_mapping_order] = generate_snake_matrix_cmesh(self.tile_num[0], self.tile_num[1], self.c)
            elif self.tile_connection == 3:
                [self.mapping_order,self.pos_mapping_order] = generate_zigzag_matrix_cmesh(self.tile_num[0], self.tile_num[1], self.c)
