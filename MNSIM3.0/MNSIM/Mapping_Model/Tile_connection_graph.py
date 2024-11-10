#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp

work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Interface.interface import *
import collections
import pandas as pd
from IPython import embed

class PE_node():
    def __init__(self, PE_id=0, ltype='conv', lnum=0):
        # PE_id: the id of PE node, ltype: layer type of this PE, lnum: layer number of this PE
        self.id = PE_id
        self.type = ltype
        self.lnum = lnum
        self.inMerge_list = []
        self.outMerge = 0

    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()

    def set_outMerge(self, Merge_id):
        self.outMerge = Merge_id


class Merge_node():
    def __init__(self, Merge_id=0, mtype=0, lnum=0):
        # Merge_id: the id of Merge node, mtype: merge type (0: add, 1: concat, 2: pooling)
        self.id = Merge_id
        self.type = mtype
        self.lnum = lnum
        self.inPE_list = []
        self.outPE_list = []
        self.inMerge_list = []
        self.outMerge_list = []

    def set_inPE(self, PE_id):
        if PE_id not in self.inPE_list:
            self.inPE_list.append(PE_id)
            self.inPE_list.sort()

    def set_outPE(self, PE_id):
        if PE_id not in self.outPE_list:
            self.outPE_list.append(PE_id)
            self.outPE_list.sort()

    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()

    def set_outMerge(self, Merge_id):
        if Merge_id not in self.outMerge_list:
            self.outMerge_list.append(Merge_id)
            self.outMerge_list.sort()

# The following matrix generations aim to conduct weights mapping on tiles

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


class TCG():
    def __init__(self, NetStruct, SimConfig_path, multiple=None,mix_mode=0,mix_tile=None):
        # NetStruct: layer structure, SimConfig_path: Hardware config path, multiple: allocate more resources for some layers (i.e., duplicate)
        TCG_config = cp.ConfigParser()
        TCG_config.read(SimConfig_path, encoding='UTF-8')
        if multiple is None:
            multiple = [1] * len(NetStruct)
        self.tile = tile(SimConfig_path)
        #linqiushi: need to change tile-->two kinds of tile
        self.mode4posi=[]
        self.mode4posj=[]
        self.mode4injection=[]
        self.mode2posi=[]
        self.mode2posj=[]
        self.mode2injection=[]
        self.net = NetStruct
        self.layer_num = len(self.net)
        self.tile_num = list(map(int, TCG_config.get('Architecture level', 'Tile_Num').split(',')))
        if self.tile_num[0] == 0:
            self.tile_num[0] = 8
            self.tile_num[1] = 8
            
        if mix_tile!=None and mix_tile.tile_num[0]!=0 and mix_tile.tile_num[1]!=0:
            self.tile_num[0]=mix_tile.tile_num[0]
            self.tile_num[1]=mix_tile.tile_num[1]
        assert self.tile_num[0] > 0, "Tile number < 0"
        assert self.tile_num[1] > 0, "Tile number < 0"
        self.tile_total_num = self.tile_num[0] * self.tile_num[1]
        self.mapping_order = -1 * np.ones(self.tile_num)
        self.pos_mapping_order = np.zeros((self.tile_num[0]*self.tile_num[1], 2), dtype=int)
        self.mapping_result = -1 * np.ones(self.tile_num)
        #linqiushi modified
        self.mapping_result_rewrite = [[[] for _ in range(self.tile_num[1])] for _ in range(self.tile_num[0])]
        self.whether_rewrite_layer=1000000
        self.rewrite_time=0
        self.layer_whether_rewrite=[]
        #linqiushi modified
        self.xbar_size_NVM= list(map(int, TCG_config.get('Mixmode3 Configuration', 'xbar_size_NVM').split(',')))
        self.xbar_size_SRAM= list(map(int, TCG_config.get('Mixmode3 Configuration', 'xbar_size_SRAM').split(',')))
        self.PE_num_NVM=list(map(int, TCG_config.get('Mixmode3 Configuration', 'PE_num_NVM').split(',')))
        self.PE_num_SRAM=list(map(int, TCG_config.get('Mixmode3 Configuration', 'PE_num_SRAM').split(',')))
        self.FPS=int(TCG_config.get('Mixmode2/4 Configuration', 'FPS'))
        self.LUT_use=int(TCG_config.get('Mixmode2/4 Configuration', 'LUT_use'))
        print(self.xbar_size_NVM,self.xbar_size_SRAM)
        if mix_mode==3:
            self.total_area_mix3=0    
            self.mixmode3_array = np.zeros((self.tile_num[0], self.tile_num[1]))
            self.mixmode3_xbar_size=np.zeros((self.tile_num[0], self.tile_num[1]))

            count=0
            for i in range(self.tile_num[0]):
                for j in range(self.tile_num[1]):
                    if count<self.tile_total_num/2:
                        self.mixmode3_array[i][j]=1
                    count+=1
            for i in range(self.tile_num[0]):
                for j in range(self.tile_num[1]):
                    if self.mixmode3_array[i][j]==1:
                        self.mixmode3_xbar_size[i][j]=self.xbar_size_SRAM[0]
                        self.total_area_mix3+=(self.mixmode3_xbar_size[i][j])*self.mixmode3_xbar_size[i][j]
                    elif self.mixmode3_array[i][j]==0:
                        self.mixmode3_xbar_size[i][j]=self.xbar_size_NVM[0]
                        self.total_area_mix3+=(self.mixmode3_xbar_size[i][j])*self.mixmode3_xbar_size[i][j]
            self.remain_area=self.total_area_mix3
            #Pim_type: 0 stands for analog; 1 stangs for digital
            #mixmode3_array gives the tile_type information ;the exact information stores in SimConfig.ini
            #the exact mapping strategy will be discussed
            self.tile_array_collection=[[[] for _ in range(self.tile_num[1])] for _ in range(self.tile_num[0])]
            for i in range(self.tile_num[0]):
                for j in range(self.tile_num[1]):
                    # in sequence:size,PE_num
                    self.tile_array_collection.append(self.mixmode3_xbar_size[i][j])
                    if self.mixmode3_array[i][j]==0:
                        self.tile_array_collection.append(self.PE_num_NVM[0])
                    elif self.mixmode3_array[i][j]==1:
                        self.tile_array_collection.append(self.PE_num_SRAM[0])
                    else:
                        assert 0,f'error in mixmode3_array'
                        
            self.mapping_in_size(self.net,self.tile_array_collection)
        self.layer_tileinfo = []
        self.xbar_polarity = int(TCG_config.get('Process element level', 'Xbar_Polarity'))
        self.tile_connection = int(TCG_config.get('Architecture level', 'Tile_Connection'))
        if mix_mode!=1:
            self.mapping_matrix_gen()
            print(self.mapping_order)
        if mix_mode==2:
            self.tile_num=mix_tile.tile_num
            self.auto_layer_mapping=mix_tile.auto_layer_mapping
            #self.mapping_in_size(self.net,mix_tile.xbar_size)
        #linqiushi above
        
        self.mix_mode=mix_mode
        #linqiushi above
        start_tileid = 0
            # the start Tile id
        self.max_inbuf_size = 0
            # the maximum input buffer size of each PE, unit: KB
        self.max_outbuf_size = 0
            # the maximum output buffer size of each tile, unit: KB
        self.global_buf_size = 0
            # the global buffer size for accumulator
        self.global_data_size = 0
        self.global_adder_num = 0
            # the global adder number in accumulator
        self.global_multiplier_num=0
        self.global_adder_bitwidth = 8
        self.global_multiplier_bitwidth=8
        self.rewrite_mode=0 #linqiushi modified: mode=1 means need to rewrite
        self.rewrite_layer_list=None
        #linqiushi modified
        if mix_mode!=0:
            self.mix_mode=mix_mode
        num = []
            # track PE number of each layer 
        total_xbar_num = 0
        mixmode3_tilecount=0
        mixmode2_tilecount=0
        mixmode4_tilecount=0
        for layer_id in range(self.layer_num):
            layer_dict = self.net[layer_id][0][0]
            tmp_tileinfo = collections.OrderedDict()
            layer_type = layer_dict['type']
            if self.xbar_polarity == 1:
                weight_precision = int(layer_dict['Weightbit'])
            else:
                assert self.xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
                weight_precision = int(layer_dict['Weightbit']) - 1
            tmp_tileinfo['startid'] = start_tileid
            input_size = 0
            inputchannel = 0
            outputchannel = 0
            data_inbuf = 0
            data_outbuf = 0
            
                
            if self.mix_mode==3:
                
                if layer_type == 'conv':
                    tmp_tileinfo['type'] = 'conv'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)))
                        # my: PE number in y-axis
                    mixmode3_area_count=0
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_xbar_size=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    i=0
                    j=0
                    
                    while(1):
                        #mapping strategy
                        i=mixmode3_tilecount//self.tile_num[0]
                        j=mixmode3_tilecount-i*self.tile_num[0]  
                        mixmode3_tilecount+=1
                        if i>=self.tile_num[1] or j>=self.tile_num[0]:
                            for m in range(self.tile_num[0]):
                                for n in range(self.tile_num[1]):
                                    self.mapping_result_rewrite[m][n].append(self.mapping_result[m][n])
                            if mixmode3_area_count==0:
                                self.final_layer.append(layer_id-1)
                            else:
                                self.final_layer.append(layer_id)
                            mixmode3_tilecount=0
                            i=0
                            j=0
                            self.rewrite_mode=1
                            if layer_id<self.whether_rewrite_layer:
                                self.whether_rewrite_layer=layer_id
                           
                        #able to change
                        
                        temp_tilenum+=1
                        tile_xbar_size.append(self.mixmode3_xbar_size[i][j])
                        if self.mixmode3_array[i][j]==1:
                            tile_device_type.append('SRAM')
                        elif self.mixmode3_array[i][j]==0:
                            tile_device_type.append('NVM')
                        self.mapping_result[i][j]=layer_id
                        tile_max_row.append(min((self.mixmode3_xbar_size[i][j] // (int(layer_dict['Kernelsize']) ** 2)),int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2))
                        
                        tile_max_column.append(min(int(layer_dict['Outputchannel']), self.mixmode3_xbar_size[i][j]))
                        mixmode3_area_count+=(self.mixmode3_xbar_size[i][j]*self.mixmode3_xbar_size[i][j])*self.tile.tile_PE_total_num
                        
                        print(mixmode3_area_count,self.mixmode3_xbar_size[i][j],self.tile.tile_PE_total_num)
                        if mixmode3_area_count> math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))*math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2)):
                            
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_tilenum*self.tile.tile_PE_total_num
                            break
                    print(math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))*math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2)))
                    tmp_tileinfo['xbar_size']=tile_xbar_size                        
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    tmp_tileinfo['device_type']=tile_device_type
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    tmp_tileinfo['tile_max_column']=tile_max_column
                    tmp_tileinfo['max_row']=tile_max_row[0]
                    tmp_tileinfo['max_column']=tile_max_column[0]
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1

                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                        # assume using the line buffer structure
                    outputchannel = int(layer_dict['Outputchannel'])
                    data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    # buffer_size: unit Byte
                elif layer_type == 'fc':
                    tmp_tileinfo['type'] = 'fc'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                        # my: PE number in y-axis
                    mixmode3_area_count=0
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    while(1):
                        #mapping strategy
                        i=mixmode3_tilecount//self.tile_num[0]
                        j=mixmode3_tilecount-i*self.tile_num[0]  
                        #able to change
                        mixmode3_tilecount+=1
                        if i>=self.tile_num[1] or j>=self.tile_num[0]:
                            for m in range(self.tile_num[0]):
                                for n in range(self.tile_num[1]):
                                    self.mapping_result_rewrite[m][n].append(self.mapping_result[m][n])
                            mixmode3_tilecount=0
                            i=0
                            j=0
                            self.rewrite_mode=1
                            if layer_id<self.whether_rewrite_layer:
                                self.whether_rewrite_layer=layer_id
                        temp_tilenum+=1
                        if self.mixmode3_array[i][j]==1:
                            tile_device_type.append('SRAM')
                        elif self.mixmode3_array[i][j]==0:
                            tile_device_type.append('NVM')
                        tile_xbar_size.append(self.mixmode3_xbar_size[i][j])
                        self.mapping_result[i][j]=layer_id
                        tile_max_row.append(min(self.mixmode3_xbar_size[i][j],int(layer_dict['Infeature'])))
                        
                        tile_max_column.append(min(int(layer_dict['Infeature']),self.mixmode3_xbar_size[i][j] ))
                        mixmode3_area_count+=(self.mixmode3_xbar_size[i][j]*self.mixmode3_xbar_size[i][j])*self.tile.tile_PE_total_num
                        print(mixmode3_area_count,self.mixmode3_xbar_size[i][j],self.tile.tile_PE_total_num)
                        if mixmode3_area_count> math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']))*math.ceil(int(layer_dict['Infeature'])):
                            
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_tilenum*self.tile.tile_PE_total_num
                            break
                    
                    tmp_tileinfo['xbar_size']=tile_xbar_size   
                    tmp_tileinfo['device_type']=tile_device_type                                
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    tmp_tileinfo['tile_max_column']=tile_max_column
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    tmp_tileinfo['max_row']=tile_max_row[0]
                    tmp_tileinfo['max_column']=tile_max_column[0]
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        if (i+layer_id) < self.layer_num:
                            tmp_layer = self.net[i + layer_id][0][0]
                            if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                                tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    input_size = int(layer_dict['Infeature'])
                    inputchannel = 1
                    data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                    # buffer_size: unit Byte
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    
                elif layer_type == 'pooling':
                    tmp_tileinfo['type'] = 'pooling'
                    tmp_tileinfo['mx'] = 1
                    tmp_tileinfo['my'] = 1
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    mixmode3_area_count=0
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    i=mixmode3_tilecount//self.tile_num[0]
                    j=mixmode3_tilecount-i*self.tile_num[0]  
                    #able to change
                    mixmode3_tilecount+=1
                    if i>=self.tile_num[1] or j>=self.tile_num[0]:
                        for m in range(self.tile_num[0]):
                            for n in range(self.tile_num[1]):
                                self.mapping_result_rewrite[m][n].append(self.mapping_result[m][n])
                        mixmode3_tilecount=0
                        i=0
                        j=0
                        self.rewrite_mode=1
                        if layer_id<self.whether_rewrite_layer:
                            self.whether_rewrite_layer=layer_id
                    temp_tilenum+=1
                    if self.mixmode3_array[i][j]==1:
                        tile_device_type.append('SRAM')
                    elif self.mixmode3_array[i][j]==0:
                        tile_device_type.append('NVM')
                    tile_xbar_size.append(self.mixmode3_xbar_size[i][j])
                    self.mapping_result[i][j]=layer_id 
                    tmp_tileinfo['tilenum']=temp_tilenum
                    tmp_tileinfo['PEnum']=temp_tilenum*self.tile.tile_PE_total_num
                        
                    
                    tmp_tileinfo['xbar_size']=tile_xbar_size   
                    tmp_tileinfo['device_type']=tile_device_type                                
                        # max_group: maximum used groups in one PE of this layer
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                    data_outbuf = 0
                        # assume the buffer size depends on the conv/fc layers
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    
                elif layer_type == 'element_sum':

                    tmp_tileinfo['type'] = 'element_sum'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                
                    previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                elif layer_type == 'element_multiply':
                    tmp_tileinfo['type'] = 'element_multiply'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = 0
                    tmp_tileinfo['tile_max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                    #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                if layer_type == 'conv' or layer_type == 'fc':
                    total_xbar_num =mixmode3_tilecount*self.tile.tile_PE_total_num * multiple[layer_id]
                start_tileid += tmp_tileinfo['tilenum']
                num.append(tmp_tileinfo['PEnum'])
                self.layer_tileinfo.append(tmp_tileinfo)
            elif self.mix_mode==2 and mix_tile.auto_layer_mapping==1:
                print("看看几次",layer_id,layer_type)
                if layer_type == 'conv':
                    tmp_tileinfo['type'] = 'conv'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)))
                        # my: PE number in y-axis
                    mixmode2_area_count=0
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_xbar_size=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    temp_PE_num=0
                    #get the tile array
                    
                    remain_area=math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))\
                    *math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2))
                    #TODO:layer mapping to mix
                    while(1):
                        #mapping strategy
                        if layer_id==9:
                            print("jiancha","hi",mixmode2_area_count>math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))*math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2)))
                        i=mixmode2_tilecount//mix_tile.tile_num[0]
                        j=mixmode2_tilecount-i*mix_tile.tile_num[0]  
                        #able to change
                        mixmode2_tilecount+=1
                        temp_tilenum+=1
                        
                        tile_xbar_size.append(mix_tile.xbar_size[i][j])
                        
                        tile_device_type.append(mix_tile.device_type)
                        self.mapping_result[i][j]=layer_id
                        if layer_id==9:
                            print("jiancha","yes",self.mapping_result[i][j],i,j,self.mapping_result[0])
                        tile_max_row.append(min((mix_tile.xbar_size[i][j] // (int(layer_dict['Kernelsize']) ** 2)),int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2))
                        
                        tile_max_column.append(min(int(layer_dict['Outputchannel']), mix_tile.xbar_size[i][j]))
                        mixmode2_area_count+=(mix_tile.xbar_size[i][j]**2)*mix_tile.PE_num[i][j]**2
                        
                        
                        if mixmode2_area_count>math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))*math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2)):
                            if temp_PE_num==0:
                                temp_PE_num=math.ceil((weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))/mix_tile.xbar_size[i][j])\
                                *math.ceil((int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2))/mix_tile.xbar_size[i][j])*multiple[layer_id]
                            else:
                                temp_PE_num+=math.ceil(remain_area/mix_tile.xbar_size[i][j]**2)
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                        temp_PE_num+=mix_tile.PE_num[i][j]**2
                        remain_area-=(mix_tile.xbar_size[i][j]**2)*mix_tile.PE_num[i][j]**2
                    
                    tmp_tileinfo['xbar_size']=tile_xbar_size                        
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    tmp_tileinfo['device_type']=tile_device_type
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    if layer_id==3:
                        print("jiancha",tile_max_row,temp_PE_num,self.mapping_result[0])
                    tmp_tileinfo['tile_max_column']=tile_max_column
                    tmp_tileinfo['max_row']=tile_max_row[0]
                    tmp_tileinfo['max_column']=tile_max_column[0]
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1

                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                        # assume using the line buffer structure
                    outputchannel = int(layer_dict['Outputchannel'])
                    data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                    
                    temp_max_PE=-1 * np.ones(self.tile_num)
                    for i in range(len(temp_max_PE)):
                        for j in range(len(temp_max_PE[i])):
                            if self.mapping_result[i][j]==layer_id:
                                if temp_PE_num>mix_tile.PE_num[i][j]**2:
                                    temp_max_PE[i][j]=mix_tile.PE_num[i][j]**2
                                    temp_PE_num-=mix_tile.PE_num[i][j]**2
                                else:
                                    temp_max_PE[i][j]=temp_PE_num
                                    
                                
                                
                    tmp_tileinfo['max_PE'] = temp_max_PE
                    print("图",layer_id,tmp_tileinfo['max_PE'][0])
                    
                    # buffer_size: unit Byte
                elif layer_type == 'fc':
                    tmp_tileinfo['type'] = 'fc'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                        # my: PE number in y-axis
                    mixmode2_area_count=0
                    tile_max_row=[]
                    tile_max_column=[]
                    remain_area=math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']))\
                        *math.ceil(int(layer_dict['Infeature']))
                    temp_tilenum=0
                    while(1):
                        #mapping strategy
                        i=mixmode2_tilecount//mix_tile.tile_num[0]
                        j=mixmode2_tilecount-i*mix_tile.tile_num[0] 
                        #able to change
                        mixmode2_tilecount+=1
                        temp_tilenum+=1
                        tile_xbar_size.append(mix_tile.xbar_size[i][j])
                        tile_device_type.append(mix_tile.device_type)
                        self.mapping_result[i][j]=layer_id
                        tile_max_row.append(min(mix_tile.xbar_size[i][j],int(layer_dict['Infeature'])))
                        
                        tile_max_column.append(min(int(layer_dict['Infeature']),mix_tile.xbar_size[i][j]))
                        mixmode2_area_count+=(mix_tile.xbar_size[i][j]**2)*mix_tile.PE_num[i][j]**2
                        if mixmode2_area_count> math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']))*math.ceil(int(layer_dict['Infeature'])):
                            if temp_PE_num==0:
                                temp_PE_num=math.ceil((weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']))/mix_tile.xbar_size[i][j])\
                                *math.ceil(int(layer_dict['Infeature'])/mix_tile.xbar_size[i][j])
                            else:
                                temp_PE_num+=math.ceil(remain_area/mix_tile.xbar_size[i][j]**2)
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                        remain_area-=(mix_tile.xbar_size[i][j]**2)*mix_tile.PE_num[i][j]**2
                        temp_PE_num+=mix_tile.PE_num[i][j]**2
                    tmp_tileinfo['xbar_size']=tile_xbar_size   
                    tmp_tileinfo['device_type']=tile_device_type                                
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    tmp_tileinfo['tile_max_column']=tile_max_column
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    tmp_tileinfo['max_row']=tile_max_row[0]
                    tmp_tileinfo['max_column']=tile_max_column[0]
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        if (i+layer_id) < self.layer_num:
                            tmp_layer = self.net[i + layer_id][0][0]
                            if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                                tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    input_size = int(layer_dict['Infeature'])
                    inputchannel = 1
                    data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                    # buffer_size: unit Byte
                    temp_max_PE=-1 * np.ones(self.tile_num)
                    for i in range(len(temp_max_PE)):
                        for j in range(len(temp_max_PE[i])):
                            if self.mapping_result[i][j]==layer_id:
                                if temp_PE_num>=mix_tile.PE_num[i][j]**2:
                                    temp_max_PE[i][j]=mix_tile.PE_num[i][j]**2
                                    temp_PE_num-=mix_tile.PE_num[i][j]**2
                                else:
                                    temp_max_PE[i][j]=temp_PE_num  
                    tmp_tileinfo['max_PE'] = temp_max_PE
                    print("图",layer_id,tmp_tileinfo['max_PE'][0])
                    
                elif layer_type == 'pooling':
                    tmp_tileinfo['type'] = 'pooling'
                    tmp_tileinfo['mx'] = 1
                    tmp_tileinfo['my'] = 1
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                    data_outbuf = 0
                        # assume the buffer size depends on the conv/fc layers
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    i=mixmode2_tilecount//mix_tile.tile_num[0]
                    j=mixmode2_tilecount-i*mix_tile.tile_num[0] 
                    #able to change
                    mixmode2_tilecount+=1
                    self.mapping_result[i][j]=layer_id
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    
                    temp_max_PE=-1 * np.ones(self.tile_num)
                    temp_max_PE[i][j]=tmp_tileinfo['PEnum']
                    tmp_tileinfo['max_PE'] = temp_max_PE
                    print("图",layer_id,tmp_tileinfo['max_PE'][0])
                    
                elif layer_type == 'element_sum':

                    tmp_tileinfo['type'] = 'element_sum'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                
                    previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                elif layer_type == 'element_multiply':
                    tmp_tileinfo['type'] = 'element_multiply'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = 0
                    tmp_tileinfo['tile_max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                    #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                if layer_type == 'conv' or layer_type == 'fc':
                    total_xbar_num =mixmode3_tilecount*self.tile.tile_PE_total_num * multiple[layer_id]
                start_tileid += tmp_tileinfo['tilenum']
                num.append(tmp_tileinfo['PEnum'])
                self.layer_tileinfo.append(tmp_tileinfo)
            elif self.mix_mode==4 and (mix_tile.auto_layer_mapping==1 or mix_tile.auto_layer_mapping==0):
                if layer_type == 'conv':
                    tmp_tileinfo['type'] = 'conv'
                    i=int(self.pos_mapping_order[mixmode4_tilecount][0])
                    j=int(self.pos_mapping_order[mixmode4_tilecount][1])
                    xbar_size4=0
                    xbar_size4=mix_tile.xbar_size[i][j]
                    PE_num4=mix_tile.PE_num[i][j]
                    tmp_tileinfo['device_type']=mix_tile.device_type[i][j]
                    tmp_tileinfo['xbar_size']=xbar_size4
                    tmp_tileinfo['PE_num_tile']=(PE_num4)
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / xbar_size4)
                        # mx: tile number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (xbar_size4 // (int(layer_dict['Kernelsize']) ** 2)))
                        # my: PE number in y-axis
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['max_row'] = min((xbar_size4 // (int(layer_dict['Kernelsize']) ** 2)),
                        int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2)
                        # max_row: maximum used row in one crossbar of this layer
                    tmp_tileinfo['max_column'] = min(int(layer_dict['Outputchannel']), xbar_size4)
                        # max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                        # assume using the line buffer structure
                    outputchannel = int(layer_dict['Outputchannel'])
                    data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                    

                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    num.append(tmp_tileinfo['PEnum'])
                    test_tile_num = math.ceil(tmp_tileinfo['PEnum'] / ((PE_num4)*(PE_num4)))
                    tmp_tileinfo['inLayer_data']=outputchannel*layer_dict['outputbit']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][1]/test_tile_num
                    for k in range(mixmode4_tilecount,mixmode4_tilecount+test_tile_num):
                        i=int(self.pos_mapping_order[k][0])
                        j=int(self.pos_mapping_order[k][1])
                        self.mapping_result[i][j]=layer_id
                        if mix_tile.xbar_size[i][j]!=xbar_size4:
                            assert 0,f'error in the mapping: not same tile in one layer'
                    mixmode4_tilecount+=test_tile_num
                    tmp_tileinfo['tilenum']=test_tile_num
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], ((PE_num4)*(PE_num4)))
                elif layer_type == 'fc':
                    tmp_tileinfo['type'] = 'fc'
                    i=int(self.pos_mapping_order[mixmode4_tilecount][0])
                    j=int(self.pos_mapping_order[mixmode4_tilecount][1])
                    xbar_size4=0
                    xbar_size4=mix_tile.xbar_size[i][j]
                    tmp_tileinfo['device_type']=mix_tile.device_type[i][j]
                    PE_num4=mix_tile.PE_num[i][j]
                    tmp_tileinfo['xbar_size']=xbar_size4
                    tmp_tileinfo['PE_num_tile']=(PE_num4)
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / xbar_size4)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / xbar_size4)
                        # my: PE number in y-axis
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['max_row'] = min(int(layer_dict['Infeature']), xbar_size4)
                        # max_row: maximum used row in one crossbar of this layer
                    tmp_tileinfo['max_column'] = min(int(layer_dict['Outfeature']), xbar_size4)
                        # max_row: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        if (i+layer_id) < self.layer_num:
                            tmp_layer = self.net[i + layer_id][0][0]
                            if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                                tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    input_size = int(layer_dict['Infeature'])
                    inputchannel = 1
                    data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    num.append(tmp_tileinfo['PEnum'])
                    test_tile_num = math.ceil(tmp_tileinfo['PEnum'] / ((PE_num4)*(PE_num4)))
                    tmp_tileinfo['inLayer_data']=int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/test_tile_num
                    for k in range(mixmode4_tilecount,mixmode4_tilecount+test_tile_num):
                        i=int(self.pos_mapping_order[k][0])
                        j=int(self.pos_mapping_order[k][1])
                        self.mapping_result[i][j]=layer_id
                        if mix_tile.xbar_size[i][j]!=xbar_size4:
                            assert 0,f'error in the mapping: not same tile in one layer'
                    mixmode4_tilecount+=test_tile_num
                    tmp_tileinfo['tilenum']=test_tile_num
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], ((PE_num4)*(PE_num4)))
                elif layer_type == 'pooling':
                    tmp_tileinfo['type'] = 'pooling'
                    tmp_tileinfo['mx'] = 1
                    tmp_tileinfo['my'] = 1
                    i=int(self.pos_mapping_order[mixmode4_tilecount][0])
                    j=int(self.pos_mapping_order[mixmode4_tilecount][1])
                    tmp_tileinfo['device_type']=mix_tile.device_type[i][j]
                    PE_num4=mix_tile.PE_num[i][j]
                    self.mapping_result[i][j]=layer_id
                    mixmode4_tilecount+=1
                    tmp_tileinfo['xbar_size']=xbar_size4
                    tmp_tileinfo['PE_num_tile']=(PE_num4)
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                    data_outbuf = 0
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    num.append(tmp_tileinfo['PEnum'])
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    start_tileid += tmp_tileinfo['tilenum']
                    tmp_tileinfo['inLayer_data']=layer_dict['Outputchannel']*layer_dict['outputbit']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][1]/1
                elif layer_type == 'element_sum':
                    tmp_tileinfo['type'] = 'element_sum'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                
                    previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    num.append(tmp_tileinfo['PEnum'])
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    tmp_tileinfo['inLayer_data']=self.layer_tileinfo[layer_id-1]['inLayer_data']
                    tmp_tileinfo['xbar_size']=0
                    tmp_tileinfo['device_type']='NVM'
                    tmp_tileinfo['PE_num_tile']=0
                    start_tileid += tmp_tileinfo['tilenum']
                elif layer_type == 'element_multiply':
                    tmp_tileinfo['type'] = 'element_multiply'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                    #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    num.append(tmp_tileinfo['PEnum'])
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    start_tileid += tmp_tileinfo['tilenum']
                if layer_type == 'conv' or layer_type == 'fc':
                    total_xbar_num =mixmode3_tilecount*self.tile.tile_PE_total_num * multiple[layer_id]
                start_tileid += tmp_tileinfo['tilenum']
                num.append(tmp_tileinfo['PEnum'])
                self.layer_tileinfo.append(tmp_tileinfo)
            elif self.mix_mode==2 and mix_tile.auto_layer_mapping==0:
                print("看看几次",layer_id,layer_type)
                if layer_type == 'conv':
                    tmp_tileinfo['type'] = 'conv'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)))
                        # my: PE number in y-axis
                    
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_xbar_size=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    temp_PE_num=0
                    #get the tile array
                    all_cap=0
                    while(1):
                        #through the tile
                        i=int(self.pos_mapping_order[mixmode2_tilecount][0])
                        j=int(self.pos_mapping_order[mixmode2_tilecount][1])
                        #able to change
                        if mixmode2_tilecount>=mix_tile.tile_num[0]*mix_tile.tile_num[0]-1:
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                        if mix_tile.layer_mapping[i][j]=='no':
                            mixmode2_tilecount+=1
                            pass
                        elif int(mix_tile.layer_mapping[i][j])==layer_id:
                            all_cap+=(mix_tile.PE_num[i][j]**2)*(mix_tile.xbar_size[i][j]**2)
                            mixmode2_tilecount+=1
                            temp_tilenum+=1
                            temp_PE_num+=mix_tile.PE_num[i][j]**2
                            tile_xbar_size.append(mix_tile.xbar_size[i][j])
                            tile_device_type.append(mix_tile.device_type)
                            self.mapping_result[i][j]=layer_id
                            tile_max_row.append(min((mix_tile.xbar_size[i][j] // (int(layer_dict['Kernelsize']) ** 2)),int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2))
                            tile_max_column.append(min(int(layer_dict['Outputchannel']), mix_tile.xbar_size[i][j]))
                        elif int(mix_tile.layer_mapping[i][j])!=layer_id:
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                    tmp_tileinfo['all_cap']=all_cap            
                    tmp_tileinfo['xbar_size']=tile_xbar_size                        
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    tmp_tileinfo['device_type']=tile_device_type
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    tmp_tileinfo['tile_max_column']=tile_max_column
                    tmp_tileinfo['max_row']=tile_max_row[0]
                    tmp_tileinfo['max_column']=tile_max_column[0]
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1

                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                    tmp_tileinfo['inLayer_data']=outputchannel*layer_dict['outputbit']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][1]/all_cap
                    temp_max_PE=-1 * np.ones(self.tile_num)

                    for i in range(len(temp_max_PE)):
                        for j in range(len(temp_max_PE[i])):
                            if self.mapping_result[i][j]==layer_id:
                                temp_max_PE[i][j]=min(temp_PE_num,mix_tile.PE_num[i][j]**2)

                    tmp_tileinfo['max_PE'] = temp_max_PE
                elif layer_type == 'fc':
                    tmp_tileinfo['type'] = 'fc'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                        # my: PE number in y-axis
                    tile_max_row=[]
                    tile_max_column=[]
                    tile_xbar_size=[]
                    tile_device_type=[]
                    temp_tilenum=0
                    temp_PE_num=0
                    all_cap=0
                    #get the tile array
                    while(1):
                        #through the tile
                        i=int(self.pos_mapping_order[mixmode2_tilecount][0])
                        j=int(self.pos_mapping_order[mixmode2_tilecount][1])
                        #able to change
                        if mixmode2_tilecount>=mix_tile.tile_num[0]*mix_tile.tile_num[0]-1:
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                        if mix_tile.layer_mapping[i][j]=='no':
                            mixmode2_tilecount+=1
                            pass
                        elif int(mix_tile.layer_mapping[i][j])==layer_id:
                            all_cap+=(mix_tile.PE_num[i][j]**2)*(mix_tile.xbar_size[i][j]**2)
                            mixmode2_tilecount+=1
                            temp_tilenum+=1
                            temp_PE_num+=mix_tile.PE_num[i][j]**2
                            tile_xbar_size.append(mix_tile.xbar_size[i][j])
                            tile_device_type.append(mix_tile.device_type)
                            self.mapping_result[i][j]=layer_id
                            tile_max_row.append(min(mix_tile.xbar_size[i][j],int(layer_dict['Infeature'])))
                            tile_max_column.append(min(int(layer_dict['Infeature']),mix_tile.xbar_size[i][j]))
                        elif int(mix_tile.layer_mapping[i][j])!=layer_id:
                            tmp_tileinfo['tilenum']=temp_tilenum
                            tmp_tileinfo['PEnum']=temp_PE_num
                            break
                    tmp_tileinfo['all_cap']=all_cap 
                    tmp_tileinfo['xbar_size']=tile_xbar_size                        
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    tmp_tileinfo['device_type']=tile_device_type
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['tile_max_row']=tile_max_row
                    tmp_tileinfo['tile_max_column']=tile_max_column
                    tmp_tileinfo['max_row']=max(tile_max_row)
                    tmp_tileinfo['max_column']=max(tile_max_column)
                        # tile_max_row: maximum used row in one crossbar of this layer
                        # tile_max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1

                    input_size = int(layer_dict['Infeature'])
                    inputchannel = 1
                    data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                    tmp_tileinfo['inLayer_data']=int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/all_cap
                    temp_max_PE=-1 * np.ones(self.tile_num)
                    for i in range(len(temp_max_PE)):
                        for j in range(len(temp_max_PE[i])):
                            if self.mapping_result[i][j]==layer_id:
                                temp_max_PE[i][j]=min(temp_PE_num,mix_tile.PE_num[i][j]**2)
                                    
                                
                                
                    tmp_tileinfo['max_PE'] = temp_max_PE
                elif layer_type == 'pooling':
                    tmp_tileinfo['type'] = 'pooling'
                    tmp_tileinfo['mx'] = 1
                    tmp_tileinfo['my'] = 1
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                    data_outbuf = 0
                        # assume the buffer size depends on the conv/fc layers
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    i=int(self.pos_mapping_order[mixmode2_tilecount][0])
                    j=int(self.pos_mapping_order[mixmode2_tilecount][1])
                    #able to change
                    mixmode2_tilecount+=1
                    tmp_tileinfo['all_cap']=(mix_tile.PE_num[i][j]**2)*(mix_tile.xbar_size[i][j]**2)
                    self.mapping_result[i][j]=layer_id
                    tmp_tileinfo['tilenum'] = 1
                    tmp_tileinfo['inLayer_data']=layer_dict['Outputchannel']*layer_dict['outputbit']*layer_dict['Outputsize'][0]*layer_dict['Outputsize'][1]/tmp_tileinfo['all_cap']
                    temp_max_PE=-1 * np.ones(self.tile_num)
                    temp_max_PE[i][j]=1
                    tmp_tileinfo['max_PE'] = temp_max_PE
                    print("图",layer_id,tmp_tileinfo['max_PE'][0])
                    
                elif layer_type == 'element_sum':

                    tmp_tileinfo['type'] = 'element_sum'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = [0]
                    tmp_tileinfo['tile_max_column'] = [0]
                    tmp_tileinfo['max_row']=0
                    tmp_tileinfo['max_column']=0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                
                    previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                    tmp_tileinfo['inLayer_data']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['all_cap']
                    tmp_tileinfo['all_cap']=1
                elif layer_type == 'element_multiply':
                    tmp_tileinfo['type'] = 'element_multiply'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['tile_max_row'] = 0
                    tmp_tileinfo['tile_max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                    #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                    tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                    
                    tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                    tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                if layer_type == 'conv' or layer_type == 'fc':
                    total_xbar_num =mixmode3_tilecount*self.tile.tile_PE_total_num * multiple[layer_id]
                start_tileid += tmp_tileinfo['tilenum']
                num.append(tmp_tileinfo['PEnum'])
                self.layer_tileinfo.append(tmp_tileinfo)
            elif self.mix_mode==1 :
                if layer_type == 'conv':
                    tmp_tileinfo['type'] = 'conv'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)))
                        # my: PE number in y-axis
                    
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['max_row'] = min((self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)),
                        int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2)
                        # max_row: maximum used row in one crossbar of this layer
                    tmp_tileinfo['max_column'] = min(int(layer_dict['Outputchannel']), self.tile.xbar_column)
                        # max_column: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                        # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                        # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i+layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1

                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                        # assume using the line buffer structure
                    outputchannel = int(layer_dict['Outputchannel'])
                    data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                    # buffer_size: unit Byte
                elif layer_type == 'fc':
                    tmp_tileinfo['type'] = 'fc'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                        # my: PE number in y-axis
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['max_row'] = min(int(layer_dict['Infeature']), self.tile.xbar_row)
                        # max_row: maximum used row in one crossbar of this layer
                    tmp_tileinfo['max_column'] = min(int(layer_dict['Outfeature']), self.tile.xbar_column)
                        # max_row: maximum used column in one crossbar of this layer
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        if (i+layer_id) < self.layer_num:
                            tmp_layer = self.net[i + layer_id][0][0]
                            if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                                tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    input_size = int(layer_dict['Infeature'])
                    inputchannel = 1
                    tmp_tileinfo['inLayer_data']=tmp_tileinfo['max_column']*layer_dict['outputbit']*layer_dict['Outputsize']
                    data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                    # buffer_size: unit Byte
                elif layer_type =='MM':
                    tmp_tileinfo['type']='MM'
                    tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                        # mx: PE number in x-axis
                    tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                        # my: PE number in y-axis
                    tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                        # max_group: maximum used groups in one PE of this layer
                    tmp_tileinfo['max_row'] = min(int(layer_dict['Infeature']), self.tile.xbar_row)
                        # max_row: maximum used row in one crossbar of this layer
                    tmp_tileinfo['max_column'] = min(int(layer_dict['Outfeature']), self.tile.xbar_column)
                        # max_row: maximum used column in one crossbar of this layer

                    tmp_tileinfo['input1_size']=layer_dict['input1_size']
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                        # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                        # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        if (i+layer_id) < self.layer_num:
                            tmp_layer = self.net[i + layer_id][0][0]
                            if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                                tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    data_inbuf = layer_dict['Inputsize'] * int(layer_dict['Inputbit'])/8
                    data_outbuf = int(layer_dict['Outputsize']) * int(layer_dict['outputbit'])/8
                    # buffer_size: unit Byte
                elif layer_type == 'pooling':
                    tmp_tileinfo['type'] = 'pooling'
                    tmp_tileinfo['mx'] = 1
                    tmp_tileinfo['my'] = 1
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Inputindex' not in layer_dict.keys():
                        tmp_tileinfo['Inputindex'] = [-1]
                    else:
                        tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                    tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                    for i in tmp_tileinfo['Outputindex']:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                            tmp_tileinfo['is_branchout'] = -1
                    input_size_list = list(map(int, layer_dict['Inputsize']))
                    input_size = input_size_list[0] * input_size_list[1]
                    inputchannel = int(layer_dict['Inputchannel'])
                    data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                    data_outbuf = 0
                        # assume the buffer size depends on the conv/fc layers
                elif layer_type == 'element_sum':

                    tmp_tileinfo['type'] = 'element_sum'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                
                    previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                elif layer_type == 'element_multiply':
                    tmp_tileinfo['type'] = 'element_multiply'
                    tmp_tileinfo['mx'] = 0
                    tmp_tileinfo['my'] = 0
                    tmp_tileinfo['max_row'] = 0
                    tmp_tileinfo['max_column'] = 0
                    tmp_tileinfo['max_group'] = 0
                    if 'Outputindex' not in layer_dict.keys():
                        tmp_tileinfo['Outputindex'] = [1]
                    else:
                        tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                    if len(tmp_tileinfo['Outputindex']) == 1:
                        tmp_tileinfo['is_branchin'] = -1
                    else:
                        tmp_tileinfo['is_branchin'] = 1
                    tmp_tileinfo['is_branchout'] = -1
                    # is_branchin: if this layer is the input layer of a branch
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    tmp_tileinfo['Inputindex'] = Inputindex_list
                    assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx+1
                        previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                    #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                    tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                        # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                    tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                        # the data precision of each branch output (bit)
                    data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                        # unit: Byte
                    self.global_data_size = self.global_data_size + data_size
                    self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                        # unit: KB
                    self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                    if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                        self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
                if layer_type == 'conv' or layer_type == 'fc' or layer_type == 'MM':
                    total_xbar_num += tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
                num.append(tmp_tileinfo['PEnum'])
                tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
                tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
                start_tileid += tmp_tileinfo['tilenum']
                print("jiancha",start_tileid,tmp_tileinfo['tilenum'])
                self.layer_tileinfo.append(tmp_tileinfo)

            inputbit = int(layer_dict['Inputbit'])
            if tmp_tileinfo['type'] == 'conv' or tmp_tileinfo['type'] == 'fc':
                tmp_inbuf_size = math.pow(2,math.ceil(math.log(data_inbuf / tmp_tileinfo['PEnum'],2)))/1024
                tmp_outbuf_size = math.pow(2,math.ceil(math.log(data_outbuf*2 / tmp_tileinfo['tilenum'],2)))/1024 # 2: ping-pong
            else:
                tmp_inbuf_size = 0
                tmp_outbuf_size = 0
            # unit: KB, restricted in 2^M KB
            if tmp_inbuf_size > self.max_inbuf_size:
                self.max_inbuf_size = tmp_inbuf_size
            if tmp_outbuf_size > self.max_outbuf_size:
                self.max_outbuf_size = tmp_outbuf_size
        if self.mix_mode==4 :
            for layer_id in range(self.layer_num):
                if layer_id!=0:
                    layer_dict=self.net[layer_id][0][0]
                    if self.layer_tileinfo[layer_id]['type']=='conv':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['tilenum']/self.layer_tileinfo[layer_id]['tilenum']
                    elif self.layer_tileinfo[layer_id]['type']=='fc':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['tilenum']/self.layer_tileinfo[layer_id]['tilenum']
                    elif self.layer_tileinfo[layer_id]['type']=='pooling':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['tilenum']/self.layer_tileinfo[layer_id]['tilenum']
        elif (self.mix_mode==2 and self.auto_layer_mapping==0):
            for layer_id in range(self.layer_num):
                if layer_id!=0:
                    layer_dict=self.net[layer_id][0][0]
                    if self.layer_tileinfo[layer_id]['type']=='conv':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['all_cap']/self.layer_tileinfo[layer_id]['all_cap']
                    elif self.layer_tileinfo[layer_id]['type']=='fc':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['all_cap']/self.layer_tileinfo[layer_id]['all_cap']
                    elif self.layer_tileinfo[layer_id]['type']=='pooling':
                        self.layer_tileinfo[layer_id]['transLayer_data_before']=self.layer_tileinfo[layer_id-1]['inLayer_data']*self.layer_tileinfo[layer_id-1]['all_cap']/self.layer_tileinfo[layer_id]['all_cap']
        #linqiushi modified
        #assert self.used_tile_num <= self.tile_total_num, "Tile number is not enough"
            # TODO: update weight rewrite in xbar
        self.used_tile_num = start_tileid
        print("wokan",self.used_tile_num,self.tile_total_num)
        if self.used_tile_num>self.tile_total_num and mix_mode==1:
            print("tile_number is not enough, starts to rewrite")
            self.rewrite_mode=2
            # layer_tileinfo record the information
            self.rewrite_mapping_new()
            #assert 0
        #linqiushi above
        print("Total crossbar number:", total_xbar_num)
        self.inLayer_distance = np.zeros([1, self.layer_num])
        self.transLayer_distance = np.zeros([1, self.layer_num])
        self.aggregate_arg = np.zeros([self.layer_num, 2])
    def mapping_in_size(self,net,tile_array,mapping_order):
        print('hi this is mapping_in_size')
        layer_num=len(net)
        tile_count=0
        capacity_layer=0
        capacity_count=0
        #tile_array:第一个元素是xbar_size,第二个元素是PE_num,都默认是正方形
        self.tile_num[0]
        for i in range(layer_num):
            layer_dict=net[layer_id][0][0]
            #初始化信息
            if self.xbar_polarity == 1:
                weight_precision = int(layer_dict['Weightbit'])
            else:
                assert self.xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
                weight_precision = int(layer_dict['Weightbit']) - 1
            #构建大矩形的长和宽
            if layer['type']=='conv':
                Lx = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']))
                #length in x       
                Ly = math.ceil(int(layer_dict['Inputchannel'])*(int(layer_dict['Kernelsize']) ** 2))
                #length in y
                #开始根据mapping order读入tile，先直到tile的总容量大于layer所需的容量
                capacity_layer=Lx*Ly
                #while(capacity_count<capacity_layer):
                    #读入tile
                    
                    
        
        

    def rewrite_mapping(self):
        #the easiest mode: sequential, no copy
        #start_layer: record the first layer in every mapping
        #final_layer: record the final layer in every mapping
        #there are multiple times of mapping in rewriting
        #every_tile_occupy: record the used_tile_num in every mapping
        #layer_whether_rewrite: the first write layers will not be counted
        self.start_layer=[]
        self.final_layer=[]
        self.every_tile_occupy=[]
        temp_tile_count=0
        temp_start_layer=0
        flag=0
        for i in range(self.layer_num):
            if self.layer_tileinfo[i]['tilenum']>self.tile_total_num and flag==0:
                print("zheshi",i,self.layer_tileinfo[i]['type'],self.layer_tileinfo[i]['mx'],self.layer_tileinfo[i]['my'],\
                      self.layer_tileinfo[i]['PEnum'],self.layer_tileinfo[i]['tilenum'],self.tile_total_num)
                
                temp_tile_count=self.layer_tileinfo[i]['tilenum']
                while(1):
                    if temp_tile_count>self.tile_total_num:
                        temp_start_layer=i
                        self.start_layer.append(temp_start_layer)
                        self.final_layer.append(i)
                        self.every_tile_occupy.append(self.tile_total_num)
                        temp_tile_count-=self.tile_total_num
                    else:
                        temp_start_layer=i
                        flag=1
                        if i==self.layer_num-1:
                            self.start_layer.append(temp_start_layer)
                            self.final_layer.append(i)
                            self.every_tile_occupy.append(temp_tile_count)
                        if i<self.layer_num-1:
                            if (temp_tile_count+self.layer_tileinfo[i+1]['tilenum'])>self.tile_total_num:
                                self.start_layer.append(temp_start_layer)
                                self.final_layer.append(i)
                                temp_start_layer=i+1
                                self.every_tile_occupy.append(temp_tile_count)
                                temp_tile_count=0
                        break
                print(self.start_layer)
                print(self.final_layer)
                print(self.every_tile_occupy)
                
            else:
                flag=0
                temp_tile_count+=self.layer_tileinfo[i]['tilenum']
                print("single cycle tile_count",temp_tile_count)
                if i==6:
                    print("info6",self.layer_tileinfo[i]['tilenum'],self.layer_tileinfo[i]['type'])
                if i<self.layer_num-1:
                    if (temp_tile_count+self.layer_tileinfo[i+1]['tilenum'])>self.tile_total_num and temp_tile_count<=self.tile_total_num:
                        self.start_layer.append(temp_start_layer)
                        self.final_layer.append(i)
                        temp_start_layer=i+1
                        self.every_tile_occupy.append(temp_tile_count)
                        temp_tile_count=0
                if i==self.layer_num-1:
                    self.start_layer.append(temp_start_layer)
                    self.final_layer.append(i)
                    self.every_tile_occupy.append(temp_tile_count)
        
        print(self.start_layer)
        print(self.final_layer)
        print(self.every_tile_occupy)
        self.rewrite_time=len(self.start_layer)
        # for k in range(self.rewrite_time):
        #                 # k: the kth write time
        #     for layer_id in range(self.start_layer[k],self.final_layer[k]+1):
        #         if k!=0:
        #             self.layer_whether_rewrite.append(1)
        #         else:
        #             self.layer_whether_rewrite.append(0)
        for layer_id in range(self.layer_num):
            if layer_id>self.final_layer[0]:
                self.layer_whether_rewrite.append(1)
            else:
                self.layer_whether_rewrite.append(0)
        
        print(self.layer_whether_rewrite)
    
    def rewrite_mapping_new(self):
        #option to copy some layer
        #option to put layers into static xbars
        #option to set up the layer_order
        #recording: records all layer Num in single rewrite cycle
        #default: set to use up 100% xbars
        #a different mode: to arbitaryly set layers, the tile used to calculate the layers(only the check of the addup of tiles,not includes the logical computing order)
        self.rewrite_layer_list=[]
        self.layer_tile_num=[]
        self.rewrite_tile_num_layer=[]
        self.layer_copy=[1]*self.layer_num
        # 1: no copy 
        # i(int) >1 : the copy rate
        self.layer_copy[0]=2
        self.layer_whether_static=[0]*self.layer_num
        # 0: no static xbar
        # 1: set to the static xbar
        # the Wq,Wk,Wv is able to be static,but the MM of V is unable
        self.set_layer_order=False 
        self.layer_order=[0]*self.layer_num
        # the set of the computing order
        for i in range(self.layer_num):
            assert self.layer_copy[i]>0
            self.layer_tile_num.append(self.layer_tileinfo[i]['tilenum']*self.layer_copy[i])
            # after the copy
        tile_remain=self.tile_total_num
        if (self.set_layer_order==False):
            temp_layer_list=[]
            temp_tilenum=[]
            for i in range(len(self.layer_tile_num)):
                # len(layer_tile_num)=len(self.layer_num)
                if self.layer_whether_static[i]==0:
                    if tile_remain<self.layer_tile_num[i]:
                        # need to rewrite
                        temp_layer_list.append(i)
                        temp_tilenum.append(tile_remain)
                        self.layer_tile_num[i]-=tile_remain
                        tile_remain=self.tile_total_num
                        self.rewrite_layer_list.append(temp_layer_list)
                        self.rewrite_tile_num_layer.append(temp_tilenum)
                        temp_layer_list=[]
                        temp_tilenum=[]
                        while(tile_remain<self.layer_tile_num[i]):
                            temp_layer_list.append(i)
                            temp_tilenum.append(tile_remain)
                            self.rewrite_layer_list.append(temp_layer_list)
                            self.rewrite_tile_num_layer.append(temp_tilenum)
                            temp_layer_list=[]
                            temp_tilenum=[]
                            tile_remain=self.tile_total_num
                            self.layer_tile_num[i]-=tile_remain
                        assert self.layer_tile_num[i]>0
                        if self.layer_tile_num[i]==tile_remain:
                            temp_layer_list.append(i)
                            temp_tilenum.append(tile_remain)
                            self.rewrite_layer_list.append(temp_layer_list)
                            self.rewrite_tile_num_layer.append(temp_tilenum)
                            temp_layer_list=[]
                            temp_tilenum=[]
                            self.layer_tile_num[i]-=tile_remain
                            tile_remain=self.tile_total_num
                        elif self.layer_tile_num[i]<tile_remain:
                            temp_layer_list.append(i)
                            temp_tilenum.append(self.layer_tile_num[i])
                            tile_remain-=self.layer_tile_num[i]
                        else:
                            assert 0
                    elif tile_remain==self.layer_tile_num[i]:
                        temp_layer_list.append(i)
                        temp_tilenum.append(tile_remain)
                        self.rewrite_layer_list.append(temp_layer_list)
                        self.rewrite_tile_num_layer.append(temp_tilenum)
                        temp_layer_list=[]
                        temp_tilenum=[]
                        tile_remain=self.tile_total_num
                    else:
                        temp_layer_list.append(i)
                        temp_tilenum.append(self.layer_tile_num[i])
                        tile_remain-=self.layer_tile_num[i]
            if tile_remain<self.tile_total_num:
                self.rewrite_layer_list.append(temp_layer_list)
                self.rewrite_tile_num_layer.append(temp_tilenum)
                temp_layer_list=[]
                temp_tilenum=[]
        self.rewrite_time=len(self.rewrite_layer_list)
        assert self.rewrite_time==len(self.rewrite_tile_num_layer)
        print("letmecheck",self.rewrite_layer_list,self.rewrite_tile_num_layer)
    def mapping_matrix_gen(self):
        if self.tile_connection == 0:
            [self.mapping_order,self.pos_mapping_order] = generate_normal_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 1:
            [self.mapping_order,self.pos_mapping_order] = generate_snake_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 2:
            [self.mapping_order,self.pos_mapping_order] = generate_hui_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 3:
            [self.mapping_order,self.pos_mapping_order] = generate_zigzag_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])

    def mapping_net(self):
        self.mapping_matrix_gen()
        
        #linqiushi modified
        if self.rewrite_mode==0:
            for i in range(self.mapping_order.shape[0]):
                for j in range(self.mapping_order.shape[1]):
                    if self.mapping_order[i][j] < self.used_tile_num:
                        for layer_id in range(self.layer_num - 1):
                            if self.layer_tileinfo[layer_id]['type'] in ['conv','pooling','fc']:
                                # only allocate tile for conv layers, pooling layers, and fc layers
                                
                                if ((self.mapping_order[i][j] >= self.layer_tileinfo[layer_id]['startid']) &
                                        (self.mapping_order[i][j] < self.layer_tileinfo[layer_id + 1]['startid'])):
                                    self.mapping_result[i][j] = layer_id
                                    break
                                elif self.mapping_order[i][j] >= self.layer_tileinfo[self.layer_num - 1]['startid']:
                                    self.mapping_result[i][j] = self.layer_num - 1
        elif self.rewrite_mode==1:
            flag=0
            for k in range(self.rewrite_time):
                # k: the kth write time
                flag=0
                for i in range(self.mapping_order.shape[0]):
                    for j in range(self.mapping_order.shape[1]):
                        if self.mapping_order[i][j]<self.every_tile_occupy[k]:
                            for layer_id in range(self.start_layer[k],self.final_layer[k]):
                                    
                                if((self.mapping_order[i][j]>=(self.layer_tileinfo[layer_id]['startid']-self.layer_tileinfo[self.start_layer[k]]['startid']))&
                                (self.mapping_order[i][j]<(self.layer_tileinfo[layer_id+1]['startid']-self.layer_tileinfo[self.start_layer[k]]['startid']))):
                                    self.mapping_result_rewrite[i][j].append(layer_id)
                                    break
                                    
                                elif self.mapping_order[i][j]>=(self.layer_tileinfo[self.final_layer[k]]['startid']-self.layer_tileinfo[self.start_layer[k]]['startid']):
                                    self.mapping_result_rewrite[i][j].append(self.final_layer[k])
                                    break
                            if self.start_layer[k]==self.final_layer[k]:    
                                self.mapping_result_rewrite[i][j].append(self.start_layer[k])
                                if flag==0:
                                    self.layer_tileinfo[self.start_layer[k]]['startid']+=self.every_tile_occupy[k]
                                    flag=1
                        else:
                            self.mapping_result_rewrite[i][j].append(-1)
                            #-1: no layer on it in this rewrite cycle
            #check
            for i in range(self.mapping_order.shape[0]):
                for j in range(self.mapping_order.shape[1]):
                    if len(self.mapping_result_rewrite[i][j])!=self.rewrite_time:
                        print(self.rewrite_time)
                        assert 0,f'rewrite cycle should be equal'
        elif self.rewrite_mode==2:
            for k in range(self.rewrite_time):
                count=0
                while(count<self.tile_total_num):
                    i=int(self.pos_mapping_order[count][0])
                    j=int(self.pos_mapping_order[count][1])
                    add=0
                    for m in range(len(self.rewrite_tile_num_layer[k])):
                        add+=self.rewrite_tile_num_layer[k][m]
                        if add>count:
                            #这里要改，改成count，改成两者之间
                            layer_num=int(self.rewrite_layer_list[k][m])
                            self.mapping_result_rewrite[i][j].append(layer_num)
                            break
                    count+=1
            print("wokanyixia",self.mapping_result_rewrite)
                        
    def calculate_transfer_distance(self):
        if self.rewrite_mode==0:
            for layer_id in range(self.layer_num - 1):
                # Determine the aggregate node for layer 0~N-1
                if self.layer_tileinfo[layer_id]['is_branchout'] == 1:
                    # for the layer which is a output layer of one branch and the next layer is element_sum
                    if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                        src_pos = np.argwhere(self.mapping_result == layer_id)
                        
                        if len(src_pos) == 1:
                            self.inLayer_distance[0][layer_id] = 0
                            self.aggregate_arg[layer_id] = src_pos[0]
                            self.transLayer_distance[0][layer_id] = abs(src_pos[0][0]-1/2*self.tile_num[0]) + src_pos[0][1]
                        else:
                            mindis_total = 1000
                            for A in range(len(src_pos)):
                                tmp_transLayer_distance = abs(src_pos[A][0]-1/2*self.tile_num[0]) + src_pos[A][1]
                                maxdis_in = 0
                                for i in range(len(src_pos)):
                                    if i != A:
                                        dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                        if dis_in > maxdis_in:
                                            maxdis_in = dis_in
                                if (maxdis_in+tmp_transLayer_distance)<mindis_total:
                                    self.inLayer_distance[0][layer_id] = maxdis_in
                                    self.transLayer_distance[0][layer_id] = tmp_transLayer_distance
                                    self.aggregate_arg[layer_id] = src_pos[A]
                                    mindis_total = maxdis_in+tmp_transLayer_distance
                else:
                    if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                        src_pos = np.argwhere(self.mapping_result == layer_id)
                        
                        if len(src_pos) == 1:
                            self.inLayer_distance[0][layer_id] = 0
                            self.aggregate_arg[layer_id] = src_pos[0]
                            maxdis = 0
                            for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                                for i in range(len(dst_pos)):
                                    dis = abs(src_pos[0][0] - dst_pos[i][0]) + abs(src_pos[0][1] - dst_pos[i][1])
                                    if dis > maxdis:
                                        maxdis = dis
                            self.transLayer_distance[0][layer_id] = maxdis
                        else:
                            mindis_total = 1000
                            for A in range(len(src_pos)):
                                maxdis_in = 0
                                maxdis_out = 0
                                for i in range(len(src_pos)):
                                    if i != A:
                                        dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                        if dis_in > maxdis_in:
                                            maxdis_in = dis_in
                                for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                    dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                                    for j in range(len(dst_pos)):
                                        dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                                        if dis_out > maxdis_out:
                                            maxdis_out = dis_out
                                tempdis = maxdis_in + maxdis_out
                                if tempdis < mindis_total:
                                    self.inLayer_distance[0][layer_id] = maxdis_in
                                    self.transLayer_distance[0][layer_id] = maxdis_out
                                    self.aggregate_arg[layer_id] = src_pos[A]
                                    mindis_total = tempdis
                    elif self.layer_tileinfo[layer_id]['type'] == 'element_sum' or self.layer_tileinfo[layer_id]['type'] == 'element_multiply':
                        maxdis_out = 0
                        for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                            dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                            for j in range(len(dst_pos)):
                                dis_out = abs(dst_pos[0][0]-1/2*self.tile_num[0]) + dst_pos[0][1]
                                if dis_out > maxdis_out:
                                    maxdis_out = dis_out
                        self.inLayer_distance[0][layer_id] = 0
                        self.transLayer_distance[0][layer_id] = maxdis_out
            final_pos = np.argwhere(self.mapping_result == self.layer_num - 1)
            # Determine the aggregate node for layer N (output layer)
            mindis = 1000
            for i in range(len(final_pos)):
                maxdis = 0
                for j in range(len(final_pos)):
                    if j != i:
                        dis = abs(final_pos[i][0] - final_pos[j][0]) + abs(final_pos[i][1] - final_pos[j][1])
                        if dis > maxdis:
                            maxdis = dis
                if maxdis < mindis:
                    mindis = maxdis
                    self.inLayer_distance[0][self.layer_num - 1] = mindis
                    self.aggregate_arg[self.layer_num - 1] = final_pos[i]
                    self.transLayer_distance[0][self.layer_num - 1] = 0
        # self.total_distance = sum(sum(self.trans_time * (self.inLayer_distance + self.transLayer_distance)))
        #linqiushi modified
        #add for the rewrite
        if self.rewrite_mode==1:
            for k in range(self.rewrite_time):
                if self.start_layer[k]!=self.final_layer[k]:
                    for layer_id in range(self.start_layer[k],self.final_layer[k]):
                        # Determine the aggregate node for layer 0~N-1
                        if self.layer_tileinfo[layer_id]['is_branchout'] == 1:
                            # for the layer which is a output layer of one branch and the next layer is element_sum
                            if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==layer_id:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][layer_id] = 0
                                    self.aggregate_arg[layer_id] = src_pos[0]
                                    self.transLayer_distance[0][layer_id] = abs(src_pos[0][0]-1/2*self.tile_num[0]) + src_pos[0][1]
                                else:
                                    mindis_total = 1000
                                    for A in range(len(src_pos)):
                                        tmp_transLayer_distance = abs(src_pos[A][0]-1/2*self.tile_num[0]) + src_pos[A][1]
                                        maxdis_in = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        if (maxdis_in+tmp_transLayer_distance)<mindis_total:
                                            self.inLayer_distance[0][layer_id] = maxdis_in
                                            self.transLayer_distance[0][layer_id] = tmp_transLayer_distance
                                            self.aggregate_arg[layer_id] = src_pos[A]
                                            mindis_total = maxdis_in+tmp_transLayer_distance
                        else:
                            if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==layer_id:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][layer_id] = 0
                                    self.aggregate_arg[layer_id] = src_pos[0]
                                    maxdis = 0
                                    for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                        
                                        dst_pos=[]
                                        for i in range(len(self.mapping_result_rewrite)):
                                            for j in range(len(self.mapping_result_rewrite[i])):
                                                if len(self.mapping_result_rewrite[i][j])>k:
                                                    if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & (layer_id+idx)<=self.final_layer[k]:
                                                        dst_pos.append([i,j])
                                        assert len(dst_pos)>0 , f'next layer not on the tiles'
                                        for i in range(len(dst_pos)):
                                            dis = abs(src_pos[0][0] - dst_pos[i][0]) + abs(src_pos[0][1] - dst_pos[i][1])
                                            if dis > maxdis:
                                                maxdis = dis
                                    self.transLayer_distance[0][layer_id] = maxdis
                                else:
                                    mindis_total = 1000
                                    for A in range(len(src_pos)):
                                        maxdis_in = 0
                                        maxdis_out = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                            dst_pos=[]
                                            for i in range(len(self.mapping_result_rewrite)):
                                                for j in range(len(self.mapping_result_rewrite[i])):
                                                    if len(self.mapping_result_rewrite[i][j])>k:
                                                        if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & (layer_id+idx)<=self.final_layer[k]:
                                                            dst_pos.append([i,j])
                                            assert len(dst_pos)>0 , f'next layer not on the tiles'
                                            for j in range(len(dst_pos)):
                                                dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                                                if dis_out > maxdis_out:
                                                    maxdis_out = dis_out
                                        tempdis = maxdis_in + maxdis_out
                                        if tempdis < mindis_total:
                                            self.inLayer_distance[0][layer_id] = maxdis_in
                                            self.transLayer_distance[0][layer_id] = maxdis_out
                                            self.aggregate_arg[layer_id] = src_pos[A]
                                            mindis_total = tempdis
                            elif self.layer_tileinfo[layer_id]['type'] == 'element_sum' or self.layer_tileinfo[layer_id]['type'] == 'element_multiply':
                                maxdis_out = 0
                                for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                    dst_pos=[]
                                    for i in range(len(self.mapping_result_rewrite)):
                                        for j in range(len(self.mapping_result_rewrite[i])):
                                            if len(self.mapping_result_rewrite[i][j])>k:
                                                if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & (layer_id+idx)<=self.final_layer[k]:
                                                    dst_pos.append([i,j])
                                    assert len(dst_pos)>0 , f'next layer not on the tiles'
                                    for j in range(len(dst_pos)):
                                        dis_out = abs(dst_pos[0][0]-1/2*self.tile_num[0]) + dst_pos[0][1]
                                        if dis_out > maxdis_out:
                                            maxdis_out = dis_out
                                self.inLayer_distance[0][layer_id] = 0
                                self.transLayer_distance[0][layer_id] = maxdis_out
                    #special case for  the final_layer[k]
                    if k==self.rewrite_time-1:
                        final_pos=[]
                        #init the src_pos:
                        for i in range(len(self.mapping_result_rewrite)):
                            for j in range(len(self.mapping_result_rewrite[i])):
                                if len(self.mapping_result_rewrite[i][j])>k:
                                    if self.mapping_result_rewrite[i][j][k]==self.final_layer[k]:
                                        final_pos.append([i,j])
                        
                        mindis = 1000
                        for i in range(len(final_pos)):
                            maxdis = 0
                            for j in range(len(final_pos)):
                                if j != i:
                                    dis = abs(final_pos[i][0] - final_pos[j][0]) + abs(final_pos[i][1] - final_pos[j][1])
                                    if dis > maxdis:
                                        maxdis = dis
                            if maxdis < mindis:
                                mindis = maxdis
                                self.inLayer_distance[0][self.final_layer[k]] = mindis
                                self.aggregate_arg[self.final_layer[k]] = final_pos[i]
                                self.transLayer_distance[0][self.final_layer[k]] = 0
                    else:
                        if self.layer_tileinfo[self.final_layer[k]]['is_branchout'] == 1:
                            # for the layer which is a output layer of one branch and the next layer is element_sum
                            if self.layer_tileinfo[self.final_layer[k]]['type'] in ['conv', 'pooling', 'fc']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==self.final_layer[k]:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][self.final_layer[k]] = 0
                                    self.aggregate_arg[self.final_layer[k]] = src_pos[0]
                                    self.transLayer_distance[0][self.final_layer[k]] = abs(src_pos[0][0]-1/2*self.tile_num[0]) + src_pos[0][1]
                                else:
                                    mindis_total = 1000
                                    for A in range(len(src_pos)):
                                        tmp_transLayer_distance = src_pos[A][0] + src_pos[A][1]
                                        maxdis_in = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        if (maxdis_in+tmp_transLayer_distance)<mindis_total:
                                            self.inLayer_distance[0][self.final_layer[k]] = maxdis_in
                                            self.transLayer_distance[0][self.final_layer[k]] = tmp_transLayer_distance
                                            self.aggregate_arg[self.final_layer[k]] = src_pos[A]
                                            mindis_total = maxdis_in+tmp_transLayer_distance
                        else:
                            if self.layer_tileinfo[self.final_layer[k]]['type'] in ['conv', 'pooling', 'fc']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==self.final_layer[k]:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][self.final_layer[k]] = 0
                                    self.aggregate_arg[self.final_layer[k]] = src_pos[0]
                                    maxdis = 0
                                    for idx in self.layer_tileinfo[self.final_layer[k]]['Outputindex']:
                                        
                                        dst_pos=[]
                                        for i in range(len(self.mapping_result_rewrite)):
                                            for j in range(len(self.mapping_result_rewrite[i])):
                                                if len(self.mapping_result_rewrite[i][j])>k+1:
                                                    if (self.mapping_result_rewrite[i][j][k+1]==self.final_layer[k]+idx) :
                                                        dst_pos.append([i,j])
                                        
                                        assert len(dst_pos)>0 , f'next layer not on the tiles'
                                        for i in range(len(dst_pos)):
                                            dis = abs(src_pos[0][0] - dst_pos[i][0]) + abs(src_pos[0][1] - dst_pos[i][1])
                                            if dis > maxdis:
                                                maxdis = dis
                                    self.transLayer_distance[0][self.final_layer[k]] = maxdis
                                else:
                                    mindis_total = 1000
                                    for A in range(len(src_pos)):
                                        maxdis_in = 0
                                        maxdis_out = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        for idx in self.layer_tileinfo[self.final_layer[k]]['Outputindex']:
                                            dst_pos=[]
                                            for i in range(len(self.mapping_result_rewrite)):
                                                for j in range(len(self.mapping_result_rewrite[i])):
                                                    if len(self.mapping_result_rewrite[i][j])>k+1:
                                                        if (self.mapping_result_rewrite[i][j][k+1]==self.final_layer[k]+idx):
                                                            dst_pos.append([i,j])
                                            if len(dst_pos)<=0:
                                                print("end",self.mapping_result_rewrite)
                                                print("没有在tile上",k,self.final_layer[k],self.final_layer[k]+idx,self.layer_tileinfo[self.final_layer[k]]['Outputindex'],self.layer_tileinfo[self.final_layer[k]]['type'])
                                                assert 0
                                            assert len(dst_pos)>0 , f'next layer not on the tiles'
                                            for j in range(len(dst_pos)):
                                                dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                                                if dis_out > maxdis_out:
                                                    maxdis_out = dis_out
                                        tempdis = maxdis_in + maxdis_out
                                        if tempdis < mindis_total:
                                            self.inLayer_distance[0][self.final_layer[k]] = maxdis_in
                                            self.transLayer_distance[0][self.final_layer[k]] = maxdis_out
                                            self.aggregate_arg[self.final_layer[k]] = src_pos[A]
                                            mindis_total = tempdis
                            elif self.layer_tileinfo[self.final_layer[k]]['type'] == 'element_sum' or self.layer_tileinfo[self.final_layer[k]]['type'] == 'element_multiply':
                                maxdis_out = 0
                                for idx in self.layer_tileinfo[self.final_layer[k]]['Outputindex']:
                                    dst_pos=[]
                                    for i in range(len(self.mapping_result_rewrite)):
                                        for j in range(len(self.mapping_result_rewrite[i])):
                                            if len(self.mapping_result_rewrite[i][j])>k+1:
                                                if (self.mapping_result_rewrite[i][j][k+1]==self.final_layer[k]+idx) :
                                                    dst_pos.append([i,j])
                                    assert len(dst_pos)>0 , f'next layer not on the tiles'
                                    for j in range(len(dst_pos)):
                                        dis_out = abs(dst_pos[0][0]-1/2*self.tile_num[0]) + dst_pos[0][1]
                                        if dis_out > maxdis_out:
                                            maxdis_out = dis_out
                                self.inLayer_distance[0][self.final_layer[k]] = 0
                                self.transLayer_distance[0][self.final_layer[k]] = maxdis_out
            # self.total_distance = sum(sum(self.trans_time * (self.inLayer_distance + self.transLayer_distance)))
        elif self.rewrite_mode==2:
            self.inLayer_distance = [[[] for _ in range(self.layer_num)] for _ in range(1)]
            self.transLayer_distance = [[[] for _ in range(self.layer_num)] for _ in range(1)]
            self.aggregate_arg = [[[] for _ in range(1)] for _ in range(self.layer_num)]

            for k in range(self.rewrite_time):
                if len(self.rewrite_layer_list[k])!=1:
                    for layer_id in self.rewrite_layer_list[k]:
                        # Determine the aggregate node for layer 0~N-1
                        if self.layer_tileinfo[layer_id]['is_branchout'] == 1:
                            # for the layer which is a output layer of one branch and the next layer is element_sum
                            if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc','MM']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==layer_id:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][layer_id].append(0)
                                    self.aggregate_arg[layer_id].append(src_pos[0])
                                    self.transLayer_distance[0][layer_id].append(abs(src_pos[0][0]-1/2*self.tile_num[0]) + src_pos[0][1])
                                else:
                                    mindis_total = 100000
                                    for A in range(len(src_pos)):
                                        tmp_transLayer_distance = abs(src_pos[A][0]-1/2*self.tile_num[0]) + src_pos[A][1]
                                        maxdis_in = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        if (maxdis_in+tmp_transLayer_distance)<mindis_total:
                                            final_in=maxdis_in
                                            final_out=tmp_transLayer_distance
                                            final_src=A
                                            mindis_total = maxdis_in+tmp_transLayer_distance
                                    self.inLayer_distance[0][layer_id].append(final_in)
                                    self.transLayer_distance[0][layer_id].append(final_out)
                                    self.aggregate_arg[layer_id].append(src_pos[final_src])
                        else:
                            if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc','MM']:
                                src_pos=[]
                                #init the src_pos:
                                for i in range(len(self.mapping_result_rewrite)):
                                    for j in range(len(self.mapping_result_rewrite[i])):
                                        if len(self.mapping_result_rewrite[i][j])>k:
                                            if self.mapping_result_rewrite[i][j][k]==layer_id:
                                                src_pos.append([i,j])
                                
                                if len(src_pos) == 1:
                                    self.inLayer_distance[0][layer_id].append(0)
                                    self.aggregate_arg[layer_id] = src_pos[0]
                                    maxdis = 0
                                    for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                        
                                        dst_pos=[]
                                        for i in range(len(self.mapping_result_rewrite)):
                                            for j in range(len(self.mapping_result_rewrite[i])):
                                                if len(self.mapping_result_rewrite[i][j])>k:
                                                    if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & ((layer_id+idx) in self.rewrite_layer_list[k]):
                                                        dst_pos.append([i,j])
                                        #assert len(dst_pos)>0 , f'next layer not on the tiles'
                                        if len(dst_pos)>0:
                                            for i in range(len(dst_pos)):
                                                dis = abs(src_pos[0][0] - dst_pos[i][0]) + abs(src_pos[0][1] - dst_pos[i][1])
                                                if dis > maxdis:
                                                    maxdis = dis
                                        elif len(dst_pos)==0:
                                            maxdis=0
                                    self.transLayer_distance[0][layer_id] = maxdis
                                else:
                                    mindis_total = 100000
                                    for A in range(len(src_pos)):
                                        maxdis_in = 0
                                        maxdis_out = 0
                                        for i in range(len(src_pos)):
                                            if i != A:
                                                dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                                if dis_in > maxdis_in:
                                                    maxdis_in = dis_in
                                        for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                            dst_pos=[]
                                            for i in range(len(self.mapping_result_rewrite)):
                                                for j in range(len(self.mapping_result_rewrite[i])):
                                                    if len(self.mapping_result_rewrite[i][j])>k:
                                                        if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & (layer_id+idx) in self.rewrite_layer_list[k]:
                                                            dst_pos.append([i,j])
                                            if len(dst_pos)>0:
                                                for j in range(len(dst_pos)):
                                                    dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                                                    if dis_out > maxdis_out:
                                                        maxdis_out = dis_out
                                            elif len(dst_pos)==0:
                                                maxdis_out=0
                                                #TODO: maybe change the maxdis_out
                                        tempdis = maxdis_in + maxdis_out
                                        if tempdis < mindis_total:
                                            final_src=A
                                            final_in=maxdis_in
                                            final_out=maxdis_out
                                            mindis_total = tempdis
                                    self.inLayer_distance[0][layer_id].append(final_in)
                                    self.transLayer_distance[0][layer_id].append(final_out)
                                    self.aggregate_arg[layer_id].append(src_pos[final_src])
                            elif self.layer_tileinfo[layer_id]['type'] == 'element_sum' or self.layer_tileinfo[layer_id]['type'] == 'element_multiply':
                                maxdis_out = 0
                                for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                    dst_pos=[]
                                    for i in range(len(self.mapping_result_rewrite)):
                                        for j in range(len(self.mapping_result_rewrite[i])):
                                            if len(self.mapping_result_rewrite[i][j])>k:
                                                if (self.mapping_result_rewrite[i][j][k]==layer_id+idx) & (layer_id+idx) in self.rewrite_layer_list[k]:
                                                    dst_pos.append([i,j])
                                    #assert len(dst_pos)>0 , f'next layer not on the tiles'
                                    if len(dst_pos)>0:
                                        for j in range(len(dst_pos)):
                                            dis_out = abs(dst_pos[0][0]-1/2*self.tile_num[0]) + dst_pos[0][1]
                                            if dis_out > maxdis_out:
                                                maxdis_out = dis_out
                                    elif len(dst_pos)==0:
                                        maxdis_out=0
                                self.inLayer_distance[0][layer_id] = 0
                                self.transLayer_distance[0][layer_id] = maxdis_out
                    for i in range(self.layer_num):
                        if i not in self.rewrite_layer_list[k]:
                            self.inLayer_distance[0][i].append(0)
                            self.transLayer_distance[0][i].append(0)
                            self.aggregate_arg[layer_id].append([0,0])
                elif len(self.rewrite_layer_list[k])==1:
                    #only one layer on the tile
                    layer_id =self.rewrite_layer_list[k][0]
                    assert self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc','MM']
                    # for i in range(len(self.mapping_result_rewrite)):
                    #     for j in range(len(self.mapping_result_rewrite[i])):
                    #         if len(self.mapping_result_rewrite[i][j])>k:
                    #             if self.mapping_result_rewrite[i][j][k]==layer_id:
                    #                 src_pos.append([i,j])
                    # mindis_total=100000
                    # for A in range(len(src_pos)):
                    #     maxdis_in = 0
                    #     for i in range(len(src_pos)):
                    #         if i != A:
                    #             dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                    #             if dis_in > maxdis_in:
                    #                 maxdis_in = dis_in       
                    #     if (maxdis_in)<mindis_total:
                    #         final_src=A
                    #         final_dis=maxdis_in
                    #         mindis_total=maxdis_in
                    final_dis=math.floor(len(self.mapping_result_rewrite)/2)+math.floor(len(self.mapping_result_rewrite[0])/2)
                    final_src=[math.floor(len(self.mapping_result_rewrite)/2),math.floor(len(self.mapping_result_rewrite[0])/2)]
                    self.inLayer_distance[0][layer_id].append(final_dis)
                    self.transLayer_distance[0][layer_id].append(0)
                    self.aggregate_arg[layer_id] = final_src
                    for i in range(self.layer_num):
                        if i not in self.rewrite_layer_list[k]:
                            self.inLayer_distance[0][i].append(0)
                            self.transLayer_distance[0][i].append(0)
                            self.aggregate_arg[layer_id].append([0,0])
            # self.total_distance = sum(sum(self.trans_time * (self.inLayer_distance + self.transLayer_distance)))
        #for i in range(self.layer_num):
            #print(self.transLayer_distance[0][i])
        #assert 0        
    def mapping_output(self,mix_tile=None):
        intra_tile_bandwidth = 1024*(10**9)
        inter_tile_bandwidth = 20*(10**9)
        with open('layer_table.txt', 'w', encoding='utf-8') as file:
            file.write(str(self.layer_num)+'\n')
            for layer_id in range(self.layer_num):
                #file.write(str(layer_id)+' ')
                src_pos = np.argwhere(self.mapping_result == layer_id)
                file.write(str(len(src_pos))+' ')
                #print(src_pos)
                for i in range(len(src_pos)):
                    file.write(str(int(src_pos[i][0]*self.tile_num[0]+src_pos[i][1]))+' ')
                #print(self.aggregate_arg[layer_id])
                file.write(str(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))+'\n')

        with open('injection.txt','w',encoding='utf-8') as file:
            if self.mix_mode==4:
                for layer_id in range(self.layer_num-1):
                    print(layer_id)
                    src_pos = np.argwhere(self.mapping_result == layer_id)
                    if (len(src_pos)==0):
                        continue
                    if len(src_pos)>1:
                        for i in range(len(src_pos)):
                            self.mode4posi.append(int(src_pos[i][0]*self.tile_num[0]+src_pos[i][1]))
                            self.mode4posj.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode4injection.append(self.layer_tileinfo[layer_id]['inLayer_data']*self.FPS/intra_tile_bandwidth)
                    dst_pos=np.argwhere(self.mapping_result == layer_id+1)
                    if len(dst_pos)==0:
                        dst_pos=np.argwhere(self.mapping_result == layer_id+2)
                        for i in range(len(dst_pos)):
                            self.mode4posj.append(int(dst_pos[i][0]*self.tile_num[0]+dst_pos[i][1]))
                            self.mode4posi.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode4injection.append(self.layer_tileinfo[layer_id+2]['transLayer_data_before']*self.FPS/inter_tile_bandwidth)
                    else:
                        for i in range(len(dst_pos)):
                            self.mode4posj.append(int(dst_pos[i][0]*self.tile_num[0]+dst_pos[i][1]))
                            self.mode4posi.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode4injection.append(self.layer_tileinfo[layer_id+1]['transLayer_data_before']*self.FPS/inter_tile_bandwidth)
                src_pos = np.argwhere(self.mapping_result == self.layer_num-1)
                if len(src_pos)>1:
                    for i in range(len(src_pos)):
                        self.mode4posi.append(int(src_pos[i][0]*self.tile_num[0]+src_pos[i][1]))
                        self.mode4posj.append(int(self.aggregate_arg[self.layer_num-1][0]*self.tile_num[0]+self.aggregate_arg[self.layer_num-1][1]))
                        self.mode4injection.append(self.layer_tileinfo[self.layer_num-1]['inLayer_data']*self.FPS/intra_tile_bandwidth)
                for i in range(len(self.mode4injection)):
                    if (self.mode4posi[i] != self.mode4posj[i]):
                        file.write(str(self.mode4posi[i])+' '+str(self.mode4posj[i])+' '+str(self.mode4injection[i])+'\n')
            elif self.mix_mode==2:
                for layer_id in range(self.layer_num-1):
                    print(layer_id)
                    src_pos = np.argwhere(self.mapping_result == layer_id)
                    if (len(src_pos)==0):
                        continue
                    if len(src_pos)>1:
                        for i in range(len(src_pos)):
                            a=int(src_pos[i][0])
                            b=int(src_pos[i][1])
                            self.mode2posi.append(int(src_pos[i][0]*self.tile_num[0]+src_pos[i][1]))
                            self.mode2posj.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode2injection.append(self.layer_tileinfo[layer_id]['inLayer_data']*(mix_tile.PE_num[a][b]**2)*(mix_tile.xbar_size[a][b]**2)*self.FPS/intra_tile_bandwidth)
                    dst_pos=np.argwhere(self.mapping_result == layer_id+1)
                    if len(dst_pos)==0:
                        dst_pos=np.argwhere(self.mapping_result == layer_id+2)
                        for i in range(len(dst_pos)):
                            a=int(dst_pos[i][0])
                            b=int(dst_pos[i][1])
                            self.mode2posj.append(int(dst_pos[i][0]*self.tile_num[0]+dst_pos[i][1]))
                            self.mode2posi.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode2injection.append(self.layer_tileinfo[layer_id+2]['transLayer_data_before']*(mix_tile.PE_num[a][b]**2)*(mix_tile.xbar_size[a][b]**2)*self.FPS/inter_tile_bandwidth)
                    else:
                        for i in range(len(dst_pos)):
                            a=int(dst_pos[i][0])
                            b=int(dst_pos[i][1])
                            self.mode2posj.append(int(dst_pos[i][0]*self.tile_num[0]+dst_pos[i][1]))
                            self.mode2posi.append(int(self.aggregate_arg[layer_id][0]*self.tile_num[0]+self.aggregate_arg[layer_id][1]))
                            self.mode2injection.append(self.layer_tileinfo[layer_id+1]['transLayer_data_before']*(mix_tile.PE_num[a][b]**2)*(mix_tile.xbar_size[a][b]**2)*self.FPS/inter_tile_bandwidth)
                src_pos = np.argwhere(self.mapping_result == self.layer_num-1)
                if len(src_pos)>1:
                    for i in range(len(src_pos)):
                        a=int(src_pos[i][0])
                        b=int(src_pos[i][1])
                        self.mode2posi.append(int(src_pos[i][0]*self.tile_num[0]+src_pos[i][1]))
                        self.mode2posj.append(int(self.aggregate_arg[self.layer_num-1][0]*self.tile_num[0]+self.aggregate_arg[self.layer_num-1][1]))
                        self.mode2injection.append(self.layer_tileinfo[self.layer_num-1]['inLayer_data']*(mix_tile.PE_num[a][b]**2)*(mix_tile.xbar_size[a][b]**2)*self.FPS/intra_tile_bandwidth)
                for i in range(len(self.mode2injection)):
                    if (self.mode2posi[i] != self.mode2posj[i]):
                        file.write(str(self.mode2posi[i])+' '+str(self.mode2posj[i])+' '+str(self.mode2injection[i])+'\n')                              
                
               
        

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()

    test = TCG(structure_file, test_SimConfig_path)
    test.mapping_net()
    test.calculate_transfer_distance()
    # print(test.total_distance)
