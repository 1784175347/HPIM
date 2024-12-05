import math
from random import random
import matplotlib.pyplot as plt
import subprocess
from IPython import embed
import random
#random.seed(2)
random.seed(5)
import torch
import os
from MNSIM.Interface.interface import *
import configparser
import time
import copy

class SA:
    def __init__(self, T0=100, Tf=10, alpha=0.99, k=1, a=1, b=1, d=1, e=1, area_des=1000000000, power_des=1000, latency_des=1000000000, energy_des=10000000000, mix_mode='2'):
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.k = k
        self.x = random.random() * 11 - 5  # 随机生成一个x的值
        self.y = random.random() * 11 - 5  # 随机生成一个y的值
        self.tilenum = 64
        self.tile_type = [['NVM' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.PE_num = [[2 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.xbar_size = [[256 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.mapping = [['no' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.auto_layer_mapping = 0
        self.area = 0
        self.power = 0
        self.latency = 0
        self.energy = 0
        self.most_best = []
        self.history = {'f': [], 'T': [], 'area': [], 'power': [], 'latency': [], 'tilenum': [], 'tile_type': [], 'PE_num': [], 'xbar_size': [], 'mapping': []}
        self.layernum = 12
        self.tile_type_layer = ['NVM' for _ in range(self.layernum)]
        self.PE_num_layer = [2 for _ in range(self.layernum)]
        self.xbar_size_layer = [256 for _ in range(self.layernum)]
        self.layertilenum = [1 for _ in range(self.layernum)]
        self.tile_type_layer_tile = [[self.tile_type_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.PE_num_layer_tile = [[self.PE_num_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.xbar_size_layer_tile = [[self.xbar_size_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.area_des = area_des				
        self.power_des = power_des
        self.latency_des = latency_des
        self.energy_des = energy_des
        self.a = a
        self.b = b
        self.d = d
        self.e = e
        self.tile_connection = 3
        self.topology = 0
        self.c = 2
        self.hetro = 1
        self.step = 0
        self.step_step = 100
        self.net = 'vgg8'
        self.dataset = 'cifar10'
        self.weightpath = "cifar10_vgg8_params.pth"
        self.mix_mode = mix_mode

    def update_ini_file(self, file_path, tile_connection):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if ((self.step%self.step_step)==0 or self.T < self.Tf):
            booksim_en = 1
            floorplan_en = 0
        else:
            booksim_en = 0
            floorplan_en = 0
        updated_lines = []
        for line in lines:
            if line.startswith('Tile_Connection'):
                line = f'Tile_Connection = {tile_connection}\n'
            if line.startswith('Booksim_en'):
                line = f'Booksim_en={booksim_en}\n'
            if line.startswith('Floorplan_en'):
                line = f'Floorplan_en={floorplan_en}\n'
            updated_lines.append(line)
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)
    
    def func1(self, area, power, latency):                 
        a = 0.9625857825131118
        b = 0.3879937140120758
        c = 1.1602570877109697
        res = ((area)**a) * ((power)**b) * ((latency)**c)/(10**14)
        if res == 0:
            return -1
        return res
    
    
    def func(self, area, power, latency, energy): 
        if area == 0 or power == 0 or latency == 0:
            return -1

        if (self.a == 1):
            area_data = area/self.area_des
        else:
            area_data = 1
            if (area > self.area_des):
                return -1
        
        if (self.b == 1):
            power_data = power/self.power_des
        else:
            power_data = 1
            if (power > self.power_des):
                return -1
            
        if (self.d == 1):
            latency_data = latency/self.latency_des
        else:
            latency_data = 1
            if (latency > self.latency_des):
                return -1
            
        if (self.e == 1):
            energy_data = energy/self.energy_des
        else:
            energy_data = 1
            if (energy > self.energy_des):
                return -1
            
        return area_data*latency_data*power_data*energy_data*10000       
    
    def generate_new_layer(self, layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection, topology, c):
        seach = random.random()
        if seach <= 0.1*self.hetro:#随机选择一个层，改变单元类型
            layer = int(random.random() * layernum)  
            if tile_type_layer[layer] == 'SRAM' :
                tile_type_layer[layer] = 'NVM'
            else :
                tile_type_layer[layer] = 'SRAM'
        elif seach <= 0.45+0.05*self.hetro:#随机选择一个层，改变PE数量
            layer = int(random.random() * layernum)
            change = int(random.random() * 3)
            PE_num_layer[layer] = 2**(change)
        elif seach <= 0.9:#随机选择一个层，改变Xbar尺寸
            layer = int(random.random() * layernum)
            change = int(random.random() * 4)
            xbar_size_layer[layer] = 2**(change+7)
            '''if xbar_size_layer[layer] == 1024 :
                xbar_size_layer[layer] = 512
            else :
                xbar_size_layer[layer] = 1024''' 
        elif seach <= 1.1:
            tile_connection = int(random.random() * 3)
            if tile_connection == 2:
                tile_connection = 3
        elif seach <= 1.1:
            topology = int(random.random() * 2)
        elif seach <= 1:
            c = int(random.random() * 2)+2
            c = 2**c
        return tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection, topology, c

    def generate_new_tile(self, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer, topology, c):
        search = layer
        layer_dict = structure_file[search][0][0]
        layer_type = layer_dict['type']
        weight_precision = int(layer_dict['Weightbit']) - 1
        if layer_type == 'conv':
            mix_chage = Mix_Tile(layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search], tile_type_layer_tile[search], int(layer_dict['Kernelsize']), int(layer_dict['Outputchannel']), int(layer_dict['Inputchannel']), weight_precision)
        elif layer_type == 'fc':
            mix_chage = Mix_Tile(layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search], tile_type_layer_tile[search], 1, int(layer_dict['Outfeature']), int(layer_dict['Infeature']), weight_precision)
        elif layer_type == 'pooling':
            for i in range(layertilenum[search]):
                PE_num_layer_tile[search][i] = 1
                xbar_size_layer_tile[search][i] = 32
            return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
        else:
            return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
        
        search_tile = int(random.random() * layertilenum[search])
        search_tile_0 = int(random.random() * layertilenum[search])
        while(search_tile_0==search_tile and layertilenum[search] != 1):
            search_tile_0 = int(random.random() * layertilenum[search])
        search_choice = random.random()
        if search_choice < 0.1*self.hetro:#随机选择一个层，改变单元类型
            if tile_type_layer_tile[search][search_tile] == 'SRAM' :
                tile_type_layer_tile[search][search_tile] = 'NVM'
            else :
                tile_type_layer_tile[search][search_tile]= 'SRAM'
        elif search_choice <= 0.45+0.05*self.hetro:#随机选择一个层，改变PE数量
            change = int(random.random() * 3)
            des = 2**(change)
            if des != PE_num_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_PE(search_tile,PE_num_layer_tile[search][search_tile],des)
        elif search_choice <= 0.9:#随机选择一个层，改变Xbar尺寸
            change = int(random.random() * 4)
            des = 2**(change+7)
            if des != xbar_size_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_Xbar(search_tile,xbar_size_layer_tile[search][search_tile],des)
        elif search_choice <= 1.1:
            tile_type_layer_tile[search][search_tile],tile_type_layer_tile[search][search_tile_0] = tile_type_layer_tile[search][search_tile_0],tile_type_layer_tile[search][search_tile]
            PE_num_layer_tile[search][search_tile],PE_num_layer_tile[search][search_tile_0] = PE_num_layer_tile[search][search_tile_0],PE_num_layer_tile[search][search_tile]
            xbar_size_layer_tile[search][search_tile],xbar_size_layer_tile[search][search_tile_0] = xbar_size_layer_tile[search][search_tile_0],xbar_size_layer_tile[search][search_tile]
        elif search_choice <= 1.1:
            topology = int(random.random() * 2)
        elif search_choice <= 1:
            c = int(random.random() * 2)+2
        return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c

    def generate_new_tile_all_space(self, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer, topology, c):
        search = layer
        layer_dict = structure_file[search][0][0]
        layer_type = layer_dict['type']
        weight_precision = int(layer_dict['Weightbit']) - 1
        if layer_type == 'conv':
            mix_chage = Mix_Tile(layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search], tile_type_layer_tile[search], int(layer_dict['Kernelsize']), int(layer_dict['Outputchannel']), int(layer_dict['Inputchannel']), weight_precision)
        elif layer_type == 'fc':
            mix_chage = Mix_Tile(layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search], tile_type_layer_tile[search], 1, int(layer_dict['Outfeature']), int(layer_dict['Infeature']), weight_precision)
        elif layer_type == 'pooling':
            for i in range(layertilenum[search]):
                PE_num_layer_tile[search][i] = 1
                xbar_size_layer_tile[search][i] = 32
            return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
        else:
            return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
        
        search_tile = int(random.random() * layertilenum[search])
        search_tile_0 = int(random.random() * layertilenum[search])
        while(search_tile_0==search_tile and layertilenum[search] != 1):
            search_tile_0 = int(random.random() * layertilenum[search])
        search_choice = random.random()
        if search_choice < 0.1*self.hetro:#随机选择一个层，改变单元类型
            if tile_type_layer_tile[search][search_tile] == 'SRAM' :
                tile_type_layer_tile[search][search_tile] = 'NVM'
            else :
                tile_type_layer_tile[search][search_tile]= 'SRAM'
        elif search_choice <= 0.45+0.05*self.hetro:#随机选择一个层，改变PE数量
            change = int(random.random() * 3)
            des = 2**(change)
            if des != PE_num_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_PE(search_tile,PE_num_layer_tile[search][search_tile],des)
        elif search_choice <= 0.9:#随机选择一个层，改变Xbar尺寸
            change = int(random.random() * 4)
            des = 2**(change+7)
            if des != xbar_size_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_Xbar(search_tile,xbar_size_layer_tile[search][search_tile],des)
        elif search_choice <= 1.1:
            tile_type_layer_tile[search][search_tile],tile_type_layer_tile[search][search_tile_0] = tile_type_layer_tile[search][search_tile_0],tile_type_layer_tile[search][search_tile]
            PE_num_layer_tile[search][search_tile],PE_num_layer_tile[search][search_tile_0] = PE_num_layer_tile[search][search_tile_0],PE_num_layer_tile[search][search_tile]
            xbar_size_layer_tile[search][search_tile],xbar_size_layer_tile[search][search_tile_0] = xbar_size_layer_tile[search][search_tile_0],xbar_size_layer_tile[search][search_tile]
        elif search_choice <= 1.1:
            topology = int(random.random() * 2)
        elif search_choice <= 1:
            c = int(random.random() * 2)+2
        return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c


    def generate_normal_matrix(self, row, column):
        matrix = np.zeros([row, column])
        start = 0
        for i in range(row):
            for j in range(column):
                matrix[i][j] = start
                start += 1
        return matrix


    def generate_snake_matrix(self, row, column):
        matrix = np.zeros([row, column])
        start = 0
        for i in range(row):
            for j in range(column):
                if i % 2:
                    matrix[i][column - j - 1] = start
                else:
                    matrix[i][j] = start
                start += 1
        return matrix

    def generate_hui_matrix(self, row, column):
        matrix = np.zeros([row, column])
        state = 0
        stride = 1
        step = 0
        start = 0
        dl = 0
        ru = 0
        i = 0
        j = 0
        for x in range(row * column):
            if x == 0:
                matrix[i][j] = start
            else:
                if state == 0:
                    j += 1
                    matrix[i][j] = start
                    state = 1
                elif state == 1:
                    if dl == 0:
                        i += 1
                        matrix[i][j] = start
                        step += 1
                        if step == stride:
                            dl = 1
                            step = 0
                    elif dl == 1:
                        j -= 1
                        matrix[i][j] = start
                        step += 1
                        if step == stride:
                            dl = 0
                            step = 0
                            stride += 1
                            state = 2
                elif state == 2:
                    i += 1
                    matrix[i][j] = start
                    state = 3
                elif state == 3:
                    if ru == 0:
                        j += 1
                        matrix[i][j] = start
                        step += 1
                        if step == stride:
                            ru = 1
                            step = 0
                    elif ru == 1:
                        i -= 1
                        matrix[i][j] = start
                        step += 1
                        if step == stride:
                            ru = 0
                            step = 0
                            stride += 1
                            state = 0
            start += 1
        return matrix

    def generate_zigzag_matrix(self, row, column):
        matrix = np.zeros([row, column])
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
            start += 1
        return matrix

    def generate_normal_matrix_cmesh(self, row, column, c=2):
        matrix_min=self.generate_normal_matrix(c,c)
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
        return matrix

    def generate_snake_matrix_cmesh(self, row, column, c=2):
        matrix_min=self.generate_snake_matrix(c,c)
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
        return matrix   

    def generate_zigzag_matrix_cmesh(self, row, column, c=2):
        matrix_min=self.generate_zigzag_matrix(c,c)
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
        return matrix

    def generate_new_layer_config(self, layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, tile_connection, topology, c):
        tilenum_layer = [0 for _ in range(layernum)]
        for i in range(layernum):
            layer_dict = structure_file[i][0][0]
            layer_type = layer_dict['type']
            weight_precision = int(layer_dict['Weightbit']) - 1
            #print(f"layer num={i}\n")
            #print(f"type={layer_dict['type']}\n")

            if layer_type == 'conv':
                '''
                mixmode2_area = (xbar_size_layer[i]**2)*PE_num_layer[i]**2
                remain_area=math.ceil(weight_precision) * math.ceil(int(layer_dict['Outputchannel']))\
                *math.ceil(int(layer_dict['Inputchannel']) *(int(layer_dict['Kernelsize']) ** 2))
                tilenum_layer[i] = math.ceil(remain_area/mixmode2_area)
                '''
                #print(f"Outputchannel={layer_dict['Outputchannel']}\n")
                #print(f"Inputchannel={layer_dict['Inputchannel']}\n")
                #print(f"Kernelsize={layer_dict['Kernelsize']}\n")
                mx = math.ceil(weight_precision) * math.ceil(int(layer_dict['Outputchannel']) / xbar_size_layer[i])
                my = math.ceil(int(layer_dict['Inputchannel']) / (xbar_size_layer[i] // (int(layer_dict['Kernelsize']) ** 2)))
                PEnum = mx * my
                tilenum_layer[i] = math.ceil(PEnum / (PE_num_layer[i]**2))
            elif layer_type == 'fc':
                '''
                mixmode2_area = (xbar_size_layer[i]**2)*PE_num_layer[i]**2
                remain_area=math.ceil(weight_precision) * math.ceil(int(layer_dict['Outfeature']))\
                *math.ceil(int(layer_dict['Infeature']))
                tilenum_layer[i] = math.ceil(remain_area/mixmode2_area)
                '''
                #print(f"Outfeature={layer_dict['Outfeature']}\n")
                #print(f"Infeature={layer_dict['Infeature']}\n")
                mx = math.ceil(weight_precision) * math.ceil(int(layer_dict['Outfeature']) / xbar_size_layer[i])
                my = math.ceil(int(layer_dict['Infeature']) / xbar_size_layer[i])
                PEnum = mx * my
                tilenum_layer[i] = math.ceil(PEnum / (PE_num_layer[i]**2))
            elif layer_type == 'pooling':
                mx = 1
                my = 1
                PEnum = mx * my
                PE_num_layer[i] = 1
                xbar_size_layer[i] = 32
                tilenum_layer[i] = math.ceil(PEnum / (PE_num_layer[i]**2))
            elif layer_type == 'element_sum':
                mx = 0
                my = 0
                PEnum = mx * my
                PE_num_layer[i] = 1
                xbar_size_layer[i] = 32
                tilenum_layer[i] = math.ceil(PEnum / (PE_num_layer[i]**2))
            elif layer_type == 'element_multiply':
                mx = 0
                my = 0
                PEnum = mx * my
                PE_num_layer[i] = 1
                xbar_size_layer[i] = 32
                tilenum_layer[i] = math.ceil(PEnum / (PE_num_layer[i]**2))
        for i in range(layernum):
            if i > 0:
                tilenum_layer[i] = tilenum_layer[i] + tilenum_layer[i-1]
        tilenum_total = tilenum_layer[layernum-1]
        tilenum_total = math.ceil(math.sqrt(tilenum_total))
        if topology == 0:
            tilenum = tilenum_total
        elif topology == 1:
            tilenum = math.ceil(tilenum_total / c) * c
        tile_type_new = [['NVM' for _ in range(tilenum)] for _ in range(tilenum)]
        PE_num_new = [[2 for _ in range(tilenum)] for _ in range(tilenum)]
        xbar_size_new = [[256 for _ in range(tilenum)] for _ in range(tilenum)]
        mapping_new = [['no' for _ in range(tilenum)] for _ in range(tilenum)]
        mapping_order = np.full((2, 2), -1)
        if topology == 0:
            if tile_connection == 0:
                mapping_order = self.generate_normal_matrix(tilenum, tilenum)
            elif tile_connection == 1:
                mapping_order = self.generate_snake_matrix(tilenum, tilenum)
            elif tile_connection == 2:
                mapping_order = self.generate_hui_matrix(tilenum, tilenum)
            elif tile_connection == 3:
                mapping_order = self.generate_zigzag_matrix(tilenum, tilenum)
        elif topology == 1:
            if tile_connection == 0:
                mapping_order = self.generate_normal_matrix_cmesh(tilenum, tilenum, c)
            elif tile_connection == 1:
                mapping_order = self.generate_snake_matrix_cmesh(tilenum, tilenum, c)
            elif tile_connection == 2:
                mapping_order = self.generate_zigzag_matrix_cmesh(tilenum, tilenum, c)    
            elif tile_connection == 3:
                mapping_order = self.generate_zigzag_matrix_cmesh(tilenum, tilenum, c)
        if mapping_order[0][0] == -1:
            embed()
        for i in range(tilenum):
            for j in range(tilenum):
                for m in range(layernum):
                    if mapping_order[i][j] < tilenum_layer[m]:
                        tile_type_new[i][j] = tile_type_layer[m]
                        PE_num_new[i][j] = PE_num_layer[m]
                        xbar_size_new[i][j] = xbar_size_layer[m]
                        mapping_new[i][j] = m
                        break

        return tilenum, tile_type_new,  PE_num_new, xbar_size_new, mapping_new

    def generate_new_tile_config(self, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection, topology, c):
        tilenum_layer = [0 for _ in range(self.layernum)]
        for i in range(self.layernum):
            if i > 0:
                tilenum_layer[i] = layertilenum[i] + tilenum_layer[i-1]
            else:
                tilenum_layer[i] = layertilenum[i]
        
        tilenum_total = tilenum_layer[self.layernum-1]
        tilenum_total = math.ceil(math.sqrt(tilenum_total))
        if topology == 0:
            tilenum = tilenum_total
        elif topology == 1:
            tilenum = math.ceil(tilenum_total / c) * c
        if topology == 0:
            if tile_connection == 0:
                mapping_order = self.generate_normal_matrix(tilenum, tilenum)
            elif tile_connection == 1:
                mapping_order = self.generate_snake_matrix(tilenum, tilenum)
            elif tile_connection == 2:
                mapping_order = self.generate_hui_matrix(tilenum, tilenum)
            elif tile_connection == 3:
                mapping_order = self.generate_zigzag_matrix(tilenum, tilenum)
        elif topology == 1:
            if tile_connection == 0:
                mapping_order = self.generate_normal_matrix_cmesh(tilenum, tilenum, c)
            elif tile_connection == 1:
                mapping_order = self.generate_snake_matrix_cmesh(tilenum, tilenum, c)
            elif tile_connection == 2:
                mapping_order = self.generate_zigzag_matrix_cmesh(tilenum, tilenum, c)    
            elif tile_connection == 3:
                mapping_order = self.generate_zigzag_matrix_cmesh(tilenum, tilenum, c)
        if mapping_order[0][0] == -1:
            embed()
        
        tile_type_new = [['NVM' for _ in range(tilenum)] for _ in range(tilenum)]
        PE_num_new = [[2 for _ in range(tilenum)] for _ in range(tilenum)]
        xbar_size_new = [[256 for _ in range(tilenum)] for _ in range(tilenum)]
        mapping_new = [['no' for _ in range(tilenum)] for _ in range(tilenum)]

        for i in range(tilenum):
            for j in range(tilenum):
                for m in range(self.layernum):
                    if mapping_order[i][j] < tilenum_layer[m]:
                        x = int(layertilenum[m]-(tilenum_layer[m]-mapping_order[i][j]))
                        tile_type_new[i][j] = tile_type_layer_tile[m][x]
                        PE_num_new[i][j] = PE_num_layer_tile[m][x]
                        xbar_size_new[i][j] = xbar_size_layer_tile[m][x]
                        mapping_new[i][j] = m
                        break

        return tilenum, tile_type_new,  PE_num_new, xbar_size_new, mapping_new

    def Metrospolis(self, f, f_new):
        return 1 if f_new <= f or random.random() < math.exp((f - f_new) / (self.T * self.k)) else 0

    def best(self):
        return min(self.history['f']) if self.history['f'] else float('inf')
    
    def HMSIM(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection, topology, c):
        area = []
        power = [] 
        latency = [] 
        energy = []
        NoC_area = 0
        NoC_power = 0
        self.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection, topology, c)
        result = subprocess.run(['python', 'main.py','--weights', self.weightpath,'--NN', self.net,'--dataset', self.dataset, '--mix_mode', self.mix_mode], capture_output=True, text=True)
        #print(result.stdout)
        lines = result.stdout.strip().split('\n')
        for item in lines:
            if "Entire latency:" in item:              
                parts = item.split(":")
                latency_str = parts[1].strip()
                latency_str = latency_str.replace(' ns', '')
                latency.append(float(latency_str))
            if "Hardware area:" in item:              
                parts = item.split(":")
                area_str = parts[1].strip()
                area_str = area_str.replace(' um^2', '')
                area.append(float(area_str))
            if "Hardware power:" in item:              
                parts = item.split(":")
                power_str = parts[1].strip()
                power_str = power_str.replace(' W', '')
                power.append(float(power_str))
            if "Final Total Area:" in item:              
                parts = item.split(":")
                area_str = parts[1].strip()
                area_str = area_str.replace(' um^2', '')
                NoC_area = float(area_str)
            if "Final Total Power:" in item:              
                parts = item.split(":")
                power_str = parts[1].strip()
                power_str = power_str.replace(' W', '')
                NoC_power = float(power_str)
            if "Floorplan area total:" in item:              
                parts = item.split(":")
                area_str = parts[1].strip()
                area_str = area_str.replace(' um^2', '')
                area_floorplan = float(area_str)
            if "Hardware energy:" in item:
                parts = item.split(":")
                energy_str = parts[1].strip()
                energy_str = energy_str.replace(' nJ', '')
                energy.append(float(energy_str))
        if len(area) == 0 or len(power) == 0 or len(latency) == 0:
            return 0, 0, 0, 0
        else:
            return area_floorplan, power[0], latency[0], 0
    
    def HMSIM_SimConfig(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection, topology, c):
        with open('mix_tileinfo.ini', 'w') as file:
            file.write(f"[tile]\n")
            file.write(f"tile_num={tilenum_new},{tilenum_new}\n")
            file.write(f"\n")
            for i in range(tilenum_new):
                file.write(f"device_type{i}=")
                for j in range(tilenum_new):
                    if j == tilenum_new - 1 :
                        file.write(f"{tile_type_new[i][j]}\n")
                    else :
                        file.write(f"{tile_type_new[i][j]},")
            file.write(f"\n")
            for i in range(tilenum_new):
                file.write(f"PE_num{i}=")
                for j in range(tilenum_new):
                    if j == tilenum_new - 1 :
                        file.write(f"{PE_num_new[i][j]}\n")
                    else :
                        file.write(f"{PE_num_new[i][j]},")
            file.write(f"\n")
            file.write(f"PE_group=1\n")
            file.write(f"\n")
            for i in range(tilenum_new):
                file.write(f"xbar_size{i}=")
                for j in range(tilenum_new):
                    if j == tilenum_new - 1 :
                        file.write(f"{xbar_size_new[i][j]}\n")
                    else :
                        file.write(f"{xbar_size_new[i][j]},")
            file.write(f"\n")
            for i in range(tilenum_new):
                file.write(f"layer_map_mix{i}=")
                for j in range(tilenum_new):
                    if j == tilenum_new - 1 :
                        file.write(f"{mapping_new[i][j]}\n")
                    else :
                        file.write(f"{mapping_new[i][j]},")
            file.write(f"\n")
            file.write(f"auto_layer_mapping={self.auto_layer_mapping}\n")
            file.write(f"\n")
            file.write(f"tile_connection={tile_connection}\n")
            file.write(f"\n")
            file.write(f"topology={topology}\n")
            file.write(f"\n")
            file.write(f"c={c}\n")
            self.update_ini_file('./SimConfig.ini',tile_connection)

    def HMSIM_SimConfig_self(self,filename):
        with open(filename, 'w') as file:
            file.write(f"[tile]\n")
            file.write(f"tile_num={self.tilenum},{self.tilenum}\n")
            file.write(f"\n")
            for i in range(self.tilenum):
                file.write(f"device_type{i}=")
                for j in range(self.tilenum):
                    if j == self.tilenum - 1 :
                        file.write(f"{self.tile_type[i][j]}\n")
                    else :
                        file.write(f"{self.tile_type[i][j]},")
            file.write(f"\n")
            for i in range(self.tilenum):
                file.write(f"PE_num{i}=")
                for j in range(self.tilenum):
                    if j == self.tilenum - 1 :
                        file.write(f"{self.PE_num[i][j]}\n")
                    else :
                        file.write(f"{self.PE_num[i][j]},")
            file.write(f"\n")
            file.write(f"PE_group=1\n")
            file.write(f"\n")
            for i in range(self.tilenum):
                file.write(f"xbar_size{i}=")
                for j in range(self.tilenum):
                    if j == self.tilenum - 1 :
                        file.write(f"{self.xbar_size[i][j]}\n")
                    else :
                        file.write(f"{self.xbar_size[i][j]},")
            file.write(f"\n")
            for i in range(self.tilenum):
                file.write(f"layer_map_mix{i}=")
                for j in range(self.tilenum):
                    if j == self.tilenum - 1 :
                        file.write(f"{self.mapping[i][j]}\n")
                    else :
                        file.write(f"{self.mapping[i][j]},")
            file.write(f"\n")
            file.write(f"auto_layer_mapping={self.auto_layer_mapping}\n")
            file.write(f"\n")
            file.write(f"tile_connection={self.tile_connection}\n")
            file.write(f"topology={self.topology}\n")
            file.write(f"\n")
            file.write(f"c={self.c}\n")
            self.update_ini_file('./SimConfig.ini',self.tile_connection)
        with open("SA.txt", "a") as file:  
            self.layertilenum = [0 for _ in range(self.layernum)]
            for i in range(self.tilenum):
                for j in range(self.tilenum):
                    if (self.mapping[i][j] != 'no'):
                        self.layertilenum[self.mapping[i][j]] += 1
            file.write(f"layertilenum={self.layertilenum}\n")


    def run_layer(self, net='vgg8', dataset='cifar10',tiletype='NVM',penum=1,xbarsize=1024,tileconnext=0,topology=0,c=2):
        start_time = time.time()
        home_path = os.getcwd()
        self.net = net
        self.dataset = dataset
        weight_path = os.path.join(home_path, f"{dataset}_{net}_params.pth") 
        self.weightpath = weight_path
        SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
        __TestInterface = TrainTestInterface(network_module=net, dataset_module=f"MNSIM.Interface.{dataset}", SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
        structure_file = __TestInterface.get_structure()
        self.layernum = len(structure_file)
        self.tile_type_layer = [tiletype for _ in range(self.layernum)]
        self.PE_num_layer = [penum for _ in range(self.layernum)]
        self.xbar_size_layer = [xbarsize for _ in range(self.layernum)]
        self.layertilenum = [1 for _ in range(self.layernum)]
        self.tile_connection = tileconnext
        self.topology = topology
        self.c = c
        self.tilenum, self.tile_type, self.PE_num, self.xbar_size, self.mapping = self.generate_new_layer_config(self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, structure_file, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy= self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time, tilenum = self.tilenum,tile_type = self.tile_type_layer, PE_num = self.PE_num_layer, xbar_size = self.xbar_size_layer, tile_connect = self.tile_connection, topology = self.topology, c = self.c)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency, self.energy))
        self.history['T'].append(self.T)
        while self.T > self.Tf:
            self.step=self.step+1
            layernum = self.layernum
            tile_type_layer = self.tile_type_layer.copy()
            PE_num_layer= self.PE_num_layer.copy()
            xbar_size_layer = self.xbar_size_layer.copy()
            tile_connection = self.tile_connection
            topology = self.topology
            tilenum = self.tilenum
            c = self.c
            search_num = int(random.random() * (self.T/4))
            for i in range(search_num+1):
                tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection, topology, c = self.generate_new_layer(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection, topology, c)
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_layer_config(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, tile_connection, topology, c)
            f = self.func(self.area, self.power, self.latency, self.energy)
            area, power, latency, energy = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection, topology, c)
            f_new = self.func(area, power, latency, energy)
            #print(self.latency)
            if(f_new >= 0) :
                if self.Metrospolis(f, f_new):
                    self.tilenum, self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, self.tile_connection, self.topology, c=tilenum_new, layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection, topology, c
                    self.area, self.power, self.latency, self.energy = area, power, latency, energy
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time, tilenum = self.tilenum, tile_type = self.tile_type_layer, PE_num = self.PE_num_layer, xbar_size = self.xbar_size_layer, tile_connect = self.tile_connection, topology = self.topology, c = self.c)
            device1.write_to_file("SA.txt")
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_layer_config(self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, structure_file, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time, tilenum = self.tilenum,tile_type = self.tile_type_layer, PE_num = self.PE_num_layer, xbar_size = self.xbar_size_layer, tile_connect = self.tile_connection, topology = self.topology, c = self.c)
        device1.write_to_file("SA.txt")
        self.HMSIM_SimConfig_self('mix_tileinfo.ini')
        print(f"Optimal F={self.most_best[-1][1]}")

    def run_tile(self, net, dataset,tile_type_layer, PE_num_layer, xbar_size_layer, layertilenum, tile_connection, topology, c):
        start_time = time.time()     
        home_path = os.getcwd()
        self.net = net
        self.dataset = dataset
        weight_path = os.path.join(home_path, f"{dataset}_{net}_params.pth") 
        self.weightpath = weight_path
        SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
        __TestInterface = TrainTestInterface(network_module=net, dataset_module=f"MNSIM.Interface.{dataset}", SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
        structure_file = __TestInterface.get_structure()
        self.layernum = len(structure_file)
        self.tile_type_layer = tile_type_layer
        self.PE_num_layer = PE_num_layer
        self.xbar_size_layer = xbar_size_layer
        self.layertilenum = layertilenum
        self.tile_type_layer_tile = [[self.tile_type_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.PE_num_layer_tile = [[self.PE_num_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.xbar_size_layer_tile = [[self.xbar_size_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.tile_connection = tile_connection
        self.topology = topology
        self.c = c
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping= self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency, self.energy))
        self.history['T'].append(self.T)
        stepnum = 0
        while self.T > self.Tf:
            self.step=self.step+1
            layertilenum = self.layertilenum.copy()
            tile_type_layer_tile = copy.deepcopy(self.tile_type_layer_tile)
            PE_num_layer_tile = copy.deepcopy(self.PE_num_layer_tile)
            xbar_size_layer_tile = copy.deepcopy(self.xbar_size_layer_tile)
            tile_connection_tile = self.tile_connection
            topology = self.topology
            c = self.c
            search_num = int(random.random() * (self.T / 4))
            layer = stepnum % self.layernum
            stepnum = stepnum + 1
            while (structure_file[layer][0][0]['type'] == 'pooling' or structure_file[layer][0][0]['type'] == 'element_sum'):
                layer = stepnum % self.layernum
                stepnum = stepnum + 1
            for i in range(search_num+1):
                layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c= self.generate_new_tile(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer, topology, c)
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_tile_config(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c)
            f = self.func(self.area, self.power, self.latency, self.energy)
            area, power, latency, energy = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection_tile, topology, c)
            f_new = self.func(area, power, latency, energy)
            #print(self.latency)
            if(f_new >= 0) :
                if self.Metrospolis(f, f_new):
                    self.tilenum, self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology,  self.c=tilenum_new, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c
                    self.area, self.power, self.latency, self.energy = area, power, latency, energy
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
            device1.write_to_file("SA.txt")
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.HMSIM_SimConfig_self('mix_tileinfo.ini')
        print(f"Optimal F={self.most_best[-1][1]}")

    def run_tile_all_space(self, net, dataset,tiletype, penum, xbarsize, tile_connection, topology, c):
        start_time = time.time()     
        home_path = os.getcwd()
        self.net = net
        self.dataset = dataset
        weight_path = os.path.join(home_path, f"{dataset}_{net}_params.pth") 
        self.weightpath = weight_path
        SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
        __TestInterface = TrainTestInterface(network_module=net, dataset_module=f"MNSIM.Interface.{dataset}", SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
        structure_file = __TestInterface.get_structure()
        self.layernum = len(structure_file)
        self.tile_type_layer = [tiletype for _ in range(self.layernum)]
        self.PE_num_layer = [penum for _ in range(self.layernum)]
        self.xbar_size_layer = [xbarsize for _ in range(self.layernum)]
        self.tile_connection = tile_connection
        self.topology = topology
        self.c = c
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_layer_config(self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, structure_file, self.tile_connection, self.topology, self.c)
        self.HMSIM_SimConfig_self('mix_tileinfo.ini')
        self.tile_type_layer_tile = [[self.tile_type_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.PE_num_layer_tile = [[self.PE_num_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.xbar_size_layer_tile = [[self.xbar_size_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping= self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        if (self.a == 1):
            self.area_des = self.area
        if (self.b == 1):
            self.power_des = self.power
        if (self.d == 1):
            self.latency_des = self.latency
        if (self.e == 1):
            self.energy_des = self.energy
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency, self.energy))
        self.history['T'].append(self.T)
        stepnum = 0
        while self.T > self.Tf:
            self.step=self.step+1
            layertilenum = self.layertilenum.copy()
            tile_type_layer_tile = copy.deepcopy(self.tile_type_layer_tile)
            PE_num_layer_tile = copy.deepcopy(self.PE_num_layer_tile)
            xbar_size_layer_tile = copy.deepcopy(self.xbar_size_layer_tile)
            tile_connection_tile = self.tile_connection
            topology = self.topology
            c = self.c
            search_num = int(random.random() * (self.T / 4))
            layer = stepnum % self.layernum
            stepnum = stepnum + 1
            while (structure_file[layer][0][0]['type'] == 'pooling' or structure_file[layer][0][0]['type'] == 'element_sum'):
                layer = stepnum % self.layernum
                stepnum = stepnum + 1
            for i in range(search_num+1):
                layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c= self.generate_new_tile_all_space(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer, topology, c)
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_tile_config(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c)
            f = self.func(self.area, self.power, self.latency, self.energy)
            area, power, latency, energy = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection_tile, topology, c)
            f_new = self.func(area, power, latency, energy)
            #print(self.latency)
            if(f_new >= 0) :
                if self.Metrospolis(f, f_new):
                    self.tilenum, self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology,  self.c= tilenum_new, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile, topology, c
                    self.area, self.power, self.latency, self.energy = area, power, latency, energy
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
            device1.write_to_file("SA.txt")
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection, self.topology, self.c)
        self.area, self.power, self.latency, self.energy = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection, self.topology, self.c)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, energy=self.energy, f=(self.func(self.area, self.power, self.latency, self.energy)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.HMSIM_SimConfig_self('mix_tileinfo.ini')
        if self.T < self.T0:
            print(f"Optimal F={self.most_best[-1][1]}")

class Device:
    def __init__(self, T, area, power, latency, energy, f, time, tilenum = None,tile_type = None, PE_num = None, xbar_size = None, tile_connect = None, topology = None, c = None):
        self.T = T
        self.area = area
        self.power = power
        self.latency = latency
        self.f = f
        self.time = time
        self.tilenum = tilenum
        self.tile_type = tile_type
        self.PE_num = PE_num
        self.xbar_size = xbar_size
        self.tile_connect = tile_connect
        self.topology = topology
        self.c = c
        self.energy = energy

    def print_info(self):
        print(f"T={self.T}\n")
        print(f"area={self.area}um^2\n")
        print(f"power={self.power}w\n")
        print(f"latency={self.latency}ns\n")
        print(f"f={self.f}\n")
        print(f"time={self.time}s\n")

    def write_to_file(self, filename):
        with open(filename, "a") as file:  
            file.write(f"T={self.T}\n")
            file.write(f"area={self.area}um^2\n")
            file.write(f"power={self.power}w\n")
            file.write(f"latency={self.latency}ns\n")
            file.write(f"energy={self.energy}nJ\n")
            file.write(f"f={self.f}\n")
            file.write(f"time={self.time}s\n")
            file.write(f"tilenum={self.tilenum}\n")
            file.write(f"tile_type={self.tile_type}\n")
            file.write(f"PE_num={self.PE_num}\n")
            file.write(f"xbar_size={self.xbar_size}\n")
            file.write(f"tile_connect={self.tile_connect}\n")
            file.write(f"topology={self.topology}\n")
            file.write(f"c={self.c}\n")
            file.write("\n")  


class Mix_Tile:
    def __init__(self, Tile_num, PE_num, Xbar_size, Type, Kernelsize, Outputchannel, Inputchannel, weight_precision):
        self.Tilenum = Tile_num
        self.PEnum = PE_num
        self.Xbarsize = Xbar_size
        self.Type = Type
        self.Kernelsize = Kernelsize
        self.Outputchannel = Outputchannel
        self.Inputchannel = Inputchannel
        self.weight_precision = weight_precision
        self.PEchoice = [1,2,4,8,16,32]
        self.Xbarchoice = [32,64,128,256,512,1024]
        self.tile = [[0 for _ in range(6)] for _ in range(6)] 
        for i in range(self.Tilenum):
            x = self.PEchoice.index(self.PEnum[i])
            y = self.Xbarchoice.index(self.Xbarsize[i])
            self.tile[y][x] = self.tile[y][x] + 1
        self.penum_per_pegroup = []
        self.inputchannel_per_pegroup = []
        for i in range(6):
            self.penum_per_pegroup.append(math.ceil(self.weight_precision) * math.ceil((self.Outputchannel) / self.Xbarchoice[i]))
            self.inputchannel_per_pegroup.append(self.Xbarchoice[i] // (int(self.Kernelsize) ** 2))

    def Sum_channel(self):
        sumchannel = 0
        for i in range(6):
            sumchannel_part = 0
            for j in range(6):
                sumchannel_part = sumchannel_part + self.tile[i][j] * self.PEchoice[j]**2
            sumchannel = sumchannel + math.floor(sumchannel_part / self.penum_per_pegroup[i]) * self.inputchannel_per_pegroup[i]
        return sumchannel
    
    def Change_PE(self, index, src, des):
        PE_indexsrc = self.PEchoice.index(src)
        PE_indexdes = self.PEchoice.index(des)
        xbarsize = self.Xbarsize[index]
        type = self.Type[index]
        Xbar_index = self.Xbarchoice.index(xbarsize)
        self.tile[Xbar_index][PE_indexsrc] = self.tile[Xbar_index][PE_indexsrc] - 1
        self.tile[Xbar_index][PE_indexdes] = self.tile[Xbar_index][PE_indexdes] + 1
        if self.Sum_channel() < self.Inputchannel:
            change = 1
            while (self.Sum_channel() < self.Inputchannel):
                self.tile[Xbar_index][PE_indexdes] = self.tile[Xbar_index][PE_indexdes] + 1
                change = change + 1
            change2 = 0
            while (self.Sum_channel() >= self.Inputchannel):
                if self.tile[Xbar_index][PE_indexsrc] == 0:
                    change2 = change2 + 1
                    break
                self.tile[Xbar_index][PE_indexsrc] = self.tile[Xbar_index][PE_indexsrc] - 1
                change2 = change2 + 1
            self.Tilenum = self.Tilenum + change - 1 - change2 + 1
            self.PEnum[index] = des
            for _ in range(change-1):
                self.PEnum.insert(index, des)
                self.Xbarsize.insert(index, xbarsize)
                self.Type.insert(index, type)
            for _ in range(change2-1):
                search = int(random.random() *len(self.PEnum))
                while(self.PEnum[search] != src):
                    search = int(random.random() *len(self.PEnum))
                self.PEnum.pop(search)
                self.Xbarsize.pop(search)
                self.Type.pop(search)

        elif self.Sum_channel() >= self.Inputchannel:
            change = 0
            while (self.Sum_channel() >= self.Inputchannel):
                if self.tile[Xbar_index][PE_indexsrc] == 0:
                    change = change + 1
                    break
                self.tile[Xbar_index][PE_indexsrc] = self.tile[Xbar_index][PE_indexsrc] - 1
                change = change + 1
            self.Tilenum = self.Tilenum - change + 1
            self.PEnum[index] = des
            for _ in range(change-1):
                search = int(random.random() *len(self.PEnum))
                while(self.PEnum[search] != src):
                    search = int(random.random() *len(self.PEnum))
                self.PEnum.pop(search)
                self.Xbarsize.pop(search)
                self.Type.pop(search)
        return self.Tilenum, self.PEnum, self.Xbarsize
    
    def Change_Xbar(self, index, src, des):
        Xbar_indexsrc = self.Xbarchoice.index(src)
        Xbar_indexdes = self.Xbarchoice.index(des)
        penum = self.PEnum[index]
        type = self.Type[index]
        PE_index = self.PEchoice.index(penum)
        self.tile[Xbar_indexsrc][PE_index] = self.tile[Xbar_indexsrc][PE_index] - 1
        self.tile[Xbar_indexdes][PE_index] = self.tile[Xbar_indexdes][PE_index] + 1
        if self.Sum_channel() < self.Inputchannel:           
            change = 1
            while (self.Sum_channel() < self.Inputchannel):
                self.tile[Xbar_indexdes][PE_index] = self.tile[Xbar_indexdes][PE_index] + 1
                change = change + 1
            change2 = 0
            while (self.Sum_channel() >= self.Inputchannel):
                if self.tile[Xbar_indexsrc][PE_index] == 0:
                    change2 = change2 + 1
                    break
                self.tile[Xbar_indexsrc][PE_index] = self.tile[Xbar_indexsrc][PE_index] - 1
                change2 = change2 + 1
            self.Tilenum = self.Tilenum + change - 1 - change2 + 1
            self.Xbarsize[index] = des
            for _ in range(change-1):
                self.Xbarsize.insert(index, des)
                self.PEnum.insert(index, penum)
                self.Type.insert(index, type)
            for _ in range(change2-1):
                search = int(random.random() *len(self.Xbarsize))
                while(self.Xbarsize[search] != src):
                    search = int(random.random() *len(self.Xbarsize))
                self.PEnum.pop(search)
                self.Xbarsize.pop(search)
                self.Type.pop(search)
        elif self.Sum_channel() >= self.Inputchannel:
            change = 0
            while (self.Sum_channel() >= self.Inputchannel):
                if self.tile[Xbar_indexsrc][PE_index] == 0:
                    change = change + 1
                    break
                self.tile[Xbar_indexsrc][PE_index] = self.tile[Xbar_indexsrc][PE_index] - 1
                change = change + 1
            self.Tilenum = self.Tilenum - change + 1
            self.Xbarsize[index] = des
            for _ in range(change-1):
                search = int(random.random() *len(self.Xbarsize))
                while(self.Xbarsize[search] != src):
                    search = int(random.random() *len(self.Xbarsize))
                self.PEnum.pop(search)
                self.Xbarsize.pop(search)
                self.Type.pop(search)
        return self.Tilenum, self.PEnum, self.Xbarsize 

def SA_run(network='alexnet', dataset='cifar10', tiletype='NVM',penum=1, xbarsize=512, tile_connection=0, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=1, power_flag=1, latency_flag=1, energy_flag=0, area_des=100000000000, power_des=10000000, latency_des=1000000000, energy_des=1000000000000,mix_mode='2'):
    sa = SA(T0=100, Tf=10, alpha=0.99, k=1, a=area_flag, b=power_flag, d=latency_flag, e=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    sa.hetro=hetro
    sa.run_tile_all_space(network,dataset,tiletype, penum, xbarsize, tile_connection, topology, c)
    sa.HMSIM_SimConfig_self(f'mix_tileinfo_{type}_all.ini')
    sa.T = sa.T0
    sa.Tf = 10
    sa.step = 0
    sa.run_layer(net=network,dataset=dataset,tiletype=tiletype,penum=penum,xbarsize=xbarsize,tileconnext=tile_connection,topology=topology,c=c)
    sa.HMSIM_SimConfig_self(f'mix_tileinfo_{type}_layer.ini')
    tile_type_layer =sa.tile_type_layer
    PE_num_layer = sa.PE_num_layer
    xbar_size_layer = sa.xbar_size_layer
    layertilenum = sa.layertilenum
    tile_connection = sa.tile_connection
    topology = sa.topology
    c = sa.c
    sa.T = sa.T0
    sa.step = 0
    sa.run_tile(network,dataset,tile_type_layer, PE_num_layer, xbar_size_layer, layertilenum, tile_connection, topology, c)
    sa.HMSIM_SimConfig_self(f'mix_tileinfo_{type}_tile.ini')

def SA_run_alexnet_111():
    area_flag = 1
    power_flag = 1
    latency_flag = 1
    energy_flag = 0
    area_des = 1000000000
    power_des = 100000
    latency_des = 50000000
    energy_des = 1000000000
    network = 'alexnet'
    dataset = 'cifar10'

    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

def SA_run_resnet18_111():
    area_flag = 1
    power_flag = 1
    latency_flag = 1
    energy_flag = 0
    area_des = 1000000000
    power_des = 100000
    latency_des = 50000000
    energy_des = 1000000000
    network = 'resnet18'
    dataset = 'cifar10'
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

def SA_run_alexnet_011():
    area_flag = 0
    power_flag = 1
    latency_flag = 1
    energy_flag = 0
    area_des = 150000000
    power_des = 100000
    latency_des = 50000000
    energy_des = 1000000000
    network = 'alexnet'
    dataset = 'cifar10'

    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

def SA_run_resnet18_011():
    area_flag = 0
    power_flag = 1
    latency_flag = 1
    energy_flag = 0
    area_des = 400000000
    power_des = 100000
    latency_des = 50000000
    energy_des = 1000000000
    network = 'resnet18'
    dataset = 'cifar10'
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

def SA_run_alexnet_001():
    area_flag = 0
    power_flag = 0
    latency_flag = 1
    energy_flag = 0
    area_des = 150000000
    power_des = 25
    latency_des = 50000000
    energy_des = 1000000000
    network = 'alexnet'
    dataset = 'cifar10'

    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

def SA_run_resnet18_001():
    area_flag = 0
    power_flag = 0
    latency_flag = 1
    energy_flag = 0
    area_des = 400000000
    power_des = 90
    latency_des = 50000000
    energy_des = 1000000000
    network = 'resnet18'
    dataset = 'cifar10'
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=1, type='hetro',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='NVM',penum=1, xbarsize=1024, tile_connection=3, topology=0,c=2,hetro=0, type='nvm',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')
    SA_run(network=network, dataset=dataset, tiletype='SRAM',penum=1, xbarsize=512, tile_connection=3, topology=0,c=2,hetro=0, type='sram',T0=100, Tf=10, alpha=0.99, k=1, area_flag=area_flag, power_flag=power_flag, latency_flag=latency_flag, energy_flag=energy_flag, area_des=area_des, power_des=power_des, latency_des=latency_des, energy_des=energy_des,mix_mode='2')

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#SA_run_resnet18_111()
SA_run_alexnet_111()