import math
from random import random
import matplotlib.pyplot as plt
import subprocess
from IPython import embed
import random
import torch
import os
from MNSIM.Interface.interface import *
import configparser
import time
import copy

class SA:
    def __init__(self, T0=100, Tf=10, alpha=0.99, k=1):
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.k = k
        self.x = random.random() * 11 - 5  # 随机生成一个x的值
        self.y = random.random() * 11 - 5  # 随机生成一个y的值
        self.tilenum = 64
        self.tile_type = [['NVM' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.PE_num = [[1 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.xbar_size = [[1024 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.mapping = [['no' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.auto_layer_mapping = 0
        self.area = 0
        self.power = 0
        self.latency = 0
        self.most_best = []
        self.history = {'f': [], 'T': [], 'area': [], 'power': [], 'latency': [], 'tilenum': [], 'tile_type': [], 'PE_num': [], 'xbar_size': [], 'mapping': []}
        self.layernum = 12
        self.tile_type_layer = ['NVM' for _ in range(self.layernum)]
        self.PE_num_layer = [1 for _ in range(self.layernum)]
        self.xbar_size_layer = [1024 for _ in range(self.layernum)]
        self.layertilenum = [1 for _ in range(self.layernum)]
        self.tile_type_layer_tile = [[self.tile_type_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.PE_num_layer_tile = [[self.PE_num_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.xbar_size_layer_tile = [[self.xbar_size_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.area_des = 153050474.2					
        self.power_des =50
        self.latency_des = 2596162.411
        self.tile_connection = 3

    def update_ini_file(self, file_path, tile_connection):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        updated_lines = []
        for line in lines:
            if line.startswith('Tile_Connection'):
                line = f'Tile_Connection = {tile_connection}\n'
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
    
    def func2(self, area, power, latency):
        if area == 0 or power == 0 or latency == 0:
            return -1
        if area < self.area_des:
            area_data = 0
        else:
            area_data = abs(area-self.area_des)/self.area_des

        if power < self.power_des:
            power_data = 0
        else:
            power_data = abs(power-self.power_des)/self.power_des

        if latency < self.latency_des:
            latency_data = 0
        else:
            latency_data = abs(latency-self.latency_des)/self.latency_des
        similar_data = area_data + power_data + latency_data
        return similar_data*10000
    
    def func3(self, area, power, latency):
        if area == 0 or power == 0 or latency == 0:
            return -1

        area_data = area/self.area_des

        power_data = power/self.power_des

        latency_data = latency/self.latency_des
        similar_data = area_data + power_data + latency_data
        return similar_data*1000
    
    def func(self, area, power, latency):
        if area == 0 or power == 0 or latency == 0:
            return -1  
        #power_data = power/self.power_des
        area_data = area/self.area_des
        if power > self.power_des:
            return -1
        latency_data = latency/self.latency_des

        return area_data*latency_data*10000

    def generate_new(self, tilenum, tile_type, PE_num, xbar_size, mapping, tile_connection):
        tilenum_new = tilenum
        tile_type_new = tile_type
        PE_num_new = PE_num
        xbar_size_new = xbar_size
        mapping_new = mapping
        seach = random.random()
        if seach < 0:#随机改变NoC尺寸（Tile数量）
            match int(random.random() * 3):
                case 0:
                    tilenum_new = 16
                case 1:
                    tilenum_new = 32
                case 2:
                    tilenum_new = 64
        elif seach <= 0.05:#随机选择一个Tile，改变单元类型
            tile_row = int(random.random() * tilenum_new)  
            tile_line = int(random.random() * tilenum_new) 
            if tile_type_new[tile_row][tile_line] == 'SRAM' :
                tile_type_new[tile_row][tile_line] = 'NVM'
            else :
                tile_type_new[tile_row][tile_line] = 'SRAM'
        elif seach <= 1:#随机选择一个Tile，改变PE数量
            tile_row = int(random.random() * tilenum_new)  
            tile_line = int(random.random() * tilenum_new)
            if PE_num_new[tile_row][tile_line] == 2 :
                PE_num_new[tile_row][tile_line] = 4
            else :
                PE_num_new[tile_row][tile_line] = 2
            '''match int(random.random() * 4):
                case 0:
                    PE_num_new[tile_row][tile_line] = 1
                case 1:
                    PE_num_new[tile_row][tile_line] = 2
                case 2:
                    PE_num_new[tile_row][tile_line] = 1
                case 3:
                    PE_num_new[tile_row][tile_line] = 2'''
        elif seach <= 1:#随机选择一个Tile，改变Xbar尺寸
            tile_row = int(random.random() * tilenum_new)  
            tile_line = int(random.random() * tilenum_new)
            if xbar_size_new[tile_row][tile_line] == 1024 :
                xbar_size_new[tile_row][tile_line] = 512
            else :
                xbar_size_new[tile_row][tile_line] = 1024
            '''match int(random.random() * 4):
                case 0:
                    xbar_size_new[tile_row][tile_line] = 512
                case 1:
                    xbar_size_new[tile_row][tile_line] = 256
                case 2:
                    xbar_size_new[tile_row][tile_line] = 512  
                case 3:
                    xbar_size_new[tile_row][tile_line] = 256'''  
        else:#随机选择两个Tile，交换网络负载
            tile_row_0 = int(random.random() * tilenum_new)  
            tile_line_0 = int(random.random() * tilenum_new)
            tile_row_1 = int(random.random() * tilenum_new)  
            tile_line_1 = int(random.random() * tilenum_new)
            mapping_new[tile_row_0][tile_line_0] = mapping[tile_row_1][tile_line_1] 
            mapping_new[tile_row_1][tile_line_1] = mapping[tile_row_0][tile_line_0]
        return tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new          
    
    def generate_new_layer(self, layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection):
        seach = random.random()
        if seach <= 0.1:#随机选择一个层，改变单元类型
            layer = int(random.random() * layernum)  
            if tile_type_layer[layer] == 'SRAM' :
                tile_type_layer[layer] = 'NVM'
            else :
                tile_type_layer[layer] = 'SRAM'
        elif seach <= 0.5:#随机选择一个层，改变PE数量
            layer = int(random.random() * layernum)
            change = int(random.random() * 6)
            PE_num_layer[layer] = 2**(change)
        elif seach <= 0.9:#随机选择一个层，改变Xbar尺寸
            layer = int(random.random() * layernum)
            change = int(random.random() * 6)
            xbar_size_layer[layer] = 2**(change+5)
            '''if xbar_size_layer[layer] == 1024 :
                xbar_size_layer[layer] = 512
            else :
                xbar_size_layer[layer] = 1024''' 
        else :
            tile_connection = int(random.random() * 3)
            if tile_connection == 2:
                tile_connection = 3
        return tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection

    def generate_new_tile(self, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer):
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
        if search_choice < 0.1:#随机选择一个层，改变单元类型
            if tile_type_layer_tile[search][search_tile] == 'SRAM' :
                tile_type_layer_tile[search][search_tile] = 'NVM'
            else :
                tile_type_layer_tile[search][search_tile]= 'SRAM'
        elif search_choice <= 0.5:#随机选择一个层，改变PE数量
            change = int(random.random() * 3)
            des = 2**(change)
            if des != PE_num_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_PE(search_tile,PE_num_layer_tile[search][search_tile],des)
        elif search_choice <= 0.9:#随机选择一个层，改变Xbar尺寸
            change = int(random.random() * 4)
            des = 2**(change+7)
            if des != xbar_size_layer_tile[search][search_tile]:
                layertilenum[search], PE_num_layer_tile[search], xbar_size_layer_tile[search] = mix_chage.Change_Xbar(search_tile,xbar_size_layer_tile[search][search_tile],des)
        else: 
            tile_type_layer_tile[search][search_tile],tile_type_layer_tile[search][search_tile_0] = tile_type_layer_tile[search][search_tile_0],tile_type_layer_tile[search][search_tile]
            PE_num_layer_tile[search][search_tile],PE_num_layer_tile[search][search_tile_0] = PE_num_layer_tile[search][search_tile_0],PE_num_layer_tile[search][search_tile]
            xbar_size_layer_tile[search][search_tile],xbar_size_layer_tile[search][search_tile_0] = xbar_size_layer_tile[search][search_tile_0],xbar_size_layer_tile[search][search_tile]
        return layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile

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


    def generate_new_layer_config(self, layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, matrix):
        tilenum_new = self.tilenum
        tilenum_layer = [0 for _ in range(layernum)]
        tile_type_new = [['NVM' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        PE_num_new = [[1 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        xbar_size_new = [[1024 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        mapping_new = [['no' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
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
        for i in range(tilenum_new):
            for j in range(tilenum_new):
                for m in range(layernum):
                    if matrix[i][j] < tilenum_layer[m]:
                        tile_type_new[i][j] = tile_type_layer[m]
                        PE_num_new[i][j] = PE_num_layer[m]
                        xbar_size_new[i][j] = xbar_size_layer[m]
                        mapping_new[i][j] = m
                        break

        return tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new

    def generate_new_tile_config(self, layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, matrix):
        tilenum_new = self.tilenum
        tilenum_layer = [0 for _ in range(self.layernum)]
        tile_type_new = [['NVM' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        PE_num_new = [[1 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        xbar_size_new = [[1024 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        mapping_new = [['no' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        for i in range(self.layernum):
            if i > 0:
                tilenum_layer[i] = layertilenum[i] + tilenum_layer[i-1]
            else:
                tilenum_layer[i] = layertilenum[i]
        
        for i in range(tilenum_new):
            for j in range(tilenum_new):
                for m in range(self.layernum):
                    if matrix[i][j] < tilenum_layer[m]:
                        x = int(layertilenum[m]-(tilenum_layer[m]-matrix[i][j]))
                        tile_type_new[i][j] = tile_type_layer_tile[m][x]
                        PE_num_new[i][j] = PE_num_layer_tile[m][x]
                        xbar_size_new[i][j] = xbar_size_layer_tile[m][x]
                        mapping_new[i][j] = m
                        break

        return tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new

    def Metrospolis(self, f, f_new):
        return 1 if f_new <= f or random.random() < math.exp((f - f_new) / (self.T * self.k)) else 0

    def best(self):
        return min(self.history['f']) if self.history['f'] else float('inf')
    
    def HMSIM(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection):
        area = []
        power = [] 
        latency = [] 
        NoC_area = 0
        NoC_power = 0
        self.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection)
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
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
        if len(area) == 0 or len(power) == 0 or len(latency) == 0:
            return 0, 0, 0
        else:
            return area[0]+NoC_area, power[0]+NoC_power, latency[0]
    
    def HMSIM_SimConfig(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection):
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
            self.update_ini_file('./SimConfig.ini',tile_connection)

    def HMSIM_SimConfig_self(self):
        with open('mix_tileinfo.ini', 'w') as file:
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
            self.update_ini_file('./SimConfig.ini',self.tile_connection)
        with open("SA.txt", "a") as file:  
            layertilenum = [0 for _ in range(self.layernum)]
            for i in range(self.tilenum):
                for j in range(self.tilenum):
                    if (self.mapping[i][j] != 'no'):
                        layertilenum[self.mapping[i][j]] += 1
            file.write(f"layertilenum={layertilenum}\n")


    def run(self):
        start_time = time.time()
        self.area, self.power, self.latency = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, tile_connection)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency))
        self.history['T'].append(self.T)
        while self.T > self.Tf:
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection = self.tilenum, self.tile_type, self.PE_num, self.xbar_size, self.mapping, self.tile_connection
            search_num = int(random.random() * self.T * 1)
            for i in range(search_num+1):
                tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection = self.generate_new(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection)
            f = self.func(self.area, self.power, self.latency)
            area, power, latency = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection)
            f_new = self.func(area, power, latency)
            #print(self.latency)
            if(f_new >= 0) :
                '''if self.area_des >= area and self.latency_des >= latency and self.power_des >= power:
                    self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection = tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
                    # 更新温度
                    self.T *= self.alpha
                    # 记录当前最佳值
                    current_best = self.best()
                    self.most_best.append((self.T, current_best))
                    device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
                    device1.write_to_file("SA.txt")
                    break'''
                if self.Metrospolis(f, f_new):
                    self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection = tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
            device1.write_to_file("SA.txt")

        self.HMSIM_SimConfig_self()
        print(f"Optimal F={self.most_best[-1][1]}")

    def run_layer(self):
        start_time = time.time()
        matrix = []
        matrix.append(self.generate_normal_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_snake_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_hui_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_zigzag_matrix(self.tilenum, self.tilenum))
        home_path = os.getcwd()
        weight_path = os.path.join(home_path, "cifar10_vgg8_params.pth") 
        SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
        __TestInterface = TrainTestInterface(network_module='vgg8', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
        structure_file = __TestInterface.get_structure()
        self.layernum = len(structure_file)
        self.area, self.power, self.latency = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time, tile_type = self.tile_type_layer, PE_num = self.PE_num_layer, xbar_size = self.xbar_size_layer, tile_connect = self.tile_connection)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency))
        self.history['T'].append(self.T)
        while self.T > self.Tf:
            layernum = self.layernum
            tile_type_layer = self.tile_type_layer.copy()
            PE_num_layer= self.PE_num_layer.copy()
            xbar_size_layer = self.xbar_size_layer.copy()
            tile_connection = self.tile_connection
            #tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_layer_config(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, matrix)
            search_num = int(random.random() * (self.T/4))
            for i in range(search_num+1):
                tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection = self.generate_new_layer(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection)
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_layer_config(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, matrix[tile_connection])
            f = self.func(self.area, self.power, self.latency)
            area, power, latency = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection)
            f_new = self.func(area, power, latency)
            #print(self.latency)
            if(f_new >= 0 and area<=self.area_des) :
                '''if self.area_des >= area and self.latency_des >= latency and self.power_des >= power:
                    print(self.T)
                    print("change")
                    self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, self.tile_connection = layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
                    # 更新温度
                    self.T *= self.alpha
                    # 记录当前最佳值
                    current_best = self.best()
                    self.most_best.append((self.T, current_best))
                    device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time, tile_type = tile_type_layer, PE_num = PE_num_layer, xbar_size = xbar_size_layer, tile_connect = tile_connection)
                    device1.write_to_file("SA.txt")
                    break'''
                if self.Metrospolis(f, f_new):
                    print(self.T)
                    print("change")
                    self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, self.tile_connection = layernum, tile_type_layer, PE_num_layer, xbar_size_layer, tile_connection
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time, tile_type = self.tile_type_layer, PE_num = self.PE_num_layer, xbar_size = self.xbar_size_layer, tile_connect = self.tile_connection)
            device1.write_to_file("SA.txt")
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_layer_config(self.layernum, self.tile_type_layer, self.PE_num_layer, self.xbar_size_layer, structure_file, matrix[self.tile_connection])
        self.HMSIM_SimConfig_self()
        print(f"Optimal F={self.most_best[-1][1]}")

    def run_tile(self):
        start_time = time.time()
        matrix = []
        matrix.append(self.generate_normal_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_snake_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_hui_matrix(self.tilenum, self.tilenum))
        matrix.append(self.generate_zigzag_matrix(self.tilenum, self.tilenum))
        home_path = os.getcwd()
        weight_path = os.path.join(home_path, "cifar10_vgg8_params.pth") 
        SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
        __TestInterface = TrainTestInterface(network_module='vgg8', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
        structure_file = __TestInterface.get_structure()
        self.layernum = len(structure_file)
        self.tile_type_layer =['NVM', 'SRAM', 'SRAM', 'NVM', 'NVM', 'SRAM', 'NVM', 'NVM', 'NVM', 'NVM', 'SRAM', 'NVM']
        self.PE_num_layer = [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 4]
        self.xbar_size_layer = [256, 512, 32, 1024, 512, 32, 1024, 1024, 32, 1024, 32, 512]
        self.layertilenum = [8, 24, 1, 16, 10, 1, 24, 40, 1, 10, 1, 1]
        self.tile_type_layer_tile = [[self.tile_type_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.PE_num_layer_tile = [[self.PE_num_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.xbar_size_layer_tile = [[self.xbar_size_layer[i] for _ in range(self.layertilenum[i])] for i in range(self.layernum)]
        self.tile_connection = 3
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping= self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, matrix[self.tile_connection])
        self.area, self.power, self.latency = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping, self.tile_connection)
        device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
        device1.write_to_file("SA.txt")
        self.history['f'].append(self.func(self.area, self.power, self.latency))
        self.history['T'].append(self.T)
        stepnum = 0
        while self.T > self.Tf:
            layertilenum = self.layertilenum.copy()
            tile_type_layer_tile = copy.deepcopy(self.tile_type_layer_tile)
            PE_num_layer_tile = copy.deepcopy(self.PE_num_layer_tile)
            xbar_size_layer_tile = copy.deepcopy(self.xbar_size_layer_tile)
            tile_connection_tile = self.tile_connection
            search_num = int(random.random() * (self.T / 4))
            layer = stepnum % self.layernum
            stepnum = stepnum + 1
            while (structure_file[layer][0][0]['type'] == 'pooling' or structure_file[layer][0][0]['type'] == 'element_sum'):
                layer = stepnum % self.layernum
                stepnum = stepnum + 1
            for i in range(search_num+1):
                layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile = self.generate_new_tile(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, structure_file, tile_connection_tile, layer)
            tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = self.generate_new_tile_config(layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, matrix[tile_connection_tile])
            f = self.func(self.area, self.power, self.latency)
            area, power, latency = self.HMSIM(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, tile_connection_tile)
            f_new = self.func(area, power, latency)
            #print(self.latency)
            if(f_new >= 0 and area<=self.area_des) :
                '''if self.area_des >= area and self.latency_des >= latency and self.power_des >= power:
                    self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection = layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
                    # 更新温度
                    self.T *= self.alpha
                    # 记录当前最佳值
                    current_best = self.best()
                    self.most_best.append((self.T, current_best))
                    device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
                    device1.write_to_file("SA.txt")
                    break'''
                if self.Metrospolis(f, f_new):
                    self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, self.tile_connection = layertilenum, tile_type_layer_tile, PE_num_layer_tile, xbar_size_layer_tile, tile_connection_tile
                    self.area, self.power, self.latency = area, power, latency
                    self.history['f'].append(f_new)
            # 更新温度
            self.T *= self.alpha
            # 记录当前最佳值
            current_best = self.best()
            self.most_best.append((self.T, current_best))
            device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=(self.func(self.area, self.power, self.latency)), time = time.time()- start_time)
            device1.write_to_file("SA.txt")
        self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping = self.generate_new_tile_config(self.layertilenum, self.tile_type_layer_tile, self.PE_num_layer_tile, self.xbar_size_layer_tile, matrix[self.tile_connection])
        self.HMSIM_SimConfig_self()
        print(f"Optimal F={self.most_best[-1][1]}")

class Device:
    def __init__(self, T, area, power, latency, f, time, tile_type = None, PE_num = None, xbar_size = None, tile_connect = None):
        self.T = T
        self.area = area
        self.power = power
        self.latency = latency
        self.f = f
        self.time = time
        self.tile_type = tile_type
        self.PE_num = PE_num
        self.xbar_size = xbar_size
        self.tile_connect = tile_connect

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
            file.write(f"f={self.f}\n")
            file.write(f"time={self.time}s\n")
            file.write(f"tile_type={self.tile_type}\n")
            file.write(f"PE_num={self.PE_num}\n")
            file.write(f"xbar_size={self.xbar_size}\n")
            file.write(f"tile_connect={self.tile_connect}\n")
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

sa = SA()
sa.run_tile()


'''
sa = SA()
matrix = []
matrix.append(sa.generate_normal_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_snake_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_hui_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_zigzag_matrix(sa.tilenum, sa.tilenum))
home_path = os.getcwd()
weight_path = os.path.join(home_path, "cifar10_vgg8_params.pth") 
SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
__TestInterface = TrainTestInterface(network_module='vgg8', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
structure_file = __TestInterface.get_structure()
sa.layernum = len(structure_file)
sa.tile_type_layer = ['NVM' for _ in range(sa.layernum)]
sa.PE_num_layer = [1 for _ in range(sa.layernum)]
sa.xbar_size_layer = [512 for _ in range(sa.layernum)]
sa.layertilenum = [1 for _ in range(sa.layernum)]
sa.tile_type_layer_tile = [[sa.tile_type_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.PE_num_layer_tile = [[sa.PE_num_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.xbar_size_layer_tile = [[sa.xbar_size_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.tile_connection = 0
sa.generate_new_tile(sa.layertilenum, sa.tile_type_layer_tile, sa.PE_num_layer_tile, sa.xbar_size_layer_tile, structure_file, sa.tile_connection)
'''
'''
sa = SA()
matrix = []
matrix.append(sa.generate_normal_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_snake_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_hui_matrix(sa.tilenum, sa.tilenum))
matrix.append(sa.generate_zigzag_matrix(sa.tilenum, sa.tilenum))
home_path = os.getcwd()
weight_path = os.path.join(home_path, "cifar10_vgg8_params.pth") 
SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
__TestInterface = TrainTestInterface(network_module='vgg8', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
structure_file = __TestInterface.get_structure()
sa.layernum = len(structure_file)
sa.tile_type_layer = ['NVM' for _ in range(sa.layernum)]
sa.PE_num_layer = [2 for _ in range(sa.layernum)]
sa.xbar_size_layer = [256 for _ in range(sa.layernum)]
sa.layertilenum = [2, 10, 1, 10, 20, 1, 40, 76, 1, 152, 1, 8]
#tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = sa.generate_new_layer_config(sa.layernum, sa.tile_type_layer, sa.PE_num_layer, sa.xbar_size_layer, structure_file, matrix[0])
#sa.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, 0)
sa.tile_type_layer_tile = [[sa.tile_type_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.PE_num_layer_tile = [[sa.PE_num_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.xbar_size_layer_tile = [[sa.xbar_size_layer[i] for _ in range(sa.layertilenum[i])] for i in range(sa.layernum)]
sa.tile_connection = 0
sa.layertilenum, sa.tile_type_layer_tile, sa.PE_num_layer_tile, sa.xbar_size_layer_tile, sa.tile_connection = sa.generate_new_tile(sa.layertilenum, sa.tile_type_layer_tile, sa.PE_num_layer_tile, sa.xbar_size_layer_tile, structure_file, sa.tile_connection)
tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = sa.generate_new_tile_config(sa.layertilenum, sa.tile_type_layer_tile, sa.PE_num_layer_tile, sa.xbar_size_layer_tile, matrix[sa.tile_connection])
sa.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, 0)
#sa.generate_new_tile(sa.layertilenum, sa.tile_type_layer_tile, sa.PE_num_layer_tile, sa.xbar_size_layer_tile, structure_file, sa.tile_connection)
'''

'''
sa = SA()
layernum = 10
tile_type_layer = ['NVM','SRAM','NVM','SRAM','NVM','SRAM','NVM','SRAM','NVM','SRAM']
PE_num_layer = [2,4,2,4,2,4,2,4,2,4]
xbar_size_layer = [256,256,256,256,256,256,256,256,256,128]
matrix = sa.generate_zigzag_matrix(16, 16)
home_path = os.getcwd()
weight_path = os.path.join(home_path, "cifar10_alexnet_99bit_params.pth") 
SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
__TestInterface = TrainTestInterface(network_module='alexnet', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
structure_file = __TestInterface.get_structure()
tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = sa.generate_new_layer_config(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, matrix)
sa.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, 3)
'''
'''
sa = SA()
layernum = 12
tile_type_layer = ['NVM' for _ in range(12)]
PE_num_layer = [1 for _ in range(12)]
xbar_size_layer = [512 for _ in range(12)]
matrix = sa.generate_normal_matrix(64, 64)
home_path = os.getcwd()
weight_path = os.path.join(home_path, "cifar10_vgg8_params.pth") 
SimConfig_path = os.path.join(home_path, "SimConfig.ini") 
__TestInterface = TrainTestInterface(network_module='vgg8', dataset_module='MNSIM.Interface.cifar10', SimConfig_path=SimConfig_path, weights_file=weight_path, device=0)
structure_file = __TestInterface.get_structure()
print(len(structure_file))
for i in range(len(structure_file)):
    print(structure_file[i][0][0]['type'])
tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new = sa.generate_new_layer_config(layernum, tile_type_layer, PE_num_layer, xbar_size_layer, structure_file, matrix)
sa.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new, 0)
'''

'''
sa = SA()
sa.run()


plt.plot(sa.history['T'], sa.history['f'])
plt.plot([x[0] for x in sa.most_best], [x[1] for x in sa.most_best], 'ro-')  # 绘制最佳值
plt.title('Simulated Annealing')
plt.xlabel('Temperature (T)')
plt.ylabel('Function Value (f)')
plt.gca().invert_xaxis()  
plt.show()
'''