import math
from random import random
import matplotlib.pyplot as plt
import subprocess
from IPython import embed
import random
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import random

class Nohetergeneous:
    def __init__(self):
        self.tilenum = 64
        self.tile_type = [['SRAM' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.PE_num = [[8 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.xbar_size = [[256 for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.mapping = [['no' for _ in range(self.tilenum)] for _ in range(self.tilenum)]
        self.auto_layer_mapping = 1
        self.area = 0
        self.power = 0
        self.latency = 0
        self.T = 0
        self.tile_connection = 0

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
    
    def func(self, area, power, latency):                 
        a = 1
        b = 1
        c = 1
        res = ((area)**a) * ((power)**b) * ((latency)**c)/(10**15)
        return res

    
    def HMSIM(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new):
        area = []
        power = [] 
        latency = [] 
        NoC_area = 0
        NoC_power = 0
        self.HMSIM_SimConfig(tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new)
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
    
    def HMSIM_SimConfig(self, tilenum_new, tile_type_new,  PE_num_new, xbar_size_new, mapping_new):
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
            file.write(f"tile_connection={self.tile_connection}\n")

    def HMSIM_SimConfig_self(self):
        with open('test.ini', 'w') as file:
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

    def run(self):
        
        
        tilenum = [64]
        tiletype = ['NVM']
        PEnum = [1]
        xbarsize = [1024]
        tileconnetion = [3]
        self.tile_connection = 0
        '''
        tilenum = [64]
        tiletype = ['SRAM','NVM']
        PEnum = [1,2,4,8,16,32]
        xbarsize = [32,64,128,256,512,1024]
        tileconnetion = [0,1,3] 
        '''
        

        for i in range(len(tilenum)):
            for j in range(len(tiletype)):
                for m in range(len(PEnum)):
                    for n in range(len(xbarsize)):
                        for k in range(len(tileconnetion)):
                            self.tilenum = tilenum[i]
                            self.tile_type = [[tiletype[j] for _ in range(self.tilenum)] for _ in range(self.tilenum)]
                            self.PE_num = [[PEnum[m] for _ in range(self.tilenum)] for _ in range(self.tilenum)]
                            self.xbar_size = [[xbarsize[n] for _ in range(self.tilenum)] for _ in range(self.tilenum)]
                            self.tile_connection = tileconnetion[k]
                            self.update_ini_file('./SimConfig.ini',self.tile_connection)
                            self.area, self.power, self.latency = self.HMSIM(self.tilenum, self.tile_type,  self.PE_num, self.xbar_size, self.mapping)
                            f=self.func(self.area, self.power, self.latency)
                            self.T = k + n*len(tileconnetion) + m*len(tileconnetion)*len(xbarsize) + j*len(tileconnetion)*len(xbarsize)*len(PEnum) + i*len(tileconnetion)*len(xbarsize)*len(PEnum)*len(tiletype)
                            #device1 = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=f)
                            #device1.write_to_file("Nohetergeneous.txt.txt")
                            device = Device(T=self.T, area=self.area, power=self.power, latency=self.latency, f=f)
                            device.write_to_csv('Nohetergeneous.csv')
                            print(f"T={self.T}\n")
                            print(f"area={self.area}um^2\n")
                            print(f"power={self.power}W\n")
                            print(f"latency={self.latency}ns\n")
                            print(f"f={f}\n")
        
        csv_file = 'Nohetergeneous.csv'
        excel_file = 'Nohetergeneous.xlsx'
        df = pd.read_csv(csv_file)
        df.to_excel(excel_file, index=False)

class Device:
    def __init__(self, T, area, power, latency, f):
        self.T = T
        self.area = area
        self.power = power
        self.latency = latency
        self.f = f

    def print_info(self):
        print(f"T={self.T}\n")
        print(f"area={self.area}um^2\n")
        print(f"power={self.power}w\n")
        print(f"latency={self.latency}ns\n")
        print(f"f={self.f}\n")

    def write_to_file(self, filename):
        with open(filename, "a") as file:  
            file.write(f"T={self.T}\n")
            file.write(f"area={self.area}um^2\n")
            file.write(f"power={self.power}w\n")
            file.write(f"latency={self.latency}ns\n")
            file.write(f"f={self.f}\n")
            file.write("\n")  
    def write_to_csv(self, filename):
            data = {
                'Temperature (T)': self.T,
                'Area (um^2)': self.area,
                'Power (w)': self.power,
                'Latency (ns)': self.latency,
                'Frequency (f)': self.f
            }
            df = pd.DataFrame([data])
            df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))

sa = Nohetergeneous()
sa.run()
'''
labels = ['area', 'power', 'latency']


data = np.array([
    [1708980399, 787.938786, 15779887.83],   
    [3350629609, 5594.753187, 4935528.424],  
    [5922425679, 785.3507798, 4935528.424],
    [5922425679, 687.2227018, 6066893.115],
    [358554460, 7026.352907, 35157405.08],
    [358554460, 7026.352907, 35156022.68],
    [358554460, 7026.352907, 35156022.68],
    [1280642600,686.285519, 18484769.34]  
])

plt.rcParams.update({'font.size': 14})


angles = np.linspace(1 / 2 * np.pi, 5 / 2 * np.pi, len(labels), endpoint=False).tolist()


max_values = np.max(data, axis=0)
normalized_data = data / max_values


normalized_data = np.concatenate((normalized_data, normalized_data[:, [0]]), axis=1)
angles += angles[:1]  


fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(polar=True))


ax.plot(angles, normalized_data[0], linewidth=1, linestyle='solid', label=f'(16,16) NVM (4,4) (256,256)')
ax.fill(angles, normalized_data[0], alpha=0.25)
ax.plot(angles, normalized_data[1], linewidth=1, linestyle='solid', label=f'(64,64) SRAM (2,2) (256,256)')
ax.fill(angles, normalized_data[1], alpha=0.25)
ax.plot(angles, normalized_data[2], linewidth=1, linestyle='solid', label=f'(64,64) NVM (2,2) (256,256)')
ax.fill(angles, normalized_data[2], alpha=0.25)
ax.plot(angles, normalized_data[3], linewidth=1, linestyle='solid', label=f'(32,32) NVM (2,2) (256,256)')
ax.fill(angles, normalized_data[3], alpha=0.25)
ax.plot(angles, normalized_data[4], linewidth=1, linestyle='solid', label=f'(16,16) SRAM (8,8) (256,256)')
ax.fill(angles, normalized_data[4], alpha=0.25)
ax.plot(angles, normalized_data[5], linewidth=1, linestyle='solid', label=f'(32,32) SRAM (8,8) (256,256)')
ax.fill(angles, normalized_data[5], alpha=0.25)
ax.plot(angles, normalized_data[6], linewidth=1, linestyle='solid', label=f'(64,64) SRAM (8,8) (256,256)')
ax.fill(angles, normalized_data[6], alpha=0.25)
ax.plot(angles, normalized_data[7], linewidth=2, linestyle='solid', label=f'SA_Search(Best)')
ax.fill(angles, normalized_data[7], alpha=0.25)

# 设置雷达图的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 添加图例
ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

# 显示图表
plt.show()
'''