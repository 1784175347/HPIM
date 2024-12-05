import csv
from IPython import embed
import math

def find_min(area=1, power=1, latency=1, data=None):
    area_des = 127500000
    power_des = 20
    latency_des = 5000000
    min_sram = 10000
    min_nvm = 10000
    index_min_sram = 0
    index_min_nvm = 0
    for i in range(len(data)):
        if i < 36:
            f=func(data[i][1],data[i][2],data[i][3],area, power, latency)
            if f>0 and f<min_sram:
                min_sram = f
                index_min_sram = i
        elif i >= 36:
            f=func(data[i][1],data[i][2],data[i][3],area, power, latency)
            if f>0 and f<min_nvm:
                min_nvm = f
                index_min_nvm = i
    print(index_min_sram)
    print(index_min_nvm)

def func(area=1, power=1, latency=1, a=1, b=1, c=1):
    area_des = 150000000
    power_des = 25
    latency_des = 5000000
    if (a == 1):
        area_data = area/area_des
    else:
        area_data = 1
        if (area > area_des):
            return -1
    
    if (b == 1):
        power_data = power/power_des
    else:
        power_data = 1
        if (power > power_des):
            return -1
        
    if (c == 1):
        latency_data = latency/latency_des
    else:
        latency_data = 1
        if (latency > latency_des):
            return -1
    return area_data*latency_data*power_data*10000

file_name = 'Nohetergeneous.csv'

data = []

with open(file_name, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        try:
            record = [float(item) for item in row if item.strip() != '']
            data.append(record)
        except ValueError:
            print(f"无法解析的行：{row}")

find_min(1,0,0,data)
find_min(0,1,0,data)
find_min(0,0,1,data)
find_min(0,1,1,data)
find_min(1,0,1,data)
find_min(1,1,0,data)
find_min(1,1,1,data)