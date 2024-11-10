import csv
from IPython import embed
import math

def similar(area,power,latency,area_des,power_des,latency_des):
    if area == 0 or power == 0 or latency == 0:
        return 100
    if area < area_des:
        area_data = 0
    else:
        area_data = abs(area-area_des)/area_des

    if power < power_des:
        power_data = 0
    else:
        power_data = abs(power-power_des)/power_des

    if latency < latency_des:
        latency_data = 0
    else:
        latency_data = abs(latency-latency_des)/latency_des
    similar_data = area_data + power_data + latency_data
    return similar_data

def find_min_and_indices(similar_data):
    if not similar_data: 
        return None, []  

    min_value = min(similar_data)  
    indices = [index for index, value in enumerate(similar_data) if value == min_value]  
    return min_value, indices

def calculate_alpha(A, Ades):
    if Ades == 0:
        raise ValueError("error")
    numerator = A - Ades
    #alpha = (2 / math.pi) * math.atan(numerator / Ades) + 1
    alpha = math.exp(numerator/Ades)

    return alpha

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


area_des = 300000000
power_des = 20
latency_des = 15000000
similar_data = []

for i in range(len(data)):
    similar_data.append(similar(data[i][1],data[i][2],data[i][3],area_des,power_des,latency_des))

min_value, indices = find_min_and_indices(similar_data)
print(f"最小值: {min_value}")
print(f"最小值的索引: {indices}")

a = calculate_alpha(data[indices[0]][1], area_des)
b = calculate_alpha(data[indices[0]][2], power_des)
c = calculate_alpha(data[indices[0]][3], latency_des)

res = ((data[indices[0]][1])**a) * ((data[indices[0]][2])**b) * ((data[indices[0]][3])**c)/(10**15)
res_des = ((area_des)**a) * ((power_des)**b) * ((latency_des)**c)/(10**15)

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"res = {res}")
print(f"res_des = {res_des}")
power = 70
area = 400000000
latency_min_sram = 1000000000000000000000000
latency_min_nvm = 1000000000000000000000000
index_min_sram = 0
index_min_nvm = 0

for i in range(len(data)):
    if i < 108:
        if data[i][1]<=area  and  data[i][2]<= power and data[i][3]<= latency_min_sram and data[i][1] != 0:
            latency_min_sram = data[i][3]
            index_min_sram = i
    elif i >= 108:
        if data[i][1]<=area and data[i][3]<= latency_min_nvm and data[i][1] != 0:
            latency_min_nvm = data[i][3]
            index_min_nvm = i
print(index_min_sram)
print(index_min_nvm)


latency_min_sram = 1000000000000000000000000
latency_min_nvm = 1000000000000000000000000
index_min_sram = 0
index_min_nvm = 0

for i in range(len(data)):
    if i < 108:
        if data[i][1]<=area and data[i][3]*data[i][2]<= latency_min_sram and data[i][1] != 0:
            latency_min_sram = data[i][3]*data[i][2]
            index_min_sram = i
    elif i >= 108:
        if data[i][1]<=area and data[i][3]*data[i][2]<= latency_min_nvm and data[i][1] != 0:
            latency_min_nvm = data[i][3]*data[i][2]
            index_min_nvm = i
print(index_min_sram)
print(index_min_nvm)

latency_min_sram = 1000000000000000000000000
latency_min_nvm = 1000000000000000000000000
index_min_sram = 0
index_min_nvm = 0

for i in range(len(data)):
    if i < 108:
        if data[i][1]*data[i][3]*data[i][2]<= latency_min_sram and data[i][1] != 0:
            latency_min_sram = data[i][1]*data[i][3]*data[i][2]
            index_min_sram = i
    elif i >= 108:
        if data[i][1]*data[i][3]*data[i][2]<= latency_min_nvm and data[i][1] != 0:
            latency_min_nvm = data[i][1]*data[i][3]*data[i][2]
            index_min_nvm = i
print(index_min_sram)
print(index_min_nvm)
