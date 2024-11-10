import sys
import os
import configparser as cp
work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
import numpy as np
from MNSIM.Interface.interface import *
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
import pandas as pd
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Adder import adder
#linqiushi modified
from MNSIM.Hardware_Model.Multiplier import multiplier
#linqiushi above
home_path = os.getcwd()
SimConfig_path=os.path.join(home_path, "SimConfig.ini")
def make_LUT():
    # 第一维的大小
    first_dimension_size = 2
    # 第二维的大小
    second_dimension_size = 6
    # 第三维的大小
    third_dimension_size = 6
    three_dimensional_array = [[[{'tile_area': i+j+k, 'tile_power_conv': f'value{i}_{j}_{k}','tile_power_fc': f'value{i}_{j}_{k}','tile_power_pooling': 1,'tile_power_element_sum': 1} 
                             for k in range(third_dimension_size)] 
                            for j in range(second_dimension_size)] 
                           for i in range(first_dimension_size)]
    l_device_type=['NVM','SRAM']
    l_xbar_size=[32,64,128,256,512,1024]
    l_PE_num=[1,2,4,8,16,32]
    l_layer_type=['conv','fc','pooling','element_sum']
    count=0
    for i in range(len(l_device_type)):
        for j in range(len(l_xbar_size)):
            for k in range(len(l_PE_num)):
                temp_tile=tile(SimConfig_path,device_type=l_device_type[i],xbar_size=[l_xbar_size[j],l_xbar_size[j]],PE_num=l_PE_num[k],mix_mode=2)
                temp_tile.calculate_tile_area_part(SimConfig_path=SimConfig_path,
                                                ADC_num_mix=int(l_xbar_size[j]/8),DAC_num_mix=int(l_xbar_size[j]/8))
                for m in l_layer_type:
                    temp_tile.calculate_tile_read_power_fast_part(max_column=l_xbar_size[j],max_row=l_xbar_size[j],max_PE=l_PE_num[k]**2,max_group=1,layer_type=m,
                                                        SimConfig_path=SimConfig_path,mix_mode=2,ADC_num_mix=l_xbar_size[j]/8,DAC_num_mix=l_xbar_size[j]/8)
                    three_dimensional_array[i][j][k]['tile_power_'+m]=temp_tile.tile_read_power
                count+=1
                print('这是',i,j,k,count)
                three_dimensional_array[i][j][k]['tile_area']=temp_tile.tile_area
                
    np.save('area_power',three_dimensional_array)
    print("cunl")
    # self.tile_buffer = buffer(SimConfig_path=SimConfig_path,buf_level=2,default_buf_size=default_outbuf_size)
    # self.inbuf = buffer(SimConfig_path=SimConfig_path,buf_level=1,default_buf_size=default_inbuf_size)
    # self.inbuf.calculate_buf_area()
    # self.PE_inbuf_area = self.inbuf.buf_area
    # self.tile_buffer_area += self.tile_buffer.buf_area #一个tile一次
    # self.tile_buffer_area += self.tile_PE_list[i][j].PE_inbuf_area# PE数量次

def load_LUT():
    loaded_array_3d = np.load('area_power.npy',allow_pickle=True)
    print(loaded_array_3d)
    print(loaded_array_3d[0][3][1],loaded_array_3d[1][3][1])
if __name__ == '__main__':
    make_LUT()
    load_LUT()