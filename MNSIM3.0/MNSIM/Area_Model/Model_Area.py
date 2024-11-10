#!/usr/bin/python
# -*-coding:utf-8-*-
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
def use_LUT(device_type,xbar_size,PE_num,op_type):
    loaded_array_3d = np.load('area_power.npy',allow_pickle=True)
    if device_type=='NVM':
        i=0
    elif device_type=='SRAM':
        i=1
    else:
        assert 0
    assert xbar_size>=32
    j=int(math.log2(int(xbar_size/32)))
    assert PE_num>=1
    k=int(math.log2(int(PE_num)))
    if op_type=='area':
        return loaded_array_3d[i][j][k]['tile_area']
    
class Model_area():
    def __init__(self, NetStruct, SimConfig_path, multiple=None, TCG_mapping=None,mix_mode=1,rewrite_mode=1):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        # modelL_config = cp.ConfigParser()
        # modelL_config.read(self.SimConfig_path, encoding='UTF-8')
        # NoC_Compute = int(modelL_config.get('Algorithm Configuration', 'NoC_enable'))
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct,SimConfig_path,multiple)
        self.graph = TCG_mapping
        self.rewrite_mode=self.graph.rewrite_mode
        self.total_layer_num = self.graph.layer_num
        self.arch_area = self.total_layer_num * [0]
        self.arch_xbar_area = self.total_layer_num * [0]
        self.arch_ADC_area = self.total_layer_num * [0]
        self.arch_DAC_area = self.total_layer_num * [0]
        self.arch_digital_area = self.total_layer_num * [0]
        self.arch_adder_area = self.total_layer_num * [0]
        self.arch_shiftreg_area = self.total_layer_num * [0]
        self.arch_iReg_area = self.total_layer_num * [0]
        self.arch_oReg_area = self.total_layer_num * [0]
        self.arch_input_demux_area = self.total_layer_num * [0]
        self.arch_output_mux_area = self.total_layer_num * [0]
        self.arch_jointmodule_area = self.total_layer_num * [0]
        self.arch_buf_area = self.total_layer_num * [0]
        self.arch_pooling_area = self.total_layer_num * [0]
        self.arch_total_area = 0
        self.arch_total_xbar_area = 0
        self.arch_total_ADC_area = 0
        self.arch_total_DAC_area = 0
        self.arch_total_digital_area = 0
        self.arch_total_adder_area = 0
        self.arch_total_shiftreg_area = 0
        self.arch_total_iReg_area = 0
        self.arch_total_oReg_area = 0
        self.arch_total_input_demux_area = 0
        self.arch_total_jointmodule_area = 0
        self.arch_total_buf_area = 0
        self.arch_total_output_mux_area = 0
        self.arch_total_pooling_area = 0
        self.TCG_mapping=TCG_mapping
        # print(data.columns)
        # if NoC_Compute == 1:
        #     path = os.getcwd() + '/Final_Results/'
        #     data = pd.read_csv(path + 'area.csv')
        #     self.arch_Noc_area = float(data.columns[0].split(' ')[-2])
        # else:
        #     self.arch_Noc_area = 0
        self.mix_mode=mix_mode
        if mix_mode==1 or mix_mode==3 or mix_mode==4:
            self.calculate_model_area()
        elif mix_mode==2:
            if TCG_mapping.LUT_use==0:
                self.calculate_model_area_mix()
            elif TCG_mapping.LUT_use==1:
                self.calculate_model_area_LUT()
            else:
                assert 0 
        

    def calculate_model_area(self): #Todo: Noc area
        if self.mix_mode==1:
            self.graph.tile.calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size)
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_area()
            self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
            #linqiushi modified 
            self.global_mul=multiplier(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_multiplier_bitwidth)
            #linqiushi above
            self.global_add.calculate_adder_area()
            for i in range(self.total_layer_num):
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                self.arch_area[i] = self.graph.tile.tile_area * tile_num
                self.arch_xbar_area[i] = self.graph.tile.tile_xbar_area * tile_num
                self.arch_ADC_area[i] = self.graph.tile.tile_ADC_area * tile_num
                self.arch_DAC_area[i] = self.graph.tile.tile_DAC_area * tile_num
                self.arch_digital_area[i] = self.graph.tile.tile_digital_area * tile_num
                self.arch_adder_area[i] = self.graph.tile.tile_adder_area * tile_num
                self.arch_shiftreg_area[i] = self.graph.tile.tile_shiftreg_area * tile_num
                self.arch_iReg_area[i] = self.graph.tile.tile_iReg_area * tile_num
                self.arch_oReg_area[i] = self.graph.tile.tile_oReg_area * tile_num
                self.arch_input_demux_area[i] = self.graph.tile.tile_input_demux_area * tile_num
                self.arch_output_mux_area[i] = self.graph.tile.tile_output_mux_area * tile_num
                self.arch_jointmodule_area[i] = self.graph.tile.tile_jointmodule_area * tile_num
                self.arch_buf_area[i] = self.graph.tile.tile_buffer_area * tile_num
                self.arch_pooling_area[i] = self.graph.tile.tile_pooling_area * tile_num
            self.arch_total_area = sum(self.arch_area)
            self.arch_total_xbar_area = sum(self.arch_xbar_area)
            self.arch_total_ADC_area = sum(self.arch_ADC_area)
            self.arch_total_DAC_area = sum(self.arch_DAC_area)
            self.arch_total_digital_area = sum(self.arch_digital_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_adder_area = sum(self.arch_adder_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
            self.arch_total_iReg_area = sum(self.arch_iReg_area)
            self.arch_total_oReg_area = sum(self.arch_oReg_area)
            self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
            self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
            self.arch_total_jointmodule_area = sum(self.arch_jointmodule_area)
            self.arch_total_buf_area = sum(self.arch_buf_area)+self.global_buf.buf_area
            self.arch_total_pooling_area = sum(self.arch_pooling_area)
            if self.rewrite_mode==2:
                tile_num = self.graph.tile_total_num
                self.arch_area_limited = self.graph.tile.tile_area * tile_num
                self.arch_xbar_area_limited = self.graph.tile.tile_xbar_area * tile_num
                self.arch_ADC_area_limited = self.graph.tile.tile_ADC_area * tile_num
                self.arch_DAC_area_limited = self.graph.tile.tile_DAC_area * tile_num
                self.arch_digital_area_limited = self.graph.tile.tile_digital_area * tile_num+self.global_add.adder_area*self.graph.global_adder_num
                self.arch_adder_area_limited = self.graph.tile.tile_adder_area * tile_num+self.global_add.adder_area*self.graph.global_adder_num
                self.arch_shiftreg_area_limited = self.graph.tile.tile_shiftreg_area * tile_num
                self.arch_iReg_area_limited = self.graph.tile.tile_iReg_area * tile_num
                self.arch_oReg_area_limited = self.graph.tile.tile_oReg_area * tile_num
                self.arch_input_demux_area_limited = self.graph.tile.tile_input_demux_area * tile_num
                self.arch_output_mux_area_limited = self.graph.tile.tile_output_mux_area * tile_num
                self.arch_jointmodule_area_limited = self.graph.tile.tile_jointmodule_area * tile_num
                self.arch_buf_area_limited = self.graph.tile.tile_buffer_area * tile_num+self.global_buf.buf_area
                self.arch_pooling_area_limited = self.graph.tile.tile_pooling_area * tile_num
        elif self.mix_mode==3:
            #create 2 types of tile
            self.graph.tile_RRAM=tile(SimConfig_path=self.SimConfig_path,device_type='NVM',xbar_size=self.graph.xbar_size_NVM)
            self.graph.tile_SRAM=tile(SimConfig_path=self.SimConfig_path,device_type='SRAM',xbar_size=self.graph.xbar_size_SRAM)
            #use tile_RRAM,tile_SRAM to calculate
            self.graph.tile_RRAM.calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size)
            self.graph.tile_SRAM.calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size)
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_area()
            self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
            #linqiushi modified 
            self.global_mul=multiplier(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_multiplier_bitwidth)
            #linqiushi above
            self.global_add.calculate_adder_area()
            
            for i in range(self.total_layer_num):
                #TODO:not conv,fc,pooling:
                print(i,self.total_layer_num)
                RRAM_num=0
                SRAM_num=0
                for le in range(len(self.graph.layer_tileinfo[i]['device_type'])):
                    if self.graph.layer_tileinfo[i]['device_type'][le] =='NVM':
                        RRAM_num+=1
                    elif self.graph.layer_tileinfo[i]['device_type'][le] =='SRAM':
                        SRAM_num+=1
                    else:
                        assert 0,f'type:NVM or SRAM!'
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                self.arch_area[i] = self.graph.tile_RRAM.tile_area * RRAM_num+self.graph.tile_SRAM.tile_area * SRAM_num
                self.arch_xbar_area[i] = self.graph.tile_RRAM.tile_xbar_area * RRAM_num+self.graph.tile_SRAM.tile_xbar_area * SRAM_num
                self.arch_ADC_area[i] = self.graph.tile_RRAM.tile_ADC_area * RRAM_num+self.graph.tile_SRAM.tile_ADC_area * SRAM_num
                self.arch_DAC_area[i] = self.graph.tile_RRAM.tile_DAC_area * RRAM_num+self.graph.tile_SRAM.tile_DAC_area * SRAM_num
                self.arch_digital_area[i] = self.graph.tile_RRAM.tile_digital_area * RRAM_num+self.graph.tile_SRAM.tile_digital_area * SRAM_num
                self.arch_adder_area[i] = self.graph.tile_RRAM.tile_adder_area * RRAM_num+self.graph.tile_SRAM.tile_adder_area * SRAM_num
                self.arch_shiftreg_area[i] = self.graph.tile_RRAM.tile_shiftreg_area * RRAM_num+self.graph.tile_SRAM.tile_shiftreg_area * SRAM_num
                self.arch_iReg_area[i] = self.graph.tile_RRAM.tile_iReg_area * RRAM_num+self.graph.tile_SRAM.tile_iReg_area * SRAM_num
                self.arch_oReg_area[i] = self.graph.tile_RRAM.tile_oReg_area * RRAM_num+self.graph.tile_SRAM.tile_oReg_area * SRAM_num
                self.arch_input_demux_area[i] = self.graph.tile_RRAM.tile_input_demux_area * RRAM_num+self.graph.tile_SRAM.tile_input_demux_area * SRAM_num
                self.arch_output_mux_area[i] = self.graph.tile_RRAM.tile_output_mux_area * RRAM_num+self.graph.tile_SRAM.tile_output_mux_area * SRAM_num
                self.arch_jointmodule_area[i] = self.graph.tile_RRAM.tile_jointmodule_area * RRAM_num+self.graph.tile_SRAM.tile_jointmodule_area * SRAM_num
                self.arch_buf_area[i] = self.graph.tile_RRAM.tile_buffer_area * RRAM_num+self.graph.tile_SRAM.tile_buffer_area * SRAM_num
                self.arch_pooling_area[i] = self.graph.tile_RRAM.tile_pooling_area * RRAM_num+self.graph.tile_SRAM.tile_pooling_area * SRAM_num
            self.arch_total_area = sum(self.arch_area)
            self.arch_total_xbar_area = sum(self.arch_xbar_area)
            self.arch_total_ADC_area = sum(self.arch_ADC_area)
            self.arch_total_DAC_area = sum(self.arch_DAC_area)
            self.arch_total_digital_area = sum(self.arch_digital_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_adder_area = sum(self.arch_adder_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
            self.arch_total_iReg_area = sum(self.arch_iReg_area)
            self.arch_total_oReg_area = sum(self.arch_oReg_area)
            self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
            self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
            self.arch_total_jointmodule_area = sum(self.arch_jointmodule_area)
            self.arch_total_buf_area = sum(self.arch_buf_area)+self.global_buf.buf_area
            self.arch_total_pooling_area = sum(self.arch_pooling_area)
        if self.mix_mode==4:
            
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_area()
            self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
            #linqiushi modified 
            self.global_mul=multiplier(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_multiplier_bitwidth)
            #linqiushi above
            self.global_add.calculate_adder_area()
            for i in range(self.total_layer_num):
                self.graph.tile_layer=tile(SimConfig_path=self.SimConfig_path,device_type=self.graph.layer_tileinfo[i]['device_type'],xbar_size=[self.graph.layer_tileinfo[i]['xbar_size'],\
                    self.graph.layer_tileinfo[i]['xbar_size']],PE_num=self.graph.layer_tileinfo[i]['PE_num_tile'],mix_mode=self.mix_mode)
                self.graph.tile_layer.calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size,
                                                mix_mode=self.mix_mode,ADC_num_mix=math.ceil(self.graph.layer_tileinfo[i]['xbar_size']/8),DAC_num_mix=math.ceil(self.graph.layer_tileinfo[i]['xbar_size']/8))
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                self.arch_area[i] = self.graph.tile_layer.tile_area * tile_num
                self.arch_xbar_area[i] = self.graph.tile_layer.tile_xbar_area * tile_num
                self.arch_ADC_area[i] = self.graph.tile_layer.tile_ADC_area * tile_num
                self.arch_DAC_area[i] = self.graph.tile_layer.tile_DAC_area * tile_num
                self.arch_digital_area[i] = self.graph.tile_layer.tile_digital_area * tile_num
                self.arch_adder_area[i] = self.graph.tile_layer.tile_adder_area * tile_num
                self.arch_shiftreg_area[i] = self.graph.tile_layer.tile_shiftreg_area * tile_num
                self.arch_iReg_area[i] = self.graph.tile_layer.tile_iReg_area * tile_num
                self.arch_oReg_area[i] = self.graph.tile_layer.tile_oReg_area * tile_num
                self.arch_input_demux_area[i] = self.graph.tile_layer.tile_input_demux_area * tile_num
                self.arch_output_mux_area[i] = self.graph.tile_layer.tile_output_mux_area * tile_num
                self.arch_jointmodule_area[i] = self.graph.tile_layer.tile_jointmodule_area * tile_num
                self.arch_buf_area[i] = self.graph.tile_layer.tile_buffer_area * tile_num
                self.arch_pooling_area[i] = self.graph.tile_layer.tile_pooling_area * tile_num
            self.arch_total_area = sum(self.arch_area)
            self.arch_total_xbar_area = sum(self.arch_xbar_area)
            self.arch_total_ADC_area = sum(self.arch_ADC_area)
            self.arch_total_DAC_area = sum(self.arch_DAC_area)
            self.arch_total_digital_area = sum(self.arch_digital_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_adder_area = sum(self.arch_adder_area)+self.global_add.adder_area*self.graph.global_adder_num
            self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
            self.arch_total_iReg_area = sum(self.arch_iReg_area)
            self.arch_total_oReg_area = sum(self.arch_oReg_area)
            self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
            self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
            self.arch_total_jointmodule_area = sum(self.arch_jointmodule_area)
            self.arch_total_buf_area = sum(self.arch_buf_area)+self.global_buf.buf_area
            self.arch_total_pooling_area = sum(self.arch_pooling_area)   
            
    def calculate_model_area_mix(self):
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_area()
        self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
        #linqiushi modified 
        self.global_mul=multiplier(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_multiplier_bitwidth)
        #linqiushi above
        self.global_add.calculate_adder_area()
        tilecount=0
        for layer_id in range(self.total_layer_num):
            temp_device_type=[]
            temp_PE_num=[]
            temp_xbar_size=[]
            temp_ADC_num=[]
            temp_DAC_num=[]
            flag=0
            tilecount=0
            while(tilecount<(self.graph.layer_tileinfo[0]['tile_num_mix'][0])**2):
                i=int(self.TCG_mapping.pos_mapping_order[tilecount][0])
                j=int(self.TCG_mapping.pos_mapping_order[tilecount][1])
                tilecount+=1
                if self.graph.auto_layer_mapping==0:
                    if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                        flag=0
                        pass
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                        flag=1
                        temp_tile_pos=[i,j]
                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                        temp_PE_num.append(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])
                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        temp_ADC_num.append(self.graph.ADC_num_mix[i][j])
                        temp_DAC_num.append(self.graph.DAC_num_mix[i][j])
                        self.graph.tile_list_mix[i][j].calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size,mix_mode=self.mix_mode,
                                                ADC_num_mix=self.graph.ADC_num_mix[i][j],DAC_num_mix=self.graph.DAC_num_mix[i][j])
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])!=layer_id:
                        flag=0
                        pass
                elif self.graph.auto_layer_mapping==1:
                    if self.graph.mapping_result[i][j]==layer_id:
                        temp_tile_pos=[i,j]
                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                        
                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        temp_ADC_num.append(self.graph.ADC_num_mix[i][j])
                        temp_DAC_num.append(self.graph.DAC_num_mix[i][j])
                    
                        self.graph.tile_list_mix[i][j].calculate_tile_area(SimConfig_path=self.SimConfig_path,
                                                default_inbuf_size = self.graph.max_inbuf_size,
                                                default_outbuf_size = self.graph.max_outbuf_size,mix_mode=self.mix_mode,
                                                ADC_num_mix=self.graph.ADC_num_mix[i][j],DAC_num_mix=self.graph.DAC_num_mix[i][j])
                        
                if flag==1:
                    self.arch_area[layer_id]+=self.graph.tile_list_mix[i][j].tile_area
                    self.arch_xbar_area[layer_id] +=self.graph.tile_list_mix[i][j].tile_xbar_area
                    self.arch_ADC_area[layer_id] += self.graph.tile_list_mix[i][j].tile_ADC_area 
                    self.arch_DAC_area[layer_id] += self.graph.tile_list_mix[i][j].tile_DAC_area
                    self.arch_digital_area[layer_id] += self.graph.tile_list_mix[i][j].tile_digital_area 
                    self.arch_adder_area[layer_id] += self.graph.tile_list_mix[i][j].tile_adder_area
                    self.arch_shiftreg_area[layer_id] += self.graph.tile_list_mix[i][j].tile_shiftreg_area 
                    self.arch_iReg_area[layer_id] += self.graph.tile_list_mix[i][j].tile_iReg_area 
                    self.arch_oReg_area[layer_id] += self.graph.tile_list_mix[i][j].tile_oReg_area
                    self.arch_input_demux_area[layer_id] += self.graph.tile_list_mix[i][j].tile_input_demux_area 
                    self.arch_output_mux_area[layer_id] += self.graph.tile_list_mix[i][j].tile_output_mux_area 
                    self.arch_jointmodule_area[layer_id] += self.graph.tile_list_mix[i][j].tile_jointmodule_area 
                    self.arch_buf_area[layer_id] += self.graph.tile_list_mix[i][j].tile_buffer_area 
                    self.arch_pooling_area[layer_id] += self.graph.tile_list_mix[i][j].tile_pooling_area 
        self.arch_total_area = sum(self.arch_area)
        self.arch_total_xbar_area = sum(self.arch_xbar_area)
        self.arch_total_ADC_area = sum(self.arch_ADC_area)
        self.arch_total_DAC_area = sum(self.arch_DAC_area)
        self.arch_total_digital_area = sum(self.arch_digital_area)+self.global_add.adder_area*self.graph.global_adder_num
        self.arch_total_adder_area = sum(self.arch_adder_area)+self.global_add.adder_area*self.graph.global_adder_num
        self.arch_total_shiftreg_area = sum(self.arch_shiftreg_area)
        self.arch_total_iReg_area = sum(self.arch_iReg_area)
        self.arch_total_oReg_area = sum(self.arch_oReg_area)
        self.arch_total_input_demux_area = sum(self.arch_input_demux_area)
        self.arch_total_output_mux_area = sum(self.arch_output_mux_area)
        self.arch_total_jointmodule_area = sum(self.arch_jointmodule_area)
        self.arch_total_buf_area = sum(self.arch_buf_area)+self.global_buf.buf_area
        self.arch_total_pooling_area = sum(self.arch_pooling_area)
    def calculate_model_area_LUT(self):
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_area()
        self.global_add = adder(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_adder_bitwidth)
        #linqiushi modified 
        self.global_mul=multiplier(SimConfig_path=self.SimConfig_path,bitwidth=self.graph.global_multiplier_bitwidth)
        #linqiushi above
        self.global_add.calculate_adder_area()
        tilecount=0
        for layer_id in range(self.total_layer_num):
            temp_device_type=[]
            temp_PE_num=[]
            temp_xbar_size=[]
            temp_ADC_num=[]
            temp_DAC_num=[]
            flag=0
            tilecount=0
            temp_tile_buffer = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.max_outbuf_size)
            temp_tile_buffer.calculate_buf_area()
            PE_inbuf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.max_inbuf_size)
            PE_inbuf.calculate_buf_area()
            while(tilecount<(self.graph.layer_tileinfo[0]['tile_num_mix'][0])**2):
                i=int(self.TCG_mapping.pos_mapping_order[tilecount][0])
                j=int(self.TCG_mapping.pos_mapping_order[tilecount][1])
                tilecount+=1
                if self.graph.auto_layer_mapping==0:
                    if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                        flag=0
                        pass
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                        flag=1
                        temp_tile_pos=[i,j]
                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                        temp_PE_num.append(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])
                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        temp_ADC_num.append(self.graph.ADC_num_mix[i][j])
                        temp_DAC_num.append(self.graph.DAC_num_mix[i][j])
                        tile_area=use_LUT(device_type=self.graph.layer_tileinfo[0]['device_type_mix'][i][j],xbar_size=self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j],PE_num=self.graph.layer_tileinfo[0]['PE_num_mix'][i][j],op_type='area')
                        tile_area += temp_tile_buffer.buf_area
                        tile_area += PE_inbuf.buf_area*self.graph.layer_tileinfo[0]['PE_num_mix'][i][j]*self.graph.layer_tileinfo[0]['PE_num_mix'][i][j]
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])!=layer_id:
                        flag=0
                        pass
                if flag==1:
                    self.arch_area[layer_id]+=tile_area
        self.arch_total_area = sum(self.arch_area)
       
    def model_area_output(self, module_information = 1, layer_information = 1):
        print("Hardware area:", self.arch_total_area, "um^2")

        if module_information:
            if self.rewrite_mode==2:
                print("-----------------------the real(limited) area---------------------------")
                print("     Hardware area:", self.arch_area_limited, "um^2")
                print("		crossbar area:", self.arch_xbar_area_limited, "um^2")
                print("		DAC area:", self.arch_DAC_area_limited, "um^2")
                print("		ADC area:", self.arch_ADC_area_limited, "um^2")
                print("		Buffer area:", self.arch_buf_area_limited, "um^2")
                print("		Pooling area:", self.arch_pooling_area_limited, "um^2")
                print("		Other digital part area:", self.arch_digital_area_limited, "um^2")
                print("			|---adder area:", self.arch_adder_area_limited, "um^2")
                print("			|---output-shift-reg area:", self.arch_shiftreg_area_limited, "um^2")
                print("			|---input-reg area:", self.arch_iReg_area_limited, "um^2")
                print("			|---output-reg area:", self.arch_oReg_area_limited, "um^2")
                print("			|---input_demux area:", self.arch_input_demux_area_limited, "um^2")
                print("			|---output_mux area:", self.arch_output_mux_area_limited, "um^2")
                print("			|---joint_module area:", self.arch_jointmodule_area_limited, "um^2")
                print("----------------------the expected(but couldn't be true) area---------------")  
                
            print("		crossbar area:", self.arch_total_xbar_area, "um^2")
            print("		DAC area:", self.arch_total_DAC_area, "um^2")
            print("		ADC area:", self.arch_total_ADC_area, "um^2")
            print("		Buffer area:", self.arch_total_buf_area, "um^2")
            print("		Pooling area:", self.arch_total_pooling_area, "um^2")
            print("		Other digital part area:", self.arch_total_digital_area, "um^2")
            print("			|---adder area:", self.arch_total_adder_area, "um^2")
            print("			|---output-shift-reg area:", self.arch_total_shiftreg_area, "um^2")
            print("			|---input-reg area:", self.arch_total_iReg_area, "um^2")
            print("			|---output-reg area:", self.arch_total_oReg_area, "um^2")
            print("			|---input_demux area:", self.arch_total_input_demux_area, "um^2")
            print("			|---output_mux area:", self.arch_total_output_mux_area, "um^2")
            print("			|---joint_module area:", self.arch_total_jointmodule_area, "um^2")
            # print("		NoC part area:", self.arch_Noc_area, "um^2")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                layer_dict = self.NetStruct[i][0][0]
                #linqiushi modified
                if layer_dict['type'] == 'element_sum':
                    print("     Hardware area (global accumulator):", self.global_add.adder_area*self.graph.global_adder_num+self.global_buf.buf_area, "um^2")
                elif layer_dict['type']=='element_multiply':
                    print("     Hardware area (global accumulator):", self.global_mul.multiplier_area*self.graph.global_multiplier_num+self.global_buf.buf_area, "um^2")
                else:
                    print("     Hardware area:", self.arch_area[i], "um^2")
                #linqiushi above
    #linqiushi modified
    def area_output_CNNParted(self):
        area_list=[]
        
        for i in range(self.total_layer_num):
            layer_dict = self.NetStruct[i][0][0]
            
            if layer_dict['type'] == 'element_sum':
                area_list.append(self.global_add.adder_area*self.graph.global_adder_num+self.global_buf.buf_area)
            elif layer_dict['type']=='element_multiply':
                area_list.append(self.global_mul.multiplier_area*self.graph.global_multiplier_num+self.global_buf.buf_area)
            else:
                area_list.append(self.arch_area[i])
        return area_list
    #linqiushi above
if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path)
    structure_file = __TestInterface.get_structure()
    __TCG_mapping = TCG(structure_file, test_SimConfig_path)
    __area = Model_area(NetStruct=structure_file,SimConfig_path=test_SimConfig_path,TCG_mapping=__TCG_mapping)
    __area.model_area_output(1,1)