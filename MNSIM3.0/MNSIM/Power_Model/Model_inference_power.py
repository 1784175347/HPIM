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
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.Pooling import Pooling
def create_nested_zeros(original_list):
    if isinstance(original_list, list):
        return [create_nested_zeros(item) for item in original_list]
    else:
        return 0
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
    elif op_type=='conv':
        return loaded_array_3d[i][j][k]['tile_power_conv']
    elif op_type=='fc':
        return loaded_array_3d[i][j][k]['tile_power_fc']
    elif op_type=='pooling':
        return loaded_array_3d[i][j][k]['tile_power_pooling']
    elif op_type=='element_sum':
        return loaded_array_3d[i][j][k]['tile_power_element_sum']
    else:
        assert 0
    
class Model_inference_power():
    def __init__(self, NetStruct, SimConfig_path, multiple=None, TCG_mapping=None,mix_mode=1):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct,SimConfig_path,multiple)
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num
        self.arch_power = self.total_layer_num * [0]
        self.arch_xbar_power = self.total_layer_num * [0]
        self.arch_ADC_power = self.total_layer_num * [0]
        self.arch_DAC_power = self.total_layer_num * [0]
        self.arch_digital_power = self.total_layer_num * [0]
        self.arch_adder_power = self.total_layer_num * [0]
        self.arch_shiftreg_power = self.total_layer_num * [0]
        self.arch_iReg_power = self.total_layer_num * [0]
        self.arch_oReg_power = self.total_layer_num * [0]
        self.arch_input_demux_power = self.total_layer_num * [0]
        self.arch_output_mux_power = self.total_layer_num * [0]
        self.arch_jointmodule_power = self.total_layer_num * [0]
        self.arch_buf_power = self.total_layer_num * [0]
        self.arch_buf_r_power = self.total_layer_num * [0]
        self.arch_buf_w_power = self.total_layer_num * [0]
        self.arch_pooling_power = self.total_layer_num * [0]
        self.arch_write_power=self.total_layer_num *[0]
        self.arch_total_power = 0
        self.arch_total_xbar_power = 0
        self.arch_total_ADC_power = 0
        self.arch_total_DAC_power = 0
        self.arch_total_digital_power = 0
        self.arch_total_adder_power = 0
        self.arch_total_shiftreg_power = 0
        self.arch_total_iReg_power = 0
        self.arch_total_oReg_power = 0
        self.arch_total_input_demux_power = 0
        self.arch_total_jointmodule_power = 0
        self.arch_total_buf_power = 0
        self.arch_total_buf_r_power = 0
        self.arch_total_buf_w_power = 0
        self.arch_total_output_mux_power = 0
        self.arch_total_pooling_power = 0
        self.arch_total_write_power =0
        self.TCG_mapping=TCG_mapping
        self.mix_mode=mix_mode
        if mix_mode==1 or mix_mode==3 or mix_mode==4:
            if TCG_mapping.rewrite_mode==0:
                self.calculate_model_power()
            else:
                self.calculate_model_power_rewrite()
            
        elif mix_mode==2:
            if TCG_mapping.LUT_use==1:
                self.calculate_model_power_LUT()
            elif TCG_mapping.LUT_use==0:
                self.calculate_model_power_mix()
            else:
                assert 0
        else:
            assert 0

    def calculate_model_power(self):
        if self.mix_mode==1:
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                    default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_read_power()
            self.global_buf.calculate_buf_write_power()
            self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
            self.global_add.calculate_adder_power()
            for i in range(self.total_layer_num):
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                max_column = self.graph.layer_tileinfo[i]['max_column']
                max_row = self.graph.layer_tileinfo[i]['max_row']
                max_PE = self.graph.layer_tileinfo[i]['max_PE']
                max_group = self.graph.layer_tileinfo[i]['max_group']
                layer_type = self.graph.net[i][0][0]['type']
                #linqiushi modified
                rewrite_mode=self.graph.rewrite_mode
                whether_rewrite=0
                if rewrite_mode==1:
                    whether_rewrite=self.graph.layer_whether_rewrite[i]
                #linqiushi modified
                #add mix mode
                #mix mode 3: inside tile
                #mix mode 2: among tile
            # if self.mix_mode==3:
                    #print("mode=3")
                    
                if rewrite_mode==1 and whether_rewrite==1:
                    print("这个层需要rewrite-power")
                #elif self.mix_mode==1 or self.mix_mode==2:
                self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size)
                self.graph.tile.calculate_tile_write_power_fast(max_column=max_column,max_row=max_row,max_group=max_group,max_PE=max_PE)
                #else:
                    #assert 0, f'mix_mode must in {1,2,3}'
                #linqiushi above
                #liinqiushi modified 
                #test:whether the cir will slow the program damagely-->no!
                '''for j in range(100*tile_num):
                    self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                                SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size)'''
                #linqiushi above
                #linqiushi modified
                #self.arch_power[i] = self.graph.tile.tile_read_power * tile_num
                #self.arch_xbar_power[i] = self.graph.tile.tile_xbar_read_power * tile_num
                #linqiushi above
                self.arch_write_power[i]=self.graph.tile.tile_write_power * tile_num
                self.arch_power[i] = (self.graph.tile.tile_read_power+self.graph.tile.tile_write_power) * tile_num
                self.arch_xbar_power[i] = (self.graph.tile.tile_xbar_read_power+self.graph.tile.tile_xbar_write_power) * tile_num
                self.arch_ADC_power[i] = self.graph.tile.tile_ADC_read_power * tile_num
                self.arch_DAC_power[i] = self.graph.tile.tile_DAC_read_power * tile_num
                self.arch_digital_power[i] = self.graph.tile.tile_digital_read_power * tile_num
                self.arch_adder_power[i] = self.graph.tile.tile_adder_read_power * tile_num
                self.arch_shiftreg_power[i] = self.graph.tile.tile_shiftreg_read_power * tile_num
                self.arch_iReg_power[i] = self.graph.tile.tile_iReg_read_power * tile_num
                self.arch_oReg_power[i] = self.graph.tile.tile_oReg_read_power * tile_num
                self.arch_input_demux_power[i] = self.graph.tile.tile_input_demux_read_power * tile_num
                self.arch_output_mux_power[i] = self.graph.tile.tile_output_mux_read_power * tile_num
                self.arch_jointmodule_power[i] = self.graph.tile.tile_jointmodule_read_power * tile_num
                self.arch_buf_power[i] = self.graph.tile.tile_buffer_read_power * tile_num
                self.arch_buf_r_power[i] = self.graph.tile.tile_buffer_r_read_power * tile_num
                self.arch_buf_w_power[i] = self.graph.tile.tile_buffer_w_read_power * tile_num
                self.arch_pooling_power[i] = self.graph.tile.tile_pooling_read_power * tile_num
            self.arch_total_write_power=sum(self.arch_write_power)
            self.arch_total_power = sum(self.arch_power)
            self.arch_total_xbar_power = sum(self.arch_xbar_power)
            self.arch_total_ADC_power = sum(self.arch_ADC_power)
            self.arch_total_DAC_power = sum(self.arch_DAC_power)
            self.arch_total_digital_power = sum(self.arch_digital_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_adder_power = sum(self.arch_adder_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
            self.arch_total_iReg_power = sum(self.arch_iReg_power)
            self.arch_total_oReg_power = sum(self.arch_oReg_power)
            self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
            self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
            self.arch_total_jointmodule_power = sum(self.arch_jointmodule_power)
            self.arch_total_buf_power = sum(self.arch_buf_power)+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3
            self.arch_total_buf_r_power = sum(self.arch_buf_r_power)+self.global_buf.buf_rpower*1e-3
            self.arch_total_buf_w_power = sum(self.arch_buf_w_power)+self.global_buf.buf_wpower*1e-3
            self.arch_total_pooling_power = sum(self.arch_pooling_power)
        elif self.mix_mode==3:
            #TODO：这里的问题是，power的计算和网络层有关，导致每一个tile的power不一样，如果详尽的话需要计算较长时间
            self.graph.tile_RRAM=tile(SimConfig_path=self.SimConfig_path,device_type='NVM',xbar_size=self.graph.xbar_size_NVM)
            self.graph.tile_SRAM=tile(SimConfig_path=self.SimConfig_path,device_type='SRAM',xbar_size=self.graph.xbar_size_SRAM)
            #use tile_RRAM,tile_SRAM to calculate
            self.graph.tile_RRAM.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size)
            self.graph.tile_SRAM.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size)
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                    default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_read_power()
            self.global_buf.calculate_buf_write_power()
            self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
            self.global_add.calculate_adder_power()
            for le in range(len(self.graph.layer_tileinfo[i]['device_type'])):
                if self.graph.layer_tileinfo[i]['device_type'][le] =='NVM':
                    RRAM_num+=1
                elif self.graph.layer_tileinfo[i]['device_type'][le] =='SRAM':
                    SRAM_num+=1
                else:
                    assert 0,f'type:NVM or SRAM!'
            for i in range(self.total_layer_num):
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                max_column = self.graph.layer_tileinfo[i]['max_column']
                max_row = self.graph.layer_tileinfo[i]['max_row']
                max_PE = self.graph.layer_tileinfo[i]['max_PE']
                max_group = self.graph.layer_tileinfo[i]['max_group']
                layer_type = self.graph.net[i][0][0]['type']
                #linqiushi modified
                rewrite_mode=self.graph.rewrite_mode
                whether_rewrite=0
                if rewrite_mode==1:
                    whether_rewrite=self.graph.layer_whether_rewrite[i]
                #linqiushi modified
                #add mix mode
                #mix mode 3: inside tile
                #mix mode 2: among tile
            # if self.mix_mode==3:
                    #print("mode=3")
                    
                if rewrite_mode==1 and whether_rewrite==1:
                    print("这个层需要rewrite-power")
                #elif self.mix_mode==1 or self.mix_mode==2:
                self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size)
                print("infer中的",max_row)
                self.graph.tile.calculate_tile_write_power_fast(max_column=max_column,max_row=max_row,max_group=max_group,max_PE=max_PE)
                #else:
                    #assert 0, f'mix_mode must in {1,2,3}'
                #linqiushi above
                #liinqiushi modified 
                #test:whether the cir will slow the program damagely-->no!
                '''for j in range(100*tile_num):
                    self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                                SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size)'''
                #linqiushi above
                #linqiushi modified
                #self.arch_power[i] = self.graph.tile.tile_read_power * tile_num
                #self.arch_xbar_power[i] = self.graph.tile.tile_xbar_read_power * tile_num
                #linqiushi above
                self.arch_write_power[i]=self.graph.tile.tile_write_power * tile_num
                self.arch_power[i] = (self.graph.tile.tile_read_power+self.graph.tile.tile_write_power) * tile_num
                self.arch_xbar_power[i] = (self.graph.tile.tile_xbar_read_power+self.graph.tile.tile_xbar_write_power) * tile_num
                self.arch_ADC_power[i] = self.graph.tile.tile_ADC_read_power * tile_num
                self.arch_DAC_power[i] = self.graph.tile.tile_DAC_read_power * tile_num
                self.arch_digital_power[i] = self.graph.tile.tile_digital_read_power * tile_num
                self.arch_adder_power[i] = self.graph.tile.tile_adder_read_power * tile_num
                self.arch_shiftreg_power[i] = self.graph.tile.tile_shiftreg_read_power * tile_num
                self.arch_iReg_power[i] = self.graph.tile.tile_iReg_read_power * tile_num
                self.arch_oReg_power[i] = self.graph.tile.tile_oReg_read_power * tile_num
                self.arch_input_demux_power[i] = self.graph.tile.tile_input_demux_read_power * tile_num
                self.arch_output_mux_power[i] = self.graph.tile.tile_output_mux_read_power * tile_num
                self.arch_jointmodule_power[i] = self.graph.tile.tile_jointmodule_read_power * tile_num
                self.arch_buf_power[i] = self.graph.tile.tile_buffer_read_power * tile_num
                self.arch_buf_r_power[i] = self.graph.tile.tile_buffer_r_read_power * tile_num
                self.arch_buf_w_power[i] = self.graph.tile.tile_buffer_w_read_power * tile_num
                self.arch_pooling_power[i] = self.graph.tile.tile_pooling_read_power * tile_num
            self.arch_total_write_power=sum(self.arch_write_power)
            self.arch_total_power = sum(self.arch_power)
            self.arch_total_xbar_power = sum(self.arch_xbar_power)
            self.arch_total_ADC_power = sum(self.arch_ADC_power)
            self.arch_total_DAC_power = sum(self.arch_DAC_power)
            self.arch_total_digital_power = sum(self.arch_digital_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_adder_power = sum(self.arch_adder_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
            self.arch_total_iReg_power = sum(self.arch_iReg_power)
            self.arch_total_oReg_power = sum(self.arch_oReg_power)
            self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
            self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
            self.arch_total_jointmodule_power = sum(self.arch_jointmodule_power)
            self.arch_total_buf_power = sum(self.arch_buf_power)+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3
            self.arch_total_buf_r_power = sum(self.arch_buf_r_power)+self.global_buf.buf_rpower*1e-3
            self.arch_total_buf_w_power = sum(self.arch_buf_w_power)+self.global_buf.buf_wpower*1e-3
            self.arch_total_pooling_power = sum(self.arch_pooling_power)
        elif self.mix_mode==4:
            self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                    default_buf_size=self.graph.global_buf_size)
            self.global_buf.calculate_buf_read_power()
            self.global_buf.calculate_buf_write_power()
            self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
            self.global_add.calculate_adder_power()
            for i in range(self.total_layer_num):
                self.graph.tile_layer=tile(SimConfig_path=self.SimConfig_path,device_type=self.graph.layer_tileinfo[i]['device_type'],xbar_size=[self.graph.layer_tileinfo[i]['xbar_size'],\
                    self.graph.layer_tileinfo[i]['xbar_size']],PE_num=self.graph.layer_tileinfo[i]['PE_num_tile'],mix_mode=self.mix_mode)
                tile_num = self.graph.layer_tileinfo[i]['tilenum']
                max_column = self.graph.layer_tileinfo[i]['max_column']
                max_row = self.graph.layer_tileinfo[i]['max_row']
                max_PE = self.graph.layer_tileinfo[i]['max_PE']
                max_group = self.graph.layer_tileinfo[i]['max_group']
                layer_type = self.graph.net[i][0][0]['type']
                size=self.graph.layer_tileinfo[i]['xbar_size']
                #linqiushi modified
                rewrite_mode=self.graph.rewrite_mode
                whether_rewrite=0
                if rewrite_mode==1:
                    whether_rewrite=self.graph.layer_whether_rewrite[i]
                #linqiushi modified
                #add mix mode
                #mix mode 3: inside tile
                #mix mode 2: among tile
            # if self.mix_mode==3:
                    #print("mode=3")
                    
                if rewrite_mode==1 and whether_rewrite==1:
                    print("这个层需要rewrite-power")
                #elif self.mix_mode==1 or self.mix_mode==2:
                self.graph.tile_layer.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size,ADC_num_mix=math.ceil(size/8),DAC_num_mix=math.ceil(size/8))
                print("infer中的",max_row)
                #self.graph.tile.calculate_tile_write_power_fast(max_column=max_column,max_row=max_row,max_group=max_group,max_PE=max_PE)
                #else:
                    #assert 0, f'mix_mode must in {1,2,3}'
                #linqiushi above
                #liinqiushi modified 
                #test:whether the cir will slow the program damagely-->no!
                '''for j in range(100*tile_num):
                    self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                                SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size)'''
                #linqiushi above
                #linqiushi modified
                #self.arch_power[i] = self.graph.tile.tile_read_power * tile_num
                #self.arch_xbar_power[i] = self.graph.tile.tile_xbar_read_power * tile_num
                #linqiushi above
                #self.arch_write_power[i]=self.graph.tile_layer.tile_write_power * tile_num
                self.arch_power[i] = (self.graph.tile_layer.tile_read_power+self.graph.tile.tile_write_power) * tile_num
                self.arch_xbar_power[i] = (self.graph.tile_layer.tile_xbar_read_power+self.graph.tile.tile_xbar_write_power) * tile_num
                self.arch_ADC_power[i] = self.graph.tile_layer.tile_ADC_read_power * tile_num
                self.arch_DAC_power[i] = self.graph.tile_layer.tile_DAC_read_power * tile_num
                self.arch_digital_power[i] = self.graph.tile_layer.tile_digital_read_power * tile_num
                self.arch_adder_power[i] = self.graph.tile_layer.tile_adder_read_power * tile_num
                self.arch_shiftreg_power[i] = self.graph.tile_layer.tile_shiftreg_read_power * tile_num
                self.arch_iReg_power[i] = self.graph.tile_layer.tile_iReg_read_power * tile_num
                self.arch_oReg_power[i] = self.graph.tile_layer.tile_oReg_read_power * tile_num
                self.arch_input_demux_power[i] = self.graph.tile_layer.tile_input_demux_read_power * tile_num
                self.arch_output_mux_power[i] = self.graph.tile_layer.tile_output_mux_read_power * tile_num
                self.arch_jointmodule_power[i] = self.graph.tile_layer.tile_jointmodule_read_power * tile_num
                self.arch_buf_power[i] = self.graph.tile_layer.tile_buffer_read_power * tile_num
                self.arch_buf_r_power[i] = self.graph.tile_layer.tile_buffer_r_read_power * tile_num
                self.arch_buf_w_power[i] = self.graph.tile_layer.tile_buffer_w_read_power * tile_num
                self.arch_pooling_power[i] = self.graph.tile_layer.tile_pooling_read_power * tile_num
            #self.arch_total_write_power=sum(self.arch_write_power)
            self.arch_total_power = sum(self.arch_power)
            self.arch_total_xbar_power = sum(self.arch_xbar_power)
            self.arch_total_ADC_power = sum(self.arch_ADC_power)
            self.arch_total_DAC_power = sum(self.arch_DAC_power)
            self.arch_total_digital_power = sum(self.arch_digital_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_adder_power = sum(self.arch_adder_power)+self.global_add.adder_power*self.graph.global_adder_num
            self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
            self.arch_total_iReg_power = sum(self.arch_iReg_power)
            self.arch_total_oReg_power = sum(self.arch_oReg_power)
            self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
            self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
            self.arch_total_jointmodule_power = sum(self.arch_jointmodule_power)
            self.arch_total_buf_power = sum(self.arch_buf_power)+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3
            self.arch_total_buf_r_power = sum(self.arch_buf_r_power)+self.global_buf.buf_rpower*1e-3
            self.arch_total_buf_w_power = sum(self.arch_buf_w_power)+self.global_buf.buf_wpower*1e-3
            self.arch_total_pooling_power = sum(self.arch_pooling_power)
    def calculate_model_power_rewrite(self):
        assert self.mix_mode==1
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_read_power()
        self.global_buf.calculate_buf_write_power()
        self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
        self.global_add.calculate_adder_power()
        #self.graph.rewrite_tile_num_layer
        
        self.arch_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_write_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_xbar_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_ADC_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_DAC_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_digital_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_adder_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_shiftreg_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_iReg_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_oReg_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_input_demux_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_output_mux_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_jointmodule_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_buf_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_buf_r_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_buf_w_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_pooling_power=create_nested_zeros(self.graph.rewrite_layer_list)
        self.arch_total_power = []
        self.arch_total_xbar_power = []
        self.arch_total_ADC_power = []
        self.arch_total_DAC_power = []
        self.arch_total_digital_power = []
        self.arch_total_adder_power = []
        self.arch_total_shiftreg_power = []
        self.arch_total_iReg_power = []
        self.arch_total_oReg_power = []
        self.arch_total_input_demux_power = []
        self.arch_total_jointmodule_power = []
        self.arch_total_buf_power = []
        self.arch_total_buf_r_power = []
        self.arch_total_buf_w_power =[]
        self.arch_total_output_mux_power = []
        self.arch_total_pooling_power = []
        self.arch_total_write_power =[]
        for j in range(len(self.graph.rewrite_layer_list)):
            for i in range(len(self.graph.rewrite_layer_list[j])):
                layer_id=self.graph.rewrite_layer_list[j][i]
                tile_num = self.graph.rewrite_tile_num_layer[j][i]
                max_column = self.graph.layer_tileinfo[layer_id]['max_column']
                max_row = self.graph.layer_tileinfo[layer_id]['max_row']
                max_PE = self.graph.layer_tileinfo[layer_id]['max_PE']
                max_group = self.graph.layer_tileinfo[layer_id]['max_group']
                layer_type = self.graph.net[layer_id][0][0]['type']
                #linqiushi modified
                rewrite_mode=self.graph.rewrite_mode
                assert rewrite_mode==1 or rewrite_mode==2
                
                self.graph.tile.calculate_tile_read_power_fast(max_column=max_column,max_row=max_row,max_PE=max_PE,max_group=max_group,layer_type=layer_type,
                                                            SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size)
                self.graph.tile.calculate_tile_write_power_fast(max_column=max_column,max_row=max_row,max_group=max_group,max_PE=max_PE)
                if layer_id==self.graph.rewrite_layer_list[j][0]:
                    self.arch_write_power[j][i]=self.graph.tile.tile_write_power * self.graph.tile_total_num
                else:
                    self.arch_write_power[j][i]=0
                self.arch_power[j][i] = (self.graph.tile.tile_read_power+self.graph.tile.tile_write_power) * tile_num
                self.arch_xbar_power[j][i] = (self.graph.tile.tile_xbar_read_power+self.graph.tile.tile_xbar_write_power) * tile_num
                self.arch_ADC_power[j][i] = self.graph.tile.tile_ADC_read_power * tile_num
                self.arch_DAC_power[j][i] = self.graph.tile.tile_DAC_read_power * tile_num
                self.arch_digital_power[j][i] = self.graph.tile.tile_digital_read_power * tile_num
                self.arch_adder_power[j][i] = self.graph.tile.tile_adder_read_power * tile_num
                self.arch_shiftreg_power[j][i] = self.graph.tile.tile_shiftreg_read_power * tile_num
                self.arch_iReg_power[j][i] = self.graph.tile.tile_iReg_read_power * tile_num
                self.arch_oReg_power[j][i] = self.graph.tile.tile_oReg_read_power * tile_num
                self.arch_input_demux_power[j][i] = self.graph.tile.tile_input_demux_read_power * tile_num
                self.arch_output_mux_power[j][i] = self.graph.tile.tile_output_mux_read_power * tile_num
                self.arch_jointmodule_power[j][i] = self.graph.tile.tile_jointmodule_read_power * tile_num
                self.arch_buf_power[j][i] = self.graph.tile.tile_buffer_read_power * tile_num
                self.arch_buf_r_power[j][i] = self.graph.tile.tile_buffer_r_read_power * tile_num
                self.arch_buf_w_power[j][i] = self.graph.tile.tile_buffer_w_read_power * tile_num
                self.arch_pooling_power[j][i] = self.graph.tile.tile_pooling_read_power * tile_num
            
            self.arch_total_write_power.append(sum(self.arch_write_power[j]))
            self.arch_total_power.append(sum(self.arch_power[j]))
            self.arch_total_xbar_power.append(sum(self.arch_xbar_power[j]))
            self.arch_total_ADC_power.append(sum(self.arch_ADC_power[j]))
            self.arch_total_DAC_power.append(sum(self.arch_DAC_power[j]))
            self.arch_total_digital_power.append(sum(self.arch_digital_power[j])+self.global_add.adder_power*self.graph.global_adder_num)
            self.arch_total_adder_power.append(sum(self.arch_adder_power[j])+self.global_add.adder_power*self.graph.global_adder_num)
            self.arch_total_shiftreg_power.append(sum(self.arch_shiftreg_power[j]))
            self.arch_total_iReg_power.append(sum(self.arch_iReg_power[j]))
            self.arch_total_oReg_power.append(sum(self.arch_oReg_power[j]))
            self.arch_total_input_demux_power.append(sum(self.arch_input_demux_power[j]))
            self.arch_total_output_mux_power.append(sum(self.arch_output_mux_power[j]))
            self.arch_total_jointmodule_power.append(sum(self.arch_jointmodule_power[j]))
            self.arch_total_buf_power.append(sum(self.arch_buf_power[j])+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3)
            self.arch_total_buf_r_power.append(sum(self.arch_buf_r_power[j])+self.global_buf.buf_rpower*1e-3)
            self.arch_total_buf_w_power.append(sum(self.arch_buf_w_power[j])+self.global_buf.buf_wpower*1e-3)
            self.arch_total_pooling_power.append(sum(self.arch_pooling_power[j]))
    def calculate_model_power_mix(self):
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                 default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_read_power()
        self.global_buf.calculate_buf_write_power()
        self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
        self.global_add.calculate_adder_power()
        
        for i in range(self.total_layer_num):
            tile_num = self.graph.layer_tileinfo[i]['tilenum']
            max_column = self.graph.layer_tileinfo[i]['max_column']
            max_row = self.graph.layer_tileinfo[i]['max_row']
            max_PE = self.graph.layer_tileinfo[i]['max_PE']
            max_group = self.graph.layer_tileinfo[i]['max_group']
            layer_type = self.graph.net[i][0][0]['type']
            #linqiushi modified
            rewrite_mode=self.graph.rewrite_mode
            whether_rewrite=0
            if rewrite_mode==1:
                whether_rewrite=self.graph.layer_whether_rewrite[i]
            #linqiushi modified
            #add mix mode
            #mix mode 3: inside tile
            #mix mode 2: among tile
           # if self.mix_mode==3:
                #print("mode=3")
                
            if rewrite_mode==1 and whether_rewrite==1:
                print("这个层需要rewrite-power")
            #elif self.mix_mode==1 or self.mix_mode==2:
            
        tilecount=0  
        for layer_id in range(self.total_layer_num):
            temp_device_type=[]
            temp_PE_num=[]
            temp_xbar_size=[]
            temp_ADC_num=[]
            temp_DAC_num=[]
            k=0
            flag=0
            tilecount=0
            while(tilecount<(self.graph.layer_tileinfo[0]['tile_num_mix'][0])**2):
                i=int(self.TCG_mapping.pos_mapping_order[tilecount][0])
                j=int(self.TCG_mapping.pos_mapping_order[tilecount][1])
                tilecount+=1
                if self.graph.auto_layer_mapping==0:
                    if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                        flag=0
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                        flag=1
                        temp_tile_pos=[i,j]
                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                        temp_PE_num.append(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])
                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        self.graph.tile_list_mix[i][j].calculate_tile_read_power_fast(max_column=self.graph.layer_tileinfo[layer_id]['tile_max_column'][k],
                                                                                    max_row=self.graph.layer_tileinfo[layer_id]['tile_max_row'][k],
                                                                                    max_PE=self.graph.layer_tileinfo[layer_id]['max_PE'][i][j],
                                                                                    max_group=self.graph.layer_tileinfo[layer_id]['max_group'],
                                                                                    layer_type=self.graph.net[layer_id][0][0]['type'],
                                                    SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                    default_outbuf_size=self.graph.max_outbuf_size,
                                                    mix_mode=self.mix_mode,ADC_num_mix=self.graph.ADC_num_mix[i][j],DAC_num_mix=self.graph.DAC_num_mix[i][j])
                        k=k+1
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])!=layer_id:
                        flag=0
                elif self.graph.auto_layer_mapping==1:
                    
                    if self.graph.mapping_result[i][j]==layer_id:
                        print(i,j,layer_id)
                        temp_tile_pos=[i,j]
                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                        temp_PE_num.append(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])
                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        self.graph.tile_list_mix[i][j].calculate_tile_read_power_fast(max_column=self.graph.layer_tileinfo[layer_id]['tile_max_column'][k],
                                                                                    max_row=self.graph.layer_tileinfo[layer_id]['tile_max_row'][k],
                                                                                    max_PE=self.graph.layer_tileinfo[layer_id]['max_PE'][i][j],
                                                                                    max_group=self.graph.layer_tileinfo[layer_id]['max_group'],
                                                                                    layer_type=self.graph.net[i][0][0]['type'],
                                                    SimConfig_path=self.SimConfig_path,default_inbuf_size=self.graph.max_inbuf_size,
                                                    default_outbuf_size=self.graph.max_outbuf_size,
                                                    mix_mode=self.mix_mode,ADC_num_mix=self.graph.ADC_num_mix[i][j],DAC_num_mix=self.graph.DAC_num_mix[i][j])
                        k+=1
                else:
                    assert 0, f'auto_layer_mapping should be 0 or 1'
                    
                if flag==1:
                    self.arch_write_power[layer_id]+=self.graph.tile_list_mix[i][j].tile_write_power 
                    self.arch_power[layer_id] += (self.graph.tile_list_mix[i][j].tile_read_power) 
                    self.arch_xbar_power[layer_id] += (self.graph.tile_list_mix[i][j].tile_xbar_read_power) 
                    self.arch_ADC_power[layer_id] += self.graph.tile_list_mix[i][j].tile_ADC_read_power 
                    self.arch_DAC_power[layer_id] += self.graph.tile_list_mix[i][j].tile_DAC_read_power 
                    self.arch_digital_power[layer_id] += self.graph.tile_list_mix[i][j].tile_digital_read_power 
                    self.arch_adder_power[layer_id] += self.graph.tile_list_mix[i][j].tile_adder_read_power 
                    self.arch_shiftreg_power[layer_id] += self.graph.tile_list_mix[i][j].tile_shiftreg_read_power
                    self.arch_iReg_power[layer_id] += self.graph.tile_list_mix[i][j].tile_iReg_read_power
                    self.arch_oReg_power[layer_id] += self.graph.tile_list_mix[i][j].tile_oReg_read_power 
                    self.arch_input_demux_power[layer_id] += self.graph.tile_list_mix[i][j].tile_input_demux_read_power 
                    self.arch_output_mux_power[layer_id] += self.graph.tile_list_mix[i][j].tile_output_mux_read_power 
                    self.arch_jointmodule_power[layer_id] += self.graph.tile_list_mix[i][j].tile_jointmodule_read_power 
                    self.arch_buf_power[layer_id] += self.graph.tile_list_mix[i][j].tile_buffer_read_power 
                    self.arch_buf_r_power[layer_id] += self.graph.tile_list_mix[i][j].tile_buffer_r_read_power
                    self.arch_buf_w_power[layer_id] += self.graph.tile_list_mix[i][j].tile_buffer_w_read_power 
                    self.arch_pooling_power[layer_id] += self.graph.tile_list_mix[i][j].tile_pooling_read_power 
        self.arch_total_write_power=sum(self.arch_write_power)
        self.arch_total_power = sum(self.arch_power)
        self.arch_total_xbar_power = sum(self.arch_xbar_power)
        self.arch_total_ADC_power = sum(self.arch_ADC_power)
        self.arch_total_DAC_power = sum(self.arch_DAC_power)
        self.arch_total_digital_power = sum(self.arch_digital_power)+self.global_add.adder_power*self.graph.global_adder_num
        self.arch_total_adder_power = sum(self.arch_adder_power)+self.global_add.adder_power*self.graph.global_adder_num
        self.arch_total_shiftreg_power = sum(self.arch_shiftreg_power)
        self.arch_total_iReg_power = sum(self.arch_iReg_power)
        self.arch_total_oReg_power = sum(self.arch_oReg_power)
        self.arch_total_input_demux_power = sum(self.arch_input_demux_power)
        self.arch_total_output_mux_power = sum(self.arch_output_mux_power)
        self.arch_total_jointmodule_power = sum(self.arch_jointmodule_power)
        self.arch_total_buf_power = sum(self.arch_buf_power)+(self.global_buf.buf_wpower+self.global_buf.buf_rpower)*1e-3
        self.arch_total_buf_r_power = sum(self.arch_buf_r_power)+self.global_buf.buf_rpower*1e-3
        self.arch_total_buf_w_power = sum(self.arch_buf_w_power)+self.global_buf.buf_wpower*1e-3
        self.arch_total_pooling_power = sum(self.arch_pooling_power)
    
    def calculate_model_power_LUT(self):
        self.global_buf = buffer(SimConfig_path=self.SimConfig_path, buf_level=1,
                                 default_buf_size=self.graph.global_buf_size)
        self.global_buf.calculate_buf_read_power()
        self.global_buf.calculate_buf_write_power()
        self.global_add = adder(SimConfig_path=self.SimConfig_path, bitwidth=self.graph.global_adder_bitwidth)
        self.global_add.calculate_adder_power()
        
        for i in range(self.total_layer_num):
            tile_num = self.graph.layer_tileinfo[i]['tilenum']
            max_column = self.graph.layer_tileinfo[i]['max_column']
            max_row = self.graph.layer_tileinfo[i]['max_row']
            max_PE = self.graph.layer_tileinfo[i]['max_PE']
            max_group = self.graph.layer_tileinfo[i]['max_group']
            layer_type = self.graph.net[i][0][0]['type']
            #linqiushi modified
            rewrite_mode=self.graph.rewrite_mode
            whether_rewrite=0
            if rewrite_mode==1:
                whether_rewrite=self.graph.layer_whether_rewrite[i]
            #linqiushi modified
            #add mix mode
            #mix mode 3: inside tile
            #mix mode 2: among tile
           # if self.mix_mode==3:
                #print("mode=3")
                
            if rewrite_mode==1 and whether_rewrite==1:
                print("这个层需要rewrite-power")
            #elif self.mix_mode==1 or self.mix_mode==2:
            
        tilecount=0  
        for layer_id in range(self.total_layer_num):
    
            k=0
            flag=0
            tilecount=0
            temp_tile_buffer = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.max_outbuf_size)
            temp_tile_buffer.calculate_buf_read_power()
            temp_tile_buffer.calculate_buf_write_power()
            
            PE_inbuf = buffer(SimConfig_path=self.SimConfig_path,buf_level=1,default_buf_size=self.graph.max_inbuf_size)
            PE_inbuf.calculate_buf_read_power()
            PE_inbuf.calculate_buf_write_power()
            while(tilecount<(self.graph.layer_tileinfo[0]['tile_num_mix'][0])**2):
                i=int(self.TCG_mapping.pos_mapping_order[tilecount][0])
                j=int(self.TCG_mapping.pos_mapping_order[tilecount][1])
                tilecount+=1
                
                if self.graph.auto_layer_mapping==0:
                    if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                        flag=0
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                        flag=1
                        tile_power=use_LUT(device_type=self.graph.layer_tileinfo[0]['device_type_mix'][i][j],xbar_size=self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j],\
                                          PE_num=self.graph.layer_tileinfo[0]['PE_num_mix'][i][j],op_type=self.graph.net[layer_id][0][0]['type'])
                        tile_power+=temp_tile_buffer.buf_rpower * 1e-3
                        tile_power+=temp_tile_buffer.buf_wpower * 1e-3
                        
                        if self.graph.net[layer_id][0][0]['type']=='conv' or self.graph.net[layer_id][0][0]['type']=='fc':
                            tile_power+=PE_inbuf.buf_rpower * 1e-3*(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j]**2)
                            tile_power+=PE_inbuf.buf_wpower * 1e-3*(self.graph.layer_tileinfo[0]['PE_num_mix'][i][j]**2)
                        
                        k=k+1
                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])!=layer_id:
                        flag=0
                if flag==1: 
                    self.arch_power[layer_id] += tile_power
                    
        self.arch_total_power = sum(self.arch_power)
        
        
    
    def model_power_output(self, module_information = 1, layer_information = 1):
        print("Hardware power:", self.arch_total_power, "W")
        if module_information:
            print("     write power:",self.arch_total_write_power, "W")
            print("		crossbar power:", self.arch_total_xbar_power, "W")
            print("		DAC power:", self.arch_total_DAC_power, "W")
            print("		ADC power:", self.arch_total_ADC_power, "W")
            print("		Buffer power:", self.arch_total_buf_power, "W")
            print("			|---read buffer power:", self.arch_total_buf_r_power, "W")
            print("			|---write buffer power:", self.arch_total_buf_w_power, "W")
            print("		Pooling power:", self.arch_total_pooling_power, "W")
            print("		Other digital part power:", self.arch_total_digital_power, "W")
            print("			|---adder power:", self.arch_total_adder_power, "W")
            print("			|---output-shift-reg power:", self.arch_total_shiftreg_power, "W")
            print("			|---input-reg power:", self.arch_total_iReg_power, "W")
            print("			|---output-reg power:", self.arch_total_oReg_power, "W")
            print("			|---input_demux power:", self.arch_total_input_demux_power, "W")
            print("			|---output_mux power:", self.arch_total_output_mux_power, "W")
            print("			|---joint_module power:", self.arch_total_jointmodule_power, "W")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                layer_dict = self.NetStruct[i][0][0]
                if layer_dict['type'] == 'element_sum':
                    print("     Hardware power (global accumulator):", self.global_add.adder_power*self.graph.global_adder_num
                          +self.global_buf.buf_wpower*1e-3+self.global_buf.buf_rpower*1e-3, "W")
                elif layer_dict['type'] == 'element_multiply':
                    print("     Hardware power (global accumulator):", self.global_add.adder_power*self.graph.global_multiplier_num
                          +self.global_buf.buf_wpower*1e-3+self.global_buf.buf_rpower*1e-3, "W")
                else:
                    print("     Hardware power:", self.arch_power[i], "W")
    def model_power_output_rewrite(self, module_information = 1, layer_information = 1):
        rewrite_mode=self.graph.rewrite_mode
        assert rewrite_mode==1 or rewrite_mode==2
        for i in range(len(self.arch_total_power)):
            print("Hardware power:at rewrite time "+f"{i}", self.arch_total_power[i], "W")
            if module_information:
                print("     write power:",self.arch_total_write_power[i], "W")
                print("		crossbar power:", self.arch_total_xbar_power[i], "W")
                print("		DAC power:", self.arch_total_DAC_power[i], "W")
                print("		ADC power:", self.arch_total_ADC_power[i], "W")
                print("		Buffer power:", self.arch_total_buf_power[i], "W")
                print("			|---read buffer power:", self.arch_total_buf_r_power[i], "W")
                print("			|---write buffer power:", self.arch_total_buf_w_power[i], "W")
                print("		Pooling power:", self.arch_total_pooling_power[i], "W")
                print("		Other digital part power:", self.arch_total_digital_power[i], "W")
                print("			|---adder power:", self.arch_total_adder_power[i], "W")
                print("			|---output-shift-reg power:", self.arch_total_shiftreg_power[i], "W")
                print("			|---input-reg power:", self.arch_total_iReg_power[i], "W")
                print("			|---output-reg power:", self.arch_total_oReg_power[i], "W")
                print("			|---input_demux power:", self.arch_total_input_demux_power[i], "W")
                print("			|---output_mux power:", self.arch_total_output_mux_power[i], "W")
                print("			|---joint_module power:", self.arch_total_jointmodule_power[i], "W")
            if layer_information:
                for j in range(len(self.graph.rewrite_layer_list[i])):
                    layer_id=self.graph.rewrite_layer_list[i][j]
                    print("Layer", layer_id, ":")
                    layer_dict = self.NetStruct[layer_id][0][0]
                    if layer_dict['type'] == 'element_sum':
                        print("     Hardware power (global accumulator):", self.global_add.adder_power*self.graph.global_adder_num
                            +self.global_buf.buf_wpower*1e-3+self.global_buf.buf_rpower*1e-3, "W")
                    elif layer_dict['type'] == 'element_multiply':
                        print("     Hardware power (global accumulator):", self.global_add.adder_power*self.graph.global_multiplier_num
                            +self.global_buf.buf_wpower*1e-3+self.global_buf.buf_rpower*1e-3, "W")
                    else:
                        print("     Hardware power:", self.arch_power[i][j], "W")
if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()
    __TCG_mapping = TCG(structure_file, test_SimConfig_path)
    __power = Model_inference_power(NetStruct=structure_file,SimConfig_path=test_SimConfig_path,TCG_mapping=__TCG_mapping)
    __power.model_power_output(1,1)