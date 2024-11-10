#!/usr/bin/python
# -*-coding:utf-8-*-
import sys
import os
import configparser as cp
from IPython import embed
import subprocess
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
from IPython import embed

def merge_interval(interval):
    if len(interval) == 0:
        return []
    result = []
    interval.sort()
    lower_bound = interval[0][0]
    upper_bound = interval[0][1]
    for index in range(1, len(interval)):
        if interval[index][0] > upper_bound:
            result.append([lower_bound, upper_bound])
            lower_bound = interval[index][0]
            upper_bound = interval[index][1]
        else:
            if interval[index][1] > upper_bound:
                upper_bound = interval[index][1]
    result.append([lower_bound, upper_bound])
    return result


def Search(value, data):
    pos = 0
    if value > data[-1]:
        return len(data)
    while (value > data[pos]):
        pos += 1
    return pos


def Split_map(padding, outputsize, multiple):  # 对下一层进行划分
    base = outputsize // multiple
    res = outputsize - base * multiple
    split = []  # split the outputsize
    if multiple == 1:
        split.append(outputsize)
    else:
        for i in range(multiple):
            if i < res:
                split.append(base + 1)
            else:
                split.append(base)
    return split

def inoutsize_conversion(kernelsize, padding, stride, outputsize):
    # calculate the input size according to the output size
    return kernelsize+(outputsize-1)*stride-2*padding


class Model_latency():
    def __init__(self, NetStruct, SimConfig_path, multiple=None, TCG_mapping=None, mix_mode=1, mix_tile=None, cnn_step=0):
        modelL_config = cp.ConfigParser()
        modelL_config.read(SimConfig_path, encoding='UTF-8')
        NoC_Compute = int(modelL_config.get('Algorithm Configuration', 'NoC_enable'))
        self.inter_tile_bandwidth = float(modelL_config.get('Tile level', 'Inter_Tile_Bandwidth'))
        self.NetStruct = NetStruct
        if multiple is None:
            multiple = [1] * len(self.NetStruct)
        if TCG_mapping is None:
            TCG_mapping = TCG(NetStruct, SimConfig_path, multiple)
        self.graph = TCG_mapping
        if mix_mode==1:
            self.graph.mapping_net()
        elif mix_mode==2:
            print("zheli")
        print(self.graph.mapping_result)
        self.graph.calculate_transfer_distance()
        self.CNN_step = int(modelL_config.get('Mixmode2/4 Configuration', 'CNN_step'))
        if (self.CNN_step == 1 and cnn_step==0):
            cnn_latency_strp = Model_latency(NetStruct=NetStruct, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping,mix_mode=mix_mode,mix_tile=mix_tile,cnn_step=1)
            cnn_latency_strp.calculate_model_latency(mode=1,mix_mode=mix_mode)
            cnn_latency_strp.model_latency_output_cnn_step()
            embed()

        if mix_mode==2 or mix_mode==4:
            self.graph.mapping_output(mix_tile)
        self.begin_time = []
        self.finish_time = []
        self.layer_tile_latency = []

        mixmodel = cp.ConfigParser()
        mixmodel.read('./mix_tileinfo.ini', encoding='UTF-8')
        self.tile_num = list(map(int, mixmodel.get('tile', 'tile_num').split(',')))
        if (cnn_step==0):
            self.Booksim_Flag = int(modelL_config.get('Mixmode2/4 Configuration', 'Booksim_Flag'))
        else:
            self.Booksim_Flag = 0
        self.Pipe_flag = int(modelL_config.get('Mixmode2/4 Configuration', 'Pipe_flag'))
        self.freq = int(modelL_config.get('Digital module', 'Digital_Frequency'))
        self.Merge_Latency = []
        self.Trans_Latency = []
        if (self.Booksim_Flag == 1):
            self.booksim()

        if NoC_Compute == 1:
            self.Noc_latency = interconnect_estimation()
        else:
            self.Noc_latency = [0] * len(self.NetStruct)
        self.SimConfig_path = SimConfig_path
        self.compute_interval = []
        self.occupancy = []
        self.multiple = multiple

        self.layer_num=TCG_mapping.layer_num
        self.buffer_latency = []
        self.buffer_r_latency = []
        self.buffer_w_latency = []
        self.inbuffer_latency = [] # PE level input buffer latency
        self.outbuffer_latency = [] # Tile level output buffer latency

        self.computing_latency = []
        self.DAC_latency = []
        self.xbar_latency = []
        self.ADC_latency = []
        self.digital_latency = []
        self.iReg_latency = []
        self.input_demux_latency = []
        self.output_mux_latency = []
        self.shiftreg_latency = []
        self.adder_latency = []
        self.oReg_latency = []
        self.jointmodule_latency = []
        self.pooling_latency = []
        self.intra_tile_latency = []
        self.inter_tile_latency = []
        self.tile_merge_latency = []
        self.tile_transfer_latency = []

        self.total_buffer_latency = []
        self.total_computing_latency = []
        self.total_DAC_latency = []
        self.total_xbar_latency = []
        self.total_ADC_latency = []
        self.total_digital_latency = []
        self.total_intra_tile_latency = []
        self.total_inter_tile_latency = []
        self.total_tile_merge_latency = []
        self.total_tile_transfer_latency = []
        self.total_iReg_latency = []
        self.total_oReg_latency = []
        self.total_input_demux_latency = []
        self.total_output_mux_latency = []
        self.total_shiftreg_latency = []
        self.total_adder_latency = []
        self.total_jointmodule_latency = []
        self.total_pooling_latency = []
        self.total_buffer_r_latency = []
        self.total_buffer_w_latency = []

        self.layer_type = []
        self.layer_split = []
        self.pre_max_time = 0

    def Judge(self, last_layer_id ,last_layer_pos, current_layer_id):
        # calculate the position of the most time consuming output of the input layer (used in replicate mode)
        layer_dict = self.NetStruct[current_layer_id][0][0]
        # print(current_layer_id)
        # if layer_dict['type'] is not 'pooling':
            # assert layer_dict['type'] == 'conv', "only conv layer could be judged"
        kernelsize = int(layer_dict['Kernelsize'])
        last_split = self.layer_split[last_layer_id]
        input_size = list(map(int, layer_dict['Inputsize']))[1]
        Row = (last_layer_pos+1) // input_size
        last_column = (last_layer_pos+1) % input_size  # begin from 0
        m = 0
        pos = 0
        while last_column > last_split[m]:
            last_column -= last_split[m]
            m += 1
        if (last_column - kernelsize >= 0) or (m == 0):
            return last_layer_pos
        else:
            for i in range(m):
                pos += last_split[m]  # get the last data point in each multiple
            return pos - 1 + Row * input_size

    def pipe_result_update(self, layer_type='conv', begin_time=0, compute_time=0, layer_id=0,
                           temp_tile_latency=None, temp_pooling_latency = None, global_buf = None,
                           merge_time=0, transfer_time=0, output_size=0):
        if layer_type == 'conv':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.tile_buf_rlatency +
                temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency+temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency+temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)

            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency +
                                                  temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency +
                                                  temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)
            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
        elif layer_type == 'fc':
            self.begin_time[layer_id] = output_size * [begin_time]
            self.finish_time[layer_id] = output_size * [compute_time]
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.tile_buf_rlatency +
                                                temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency+temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency+temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency +
                                                  temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency +
                                                  temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)
            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
        elif layer_type=='MM':
            self.begin_time[layer_id] = output_size * [begin_time]
            self.finish_time[layer_id] = output_size * [compute_time]
            self.compute_interval[layer_id].append([begin_time, compute_time])
            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.tile_buf_rlatency +
                                                temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency+temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency+temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency +
                                                  temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency +
                                                  temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)
            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
        elif layer_type == 'pooling':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])
            self.buffer_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.inbuf_rlatency +
                                                 temp_pooling_latency.outbuf_wlatency + temp_pooling_latency.outbuf_rlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(temp_pooling_latency.inbuf_rlatency + temp_pooling_latency.outbuf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.outbuf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(0)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)

            self.digital_latency[layer_id].append(0)
            self.pooling_latency[layer_id].append(temp_pooling_latency.digital_latency)
            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
        elif layer_type == 'element_sum':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])
            self.buffer_latency[layer_id].append(global_buf.buf_rlatency+global_buf.buf_wlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(global_buf.buf_rlatency)
            self.buffer_w_latency[layer_id].append(global_buf.buf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(0)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)

            self.digital_latency[layer_id].append(10)
            self.pooling_latency[layer_id].append(0)
            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
        
        elif layer_type == 'element_multiply':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])
            self.buffer_latency[layer_id].append(global_buf.buf_rlatency+global_buf.buf_wlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(global_buf.buf_rlatency)
            self.buffer_w_latency[layer_id].append(global_buf.buf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(0)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)

            self.digital_latency[layer_id].append(10)
            self.pooling_latency[layer_id].append(0)
            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)
       
    def calculate_model_latency_nopipe(self):
        # TODO: CHECK THIS FUNCTION
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            if layer_id == 0:
                # for the first layer, first layer must be conv layer
                self.layer_latency_initial()
                output_size = list(map(int, layer_dict['Outputsize']))
                input_size = list(map(int, layer_dict['Inputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['outputbit'])
                # print(self.graph.layer_tileinfo[layer_id]['max_row'])
                input_channel_PE = self.graph.layer_tileinfo[layer_id]['max_row'] / (kernelsize ** 2)
                # the input channel number each PE processes
                temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                          default_inbuf_size=self.graph.max_inbuf_size,
                                                          default_outbuf_size=self.graph.max_outbuf_size
                                                          )
                temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['max_column']*outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                if self.Booksim_Flag == 0 :
                    merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                else :
                    merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq
                # Todo: update merge time (adder tree) and transfer data volume
                # transfer_time = self.graph.transLayer_distance[0][layer_id] * (
                #             outputchannel * outputbit / self.inter_tile_bandwidth)
                if self.Booksim_Flag == 0 :
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                else :
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq
                
                # Todo: update transfer data volume
                for i in range(output_size[0]):
                    for j in range(output_size[1]):
                        if (i == 0) & (j == 0):
                            # the first output
                            indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0)+max(kernelsize - padding, 0)) * inputbit / 8
                            # fill the line buffer
                            rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                            # from the line buffer to the input reg
                            temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                            temp_tile_latency_max=temp_tile_latency.tile_latency
                            if self.Pipe_flag == 1 :
                                compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time)
                            else :
                                compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time
                            begin_time = 0
                            self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                        elif j == 0:
                            indata = input_channel_PE * (input_size[1]*(stride-1)+max(kernelsize-padding,0)) * inputbit / 8
                            # line feed in line buffer
                            rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                            # from the line buffer to the input reg
                            temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                            temp_tile_latency_max=temp_tile_latency.tile_latency
                            begin_time = self.finish_time[0][(i - 1) * output_size[1] + output_size[1] - 1]
                            if self.Pipe_flag == 1 :
                                compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                            else :
                                compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                            self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                        else:
                            indata = input_channel_PE * stride * inputbit /8
                            # write new input data to line buffer
                            rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                            temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                            temp_tile_latency_max=temp_tile_latency.tile_latency
                            begin_time = self.finish_time[0][i * output_size[1] + j - 1]
                            if self.Pipe_flag == 1 :
                                compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                            else :
                                compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                           
                            self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
            else:
                if layer_dict['type'] == 'conv':
                    self.layer_latency_initial()
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    inputindex = Inputindex_list[0]
                    input_channel_PE = self.graph.layer_tileinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    # the input channel number each PE processes
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                              read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                              read_column=self.graph.layer_tileinfo[layer_id][
                                                                  'max_column'],
                                                              indata=0, rdata=0, inprecision=inputbit,
                                                              PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                              default_inbuf_size=self.graph.max_inbuf_size,
                                                              default_outbuf_size=self.graph.max_outbuf_size
                                                              )
                    temp_tile_latency.outbuf.calculate_buf_read_latency(rdata=(self.graph.layer_tileinfo[layer_id]['max_column'] *
                               outputbit * self.graph.layer_tileinfo[layer_id]['max_PE'] / 8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                    if self.Booksim_Flag == 0 :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                    else :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                        
                    # Todo: update merge time (adder tree) and transfer data volume
                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    # Todo: update transfer data volume
                    last_layer_finish_time = max(self.finish_time[layer_id+inputindex])
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            if kernelsize > 1:
                                last_layer_pos = (min(max(kernelsize-padding,1) + stride * i, input_size[0]) - 1) * \
                                                 input_size[1] + min(max(kernelsize-padding,1) + stride * j, input_size[1]) - 1
                            else:
                                last_layer_pos = i*stride*input_size[1]+j*stride

                            if last_layer_pos > len(self.finish_time[layer_id+inputindex]) - 1:
                                print("pos error", i, j)
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) +
                                    max(kernelsize - padding, 0)) * inputbit / 8
                                # fill the line buffer
                                rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                # from the line buffer to the input reg
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                begin_time = last_layer_finish_time
                                
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                                # consider the input data generation time
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                    temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                            elif j == 0:
                                indata = input_channel_PE * (input_size[1] * (stride - 1) + max(kernelsize - padding,0)) * inputbit / 8
                                # line feed in line buffer
                                rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                # from the line buffer to the input reg
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                begin_time = self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1]
                                
                                # max (the required input data generation time, previous point computation complete time)
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                    temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                            else:
                                indata = input_channel_PE * stride * inputbit / 8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                begin_time = self.finish_time[layer_id][i * output_size[1] + j - 1]
                                # max (the required input data generation time, previous point computation complete time)
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                    temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                               
                elif layer_dict['type'] == 'fc':
                    output_size = int(layer_dict['Outfeature'])
                    input_size = int(layer_dict['Infeature'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    inputindex = Inputindex_list[0]
                    self.layer_latency_initial()
                    indata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                    rdata = indata
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                  read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                                  read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                                  indata=indata, rdata=rdata, inprecision=inputbit,
                                                                  PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                  default_inbuf_size=self.graph.max_inbuf_size,
                                                                  default_outbuf_size=self.graph.max_outbuf_size
                                                                  )
                    temp_tile_latency.outbuf.calculate_buf_read_latency(rdata=(self.graph.layer_tileinfo[layer_id]['max_column'] *
                        outputbit * self.graph.layer_tileinfo[layer_id]['max_PE'] / 8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                    if self.Booksim_Flag == 0 :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                    else :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    # Todo: update merge time (adder tree) and transfer data volume
                    

                    begin_time = max(self.finish_time[layer_id+inputindex])
                    if self.Pipe_flag == 1 :
                        compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                    else :
                        compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time
                    
                    self.pipe_result_update(layer_type='fc', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                        temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time, output_size=output_size)
                elif layer_dict['type'] == 'pooling':
                    self.layer_latency_initial()
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    inputindex = Inputindex_list[0]
                    temp_pooling_latency = pooling_latency_analysis(SimConfig_path=self.SimConfig_path,
                        indata=0, rdata=0, outprecision = outputbit,
                        default_inbuf_size = self.graph.max_inbuf_size,
                        default_outbuf_size = self.graph.max_outbuf_size,
                        default_inchannel = inputchannel, default_size = (kernelsize**2))
                  
                    temp_pooling_latency.outbuf.calculate_buf_read_latency(rdata=(outputchannel*outputbit/8))
                    temp_pooling_latency.outbuf_rlatency = temp_pooling_latency.outbuf.buf_rlatency
                    merge_time = temp_pooling_latency.outbuf_rlatency
                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    # Todo: update transfer data volume
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            if (i == 0) & (j == 0):
                                # the first output
                                indata = inputchannel * (input_size[1] * max(kernelsize - padding - 1, 0) + max(
                                    kernelsize - padding, 0)) * inputbit / 8
                                # fill the line buffer
                                rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                # from the line buffer to the input reg
                                temp_pooling_latency.update_pooling_latency(indata=indata,rdata=rdata)
                                begin_time = max(self.finish_time[layer_id+inputindex])
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_pooling_latency.pooling_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)
                            elif j == 0:
                                indata = inputchannel * (input_size[1] * (stride - 1) + max(kernelsize - padding, 0)) * inputbit/8
                                # line feed in line buffer
                                rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                # from the line buffer to the input reg
                                actual_num = indata / inputchannel / (inputbit / 8)
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                begin_time = self.finish_time[layer_id][(i - 1) * output_size[1] + output_size[1] - 1]
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_pooling_latency.pooling_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)
                               
                            else:
                                indata = inputchannel * stride * inputbit / 8
                                # write new input data to line buffer
                                rdata = stride * kernelsize * inputchannel * inputbit / 8
                                actual_num = indata / inputchannel / (inputbit / 8)
                                temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata) 
                                begin_time = self.finish_time[layer_id][i * output_size[1] + j - 1]
                                if self.Pipe_flag == 1 :
                                    compute_time = max(temp_pooling_latency.pooling_latency, merge_time + transfer_time) + begin_time
                                else :
                                    compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)
                elif layer_dict['type'] == 'element_sum':
                    self.layer_latency_initial()
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    assert len(Inputindex_list) > 1, "the number of element_sum's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_sum':
                        idx = idx + 1
                        previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[idx]][0][0]
                    output_size = list(map(int, previous_layer_dict['Outputsize']))
                    input_size = list(map(int, previous_layer_dict['Outputsize']))
                    self.layer_split.append([input_size[1]])
                    kernelsize = int(previous_layer_dict['Kernelsize'])
                    inputchannel = int(previous_layer_dict['Outputchannel'])
                    outputchannel = int(previous_layer_dict['Outputchannel'])
                    inputbit = int(previous_layer_dict['outputbit'])
                    outputbit = int(previous_layer_dict['outputbit'])
                    merge_time = 0
                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.global_buf_size)
                    global_buf.calculate_buf_read_latency(rdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                    global_buf.calculate_buf_write_latency(wdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                    
                    self.pre_max_time = 0
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            max_prelayer_time = 0
                            # the maximum time of the required input data (in all input layers)
                            for idx in Inputindex_list:
                                tmp_time = self.finish_time[layer_id+idx][i*input_size[1]+j]
                                if tmp_time > max_prelayer_time:
                                    max_prelayer_time = tmp_time
                            begin_time = max(max_prelayer_time, self.pre_max_time)
                            compute_time = 10+merge_time+transfer_time+begin_time+global_buf.buf_rlatency+global_buf.buf_wlatency
                            self.pre_max_time = compute_time
                            self.pipe_result_update(layer_type='element_sum', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, global_buf=global_buf, merge_time=merge_time, transfer_time=transfer_time)    
               
                elif layer_dict['type'] == 'element_multiply':
                    self.layer_latency_initial()
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    assert len(Inputindex_list) > 1, "the number of element_multiply's previous layers must > 1"
                    idx = 0
                    previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[0]][0][0]
                    while previous_layer_dict['type'] == 'element_multiply':
                        idx = idx + 1
                        previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[idx]][0][0]
                    output_size = list(map(int, previous_layer_dict['Outputsize']))
                    input_size = list(map(int, previous_layer_dict['Outputsize']))
                    self.layer_split.append([input_size[1]])
                    kernelsize = int(previous_layer_dict['Kernelsize'])
                    inputchannel = int(previous_layer_dict['Outputchannel'])
                    outputchannel = int(previous_layer_dict['Outputchannel'])
                    inputbit = int(previous_layer_dict['outputbit'])
                    outputbit = int(previous_layer_dict['outputbit'])
                    merge_time = 0
                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.global_buf_size)
                    global_buf.calculate_buf_read_latency(rdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                    global_buf.calculate_buf_write_latency(wdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                    self.pre_max_time = 0
                    for i in range(output_size[0]):
                        for j in range(output_size[1]):
                            max_prelayer_time = 0
                            # the maximum time of the required input data (in all input layers)
                            for idx in Inputindex_list:
                                tmp_time = self.finish_time[layer_id+idx][i*input_size[1]+j]
                                if tmp_time > max_prelayer_time:
                                    max_prelayer_time = tmp_time
                            begin_time = max(max_prelayer_time, self.pre_max_time)
                            compute_time = 10+merge_time+transfer_time+begin_time+global_buf.buf_rlatency+global_buf.buf_wlatency
                            self.pre_max_time = compute_time
                            self.pipe_result_update(layer_type='element_multiply', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, global_buf=global_buf, merge_time=merge_time, transfer_time=transfer_time)      
            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1] - self.compute_interval[layer_id][l][0])
            self.occupancy.append(1)
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_inter_tile_latency.append(sum(self.inter_tile_latency[layer_id]))
            self.total_intra_tile_latency.append(sum(self.intra_tile_latency[layer_id]))
            self.total_tile_merge_latency.append(sum(self.tile_merge_latency[layer_id]))
            self.total_tile_transfer_latency.append(sum(self.tile_transfer_latency[layer_id]))
            self.total_iReg_latency.append(sum(self.iReg_latency[layer_id]))
            self.total_oReg_latency.append(sum(self.oReg_latency[layer_id]))
            self.total_input_demux_latency.append(sum(self.input_demux_latency[layer_id]))
            self.total_output_mux_latency.append(sum(self.output_mux_latency[layer_id]))
            self.total_shiftreg_latency.append(sum(self.shiftreg_latency[layer_id]))
            self.total_adder_latency.append(sum(self.adder_latency[layer_id]))
            self.total_jointmodule_latency.append(sum(self.jointmodule_latency[layer_id]))
            self.total_pooling_latency.append(sum(self.pooling_latency[layer_id]))
            self.total_buffer_r_latency.append(sum(self.buffer_r_latency[layer_id]))
            self.total_buffer_w_latency.append(sum(self.buffer_w_latency[layer_id]))

    def Latency_stall_calculate(self):
        ''' should be used after the calculate_model '''
        Linebuffer_Size = 2048  # Bytes
        OutputBuffer_Size = 32 * 1024  # Bytes
        layer_occu = []
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            self.layer_type.append(layer_dict['type'])
            if (self.occupancy[layer_id] == 1) and (layer_dict['type'] == 'conv'):
                # if ((self.occupancy[layer_id] == 1) and (layer_dict['type'] == 'conv')) or (layer_dict['type'] == 'pooling'):
                layer_occu.append(layer_id)
        ''' check the consecuive of the layer '''
        if len(layer_occu) == 0:
            return
        print(layer_occu)
        layer_stall = []
        start = layer_occu[0]
        end = start
        for i in range(len(layer_occu) - 1):
            if layer_occu[i + 1] == layer_occu[i] + 1:
                end = layer_occu[i + 1]
            else:
                if start < end:
                    layer_stall.append([start, end])
                start = layer_occu[i + 1]
                end = start
        if end > start:
            layer_stall.append([start, end])
        if len(layer_stall) == 0:
            print("No need to be stalled")
            return
        else:
            # print(layer_stall)
            for i in range(len(layer_stall)):
                for layer_id in range(layer_stall[i][1], layer_stall[i][0], -1):
                    layer_dict = self.NetStruct[layer_id][0][0]
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    inputindex = Inputindex_list[0]
                    input_channel_PE = self.graph.layer_tileinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    ''' get the point number of this layer and then go back to the previous layer '''
                    # TODO: update the tile usage of this
                    tile_num = self.graph.layer_tileinfo[layer_id]['tilenum']
                    pre_point = 0
                    cur_point = 0
                    res = 0
                    if layer_dict['type'] == 'conv':
                        storage_capacity = Linebuffer_Size / input_channel_PE + OutputBuffer_Size * tile_num / outputchannel
                    else:
                        storage_capacity = Linebuffer_Size / inputchannel + OutputBuffer_Size * tile_num / outputchannel
                    # print("Storage is: ", storage_capacity)
                    for cur_point in range(len(self.begin_time[layer_id])):
                        cur_row = cur_point // output_size[1]  # begin from 0
                        cur_column = cur_point - cur_row * output_size[1]  # begin from 0
                        used_point = (stride * cur_row - padding) * input_size[1] + \
                                     (cur_column * stride - padding) * stride
                        pre_point = Search(self.begin_time[layer_id][cur_point], self.begin_time[layer_id+inputindex])
                        # begin from 1
                        res = storage_capacity - (pre_point + cur_point - used_point)
                        # print(res)
                        if res <= 0:
                            print("You need to stall the Pipeline on Layer %d" % (layer_id+inputindex))
                            break
                    # update the stall time
                    if res > 0:
                        print("No need to be stalled")
                        continue
                    else:
                        pre_point = pre_point - 1
                        # print(pre_point)
                        while (pre_point < input_size[0] * input_size[1]):
                            delta = self.begin_time[layer_id][cur_point] - self.begin_time[layer_id+inputindex][pre_point]
                            assert delta > 0, "delta is not 0, something error"
                            # self.begin_time[layer_id - 1][pre_point] = self.begin_time[layer_id][cur_point]
                            consumption = stride ** 2
                            for num in range(consumption):
                                self.begin_time[layer_id+inputindex][pre_point + num] += delta
                                self.finish_time[layer_id+inputindex][pre_point + num] += delta
                                pre_point += consumption
                            cur_point += 1
                        interval = []
                        for i in range(len(self.begin_time[layer_id+inputindex])):
                            interval.append([self.begin_time[layer_id+inputindex][i], self.finish_time[layer_id+inputindex][i]])
                        stall_interval = merge_interval(interval)
                        self.compute_interval[layer_id+inputindex] = stall_interval
                        print("++++++++++++++++++++++++++++++++")
                        print("updated: ", self.begin_time[layer_id+inputindex])
                        print("         ", self.finish_time[layer_id+inputindex])
                        print("         ", self.compute_interval[layer_id+inputindex])
                        print(len(stall_interval))
        return

    def model_latency_output(self, module_information=1, layer_information=1):
        print(' ')
        if (layer_information):
            for i in range(len(self.begin_time)):
                print("Layer", i, " type:", self.NetStruct[i][0][0]['type'])
                # print("start time: ", self.begin_time[i])
                # print("finish time:", self.finish_time[i])
                # print("Time interval of working:", self.compute_interval[i])
                print("Occupancy:", self.occupancy[i])
                #     # print(self.xbar_latency[i])
                total_latency = self.total_buffer_latency[i] + self.total_computing_latency[i] + \
                                self.total_digital_latency[i] + self.total_intra_tile_latency[i] + \
                                self.total_inter_tile_latency[i]
                if (module_information):
                    ##### for test #####
                    input_l=self.NetStruct[i][0][0]['Inputindex']
                    final_idx=list(map(int, input_l))
                    print("total latency:", total_latency)
                    if i == 0:
                        print("layer latency:", max(self.finish_time[i]))
                    else:
                        print("layer latency:", max(self.finish_time[i])-max(self.finish_time[i+final_idx[0]]))

                    print("Buffer latency of layer", i, ":", self.total_buffer_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_latency[i] / total_latency), '%)')
                    print("     read buffer latency of layer", i, ":", self.total_buffer_r_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_r_latency[i] / total_latency), '%)')
                    print("     write buffer latency of layer", i, ":", self.total_buffer_w_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_w_latency[i] / total_latency), '%)')
                    print("Computing latency of layer", i, ":", self.total_computing_latency[i], '(',
                          "%.2f" % (100 * self.total_computing_latency[i] / total_latency), '%)')
                    print("     DAC latency of layer", i, ":", self.total_DAC_latency[i], '(',
                          "%.2f" % (100 * self.total_DAC_latency[i] / total_latency), '%)')
                    print("     ADC latency of layer", i, ":", self.total_ADC_latency[i], '(',
                          "%.2f" % (100 * self.total_ADC_latency[i] / total_latency), '%)')
                    print("     xbar latency of layer", i, ":", self.total_xbar_latency[i], '(',
                          "%.2f" % (100 * self.total_xbar_latency[i] / total_latency), '%)')
                    print("Digital part latency of layer", i, ":", self.total_digital_latency[i], '(',
                          "%.2f" % (100 * self.total_digital_latency[i] / total_latency), '%)')
                    print("     iReg latency of layer", i, ":", self.total_iReg_latency[i], '(',
                          "%.2f" % (100 * self.total_iReg_latency[i] / total_latency), '%)')
                    print("     oReg latency of layer", i, ":", self.total_oReg_latency[i], '(',
                          "%.2f" % (100 * self.total_oReg_latency[i] / total_latency), '%)')
                    print("     input demux latency of layer", i, ":", self.total_input_demux_latency[i], '(',
                          "%.2f" % (100 * self.total_input_demux_latency[i] / total_latency), '%)')
                    print("     output mux latency of layer", i, ":", self.total_output_mux_latency[i], '(',
                          "%.2f" % (100 * self.total_output_mux_latency[i] / total_latency), '%)')
                    print("     shiftreg latency of layer", i, ":", self.total_shiftreg_latency[i], '(',
                          "%.2f" % (100 * self.total_shiftreg_latency[i] / total_latency), '%)')
                    print("     adder latency of layer", i, ":", self.total_adder_latency[i], '(',
                          "%.2f" % (100 * self.total_adder_latency[i] / total_latency), '%)')
                    print("     Jointmodule latency of layer", i, ":", self.total_jointmodule_latency[i], '(',
                          "%.2f" % (100 * self.total_jointmodule_latency[i] / total_latency), '%)')
                    print("Pooling module latency of layer", i, ":", self.total_pooling_latency[i], '(',
                          "%.2f" % (100 * self.total_pooling_latency[i] / total_latency), '%)')
                    print("Intra tile communication latency of layer", i, ":", self.total_intra_tile_latency[i], '(',
                          "%.2f" % (100 * self.total_intra_tile_latency[i] / total_latency), '%)')
                    print("Inter tile communication latency of layer", i, ":", self.total_inter_tile_latency[i], '(',
                          "%.2f" % (100 * self.total_inter_tile_latency[i] / total_latency), '%)')
                    print("     One layer merge latency of layer", i, ":", self.total_tile_merge_latency[i], '(',
                          "%.2f" % (100 * self.total_tile_merge_latency[i] / total_latency), '%)')
                    print("     Inter tile transfer latency of layer", i, ":", self.total_tile_transfer_latency[i], '(',
                          "%.2f" % (100 * self.total_tile_transfer_latency[i] / total_latency), '%)')
                print('----------------------------------------------')
        # print("Latency simulation finished!")
        print("Entire latency:", max(max(self.finish_time)), "ns")

    def model_latency_output_cnn_step(self, module_information=1, layer_information=1):
        print(' ')
        if (layer_information):
            for i in range(len(self.begin_time)):
                print("Layer", i, " type:", self.NetStruct[i][0][0]['type'])
                # print("start time: ", self.begin_time[i])
                # print("finish time:", self.finish_time[i])
                # print("Time interval of working:", self.compute_interval[i])
                print("Occupancy:", self.occupancy[i])
                #     # print(self.xbar_latency[i])
                total_latency = self.total_buffer_latency[i] + self.total_computing_latency[i] + \
                                self.total_digital_latency[i] + self.total_intra_tile_latency[i] + \
                                self.total_inter_tile_latency[i]
                if (module_information):
                    ##### for test #####
                    input_l=self.NetStruct[i][0][0]['Inputindex']
                    final_idx=list(map(int, input_l))
                    print("total latency:", total_latency)
                    if i == 0:
                        print("layer latency:", max(self.finish_time[i]))
                    else:
                        print("layer latency:", max(self.finish_time[i])-max(self.finish_time[i+final_idx[0]]))

                    print("Buffer latency of layer", i, ":", self.total_buffer_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_latency[i] / total_latency), '%)')
                    print("     read buffer latency of layer", i, ":", self.total_buffer_r_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_r_latency[i] / total_latency), '%)')
                    print("     write buffer latency of layer", i, ":", self.total_buffer_w_latency[i], '(',
                          "%.2f" % (100 * self.total_buffer_w_latency[i] / total_latency), '%)')
                    print("Computing latency of layer", i, ":", self.total_computing_latency[i], '(',
                          "%.2f" % (100 * self.total_computing_latency[i] / total_latency), '%)')
                    print("     DAC latency of layer", i, ":", self.total_DAC_latency[i], '(',
                          "%.2f" % (100 * self.total_DAC_latency[i] / total_latency), '%)')
                    print("     ADC latency of layer", i, ":", self.total_ADC_latency[i], '(',
                          "%.2f" % (100 * self.total_ADC_latency[i] / total_latency), '%)')
                    print("     xbar latency of layer", i, ":", self.total_xbar_latency[i], '(',
                          "%.2f" % (100 * self.total_xbar_latency[i] / total_latency), '%)')
                    print("Digital part latency of layer", i, ":", self.total_digital_latency[i], '(',
                          "%.2f" % (100 * self.total_digital_latency[i] / total_latency), '%)')
                    print("     iReg latency of layer", i, ":", self.total_iReg_latency[i], '(',
                          "%.2f" % (100 * self.total_iReg_latency[i] / total_latency), '%)')
                    print("     oReg latency of layer", i, ":", self.total_oReg_latency[i], '(',
                          "%.2f" % (100 * self.total_oReg_latency[i] / total_latency), '%)')
                    print("     input demux latency of layer", i, ":", self.total_input_demux_latency[i], '(',
                          "%.2f" % (100 * self.total_input_demux_latency[i] / total_latency), '%)')
                    print("     output mux latency of layer", i, ":", self.total_output_mux_latency[i], '(',
                          "%.2f" % (100 * self.total_output_mux_latency[i] / total_latency), '%)')
                    print("     shiftreg latency of layer", i, ":", self.total_shiftreg_latency[i], '(',
                          "%.2f" % (100 * self.total_shiftreg_latency[i] / total_latency), '%)')
                    print("     adder latency of layer", i, ":", self.total_adder_latency[i], '(',
                          "%.2f" % (100 * self.total_adder_latency[i] / total_latency), '%)')
                    print("     Jointmodule latency of layer", i, ":", self.total_jointmodule_latency[i], '(',
                          "%.2f" % (100 * self.total_jointmodule_latency[i] / total_latency), '%)')
                    print("Pooling module latency of layer", i, ":", self.total_pooling_latency[i], '(',
                          "%.2f" % (100 * self.total_pooling_latency[i] / total_latency), '%)')
                    print("Intra tile communication latency of layer", i, ":", self.total_intra_tile_latency[i], '(',
                          "%.2f" % (100 * self.total_intra_tile_latency[i] / total_latency), '%)')
                    print("Inter tile communication latency of layer", i, ":", self.total_inter_tile_latency[i], '(',
                          "%.2f" % (100 * self.total_inter_tile_latency[i] / total_latency), '%)')
                    print("     One layer merge latency of layer", i, ":", self.total_tile_merge_latency[i], '(',
                          "%.2f" % (100 * self.total_tile_merge_latency[i] / total_latency), '%)')
                    print("     Inter tile transfer latency of layer", i, ":", self.total_tile_transfer_latency[i], '(',
                          "%.2f" % (100 * self.total_tile_transfer_latency[i] / total_latency), '%)')
                print('----------------------------------------------')
        # print("Latency simulation finished!")
        print("Entire latency:", max(max(self.finish_time)), "ns")

    def layer_latency_initial(self):
        self.begin_time.append([])
        self.finish_time.append([])
        self.compute_interval.append([])
        self.buffer_latency.append([])
        self.computing_latency.append([])
        self.DAC_latency.append([])
        self.xbar_latency.append([])
        self.ADC_latency.append([])
        self.buffer_r_latency.append([])
        self.buffer_w_latency.append([])
        self.inbuffer_latency.append([])
        self.outbuffer_latency.append([])
        self.iReg_latency.append([])
        self.input_demux_latency.append([])
        self.output_mux_latency.append([])
        self.shiftreg_latency.append([])
        self.adder_latency.append([])
        self.oReg_latency.append([])
        self.jointmodule_latency.append([])
        self.pooling_latency.append([])
        self.digital_latency.append([])
        self.intra_tile_latency.append([])
        self.inter_tile_latency.append([])
        self.tile_merge_latency.append([])
        self.tile_transfer_latency.append([])

    def calculate_model_latency(self, mode=0,mix_mode=1):
        '''
        merge the latency_0 and latency_1
        :param mode: 0: fill in input data row by row, 1: fill in input data kerlenl size by kernel size (column direction)
        :return:
        '''
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            #linqiushi modified
            if mix_mode==1 or mix_mode==2:
                whether_rewrite=0
                rewrite_mode=self.graph.rewrite_mode
                if rewrite_mode==1:
                    for i in range(len(self.graph.start_layer)):
                        if i!=0 and layer_id==self.graph.start_layer[i]:
                            whether_rewrite=1
                
                #whether_rewrite=0
                if rewrite_mode==1 and whether_rewrite==1:
                    print("这个层需要rewrite-latency")
            elif mix_mode==3:
                whether_rewrite=0
                rewrite_mode=self.graph.rewrite_mode
                if rewrite_mode==1:
                    for i in range(len(self.graph.final_layer)):
                        if layer_id==self.graph.final_layer[i]:
                            whether_rewrite=1
                self.graph.layer_whether_rewrite.append(whether_rewrite)
            elif mix_mode==4:
                rewrite_mode=self.graph.rewrite_mode
            #linqiushi above
            if layer_id == 0:
                # for the first layer, first layer must be conv layer
                self.layer_latency_initial()
                output_size = list(map(int, layer_dict['Outputsize']))
                input_size = list(map(int, layer_dict['Inputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                inputchannel = int(layer_dict['Inputchannel'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['outputbit'])
                #linqiushi modified
                if mix_mode==3:
                    input_channel_PE=self.graph.layer_tileinfo[layer_id]['tile_max_row'][0] / (kernelsize ** 2)
                #linqiushi above
                else:
                    input_channel_PE = self.graph.layer_tileinfo[layer_id]['max_row'] / (kernelsize ** 2)
                # the input channel number each PE processes
                if mix_mode==1:
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                          default_inbuf_size=self.graph.max_inbuf_size,
                                                          default_outbuf_size=self.graph.max_outbuf_size,
                                                          mix_mode=mix_mode
                                                          )
                #linqiushi modified
                elif mix_mode==2:
                    temp_device_type=[]
                    temp_PE_num=[]
                    temp_xbar_size=[]
                    temp_pos=[]
                    for i in range(self.graph.layer_tileinfo[0]['tile_num_mix'][0]):
                        for j in range(self.graph.layer_tileinfo[0]['tile_num_mix'][1]):
                            if self.graph.auto_layer_mapping==0:
                                if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                                    pass
                                elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                                    
                                    temp_pos.append([i,j])
                                    temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                    temp_PE_num.append((self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])**2)
                                    temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                            else:
                                if self.graph.mapping_result[i][j]==layer_id:
                                    temp_pos.append([i,j])
                                    temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                    temp_PE_num.append(self.graph.layer_tileinfo[layer_id]['max_PE'][i][j])
                                    temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                                
                    
                    temp_tile_latency_max=0
                    for i in range(len(temp_device_type)):
                        
                        temp_tile_latency0 = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                            read_row=self.graph.layer_tileinfo[layer_id]['max_row_mix'][i],
                                                            read_column=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i],
                                                            indata=0, rdata=0, inprecision=inputbit,
                                                            PE_num=temp_PE_num[i],
                                                            default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size,
                                                            device_type=temp_device_type[i],
                                                            xbar_size=[int(temp_xbar_size[i]),int(temp_xbar_size[i])],
                                                            mix_mode=mix_mode
                                                            )
                        if temp_tile_latency0.tile_latency>temp_tile_latency_max:
                            temp_tile_latency_max=temp_tile_latency0.tile_latency
                            temp_tile_latency = temp_tile_latency0
                            
                elif mix_mode==3:
                    
                    temp_tile_latency_max=0
                    for i in range(len(self.graph.layer_tileinfo[layer_id]['tile_max_row'])):
                        
                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                            read_row=self.graph.layer_tileinfo[layer_id]['tile_max_row'][i],
                                                            read_column=self.graph.layer_tileinfo[layer_id]['tile_max_column'][i],
                                                            indata=0, rdata=0, inprecision=inputbit,
                                                            PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                            default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size,
                                                            device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                            xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i]),int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i])],
                                                            mix_mode=mix_mode
                                                            )  
                        if temp_tile_latency.tile_latency>temp_tile_latency_max:
                            temp_tile_latency_max=temp_tile_latency.tile_latency  
                elif mix_mode==4:
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                          read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                          read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                          indata=0, rdata=0, inprecision=inputbit,
                                                          device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                          PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                          default_inbuf_size=self.graph.max_inbuf_size,
                                                          default_outbuf_size=self.graph.max_outbuf_size,
                                                          xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size']),int(self.graph.layer_tileinfo[layer_id]['xbar_size'])],
                                                          mix_mode=mix_mode
                                                          ) 
                #assert 0
                #linqiushi above
                #linqiushi modified:
                if mix_mode==3:
                    temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['tile_max_column'][0]*
                                                                             outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                    if self.Booksim_Flag == 0 :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                    else :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                elif mix_mode==2:
                    
                    MAX=0
                    for i in range(len(temp_pos)):
                        j=temp_pos[i][0]
                        k=temp_pos[i][1]
                        
                        if MAX<self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]:
                            MAX=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]
                    temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (MAX*outputbit/8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                    if self.Booksim_Flag == 0 :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth)
                    else :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth) *  1000 / self.freq + self.Merge_Latency[layer_id] *  1000 / self.freq

                    print("现在是",MAX,MAX*outputbit/8,merge_time)
                elif mix_mode==1 or mix_mode==4:
                    temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['max_column']*
                                                                                outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                    if self.Booksim_Flag == 0 :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                    else :
                        merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                    
                # Todo: update merge time (adder tree) and transfer data volume
                if self.Booksim_Flag == 0 :
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                else :
                    transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                print("transfertime",transfer_time)
                cur_multiple = self.multiple[layer_id]
                split_size = Split_map(padding=padding, outputsize=output_size[1], multiple=cur_multiple)
                self.layer_split.append(split_size)
                max_time = [0] * cur_multiple
                # Todo: update transfer data volume
                for i in range(output_size[0]):
                    for m in range(cur_multiple):
                        for j in range(split_size[m]):
                            self.pre_max_time = max_time[m]
                            if (i == 0) & (j == 0):
                                # the first output
                                if mode == 0:
                                    if cur_multiple == 1:
                                        indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) +
                                                                     max(kernelsize - padding, 0)) * inputbit / 8
                                    elif m == 0:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=padding/2, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                                     max(kernelsize - padding, 0)) * inputbit / 8
                                    elif m == cur_multiple-1:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=padding/2, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                                     kernelsize) * inputbit / 8
                                    else:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=0, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                                     kernelsize) * inputbit / 8
                                else:
                                    if cur_multiple == 1:
                                        indata = input_channel_PE * (max(kernelsize - padding, 0)**2) * inputbit / 8
                                    elif m == 0:
                                        indata = input_channel_PE * (max(kernelsize - padding, 0)**2) * inputbit / 8
                                    else:
                                        indata = input_channel_PE * (max(kernelsize-padding,0)*kernelsize) * inputbit / 8                  
                                # fill the line buffer
                                rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                if mix_mode==1 or mix_mode==4:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time)
                                    else :
                                        compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time 
                                elif mix_mode==2 or mix_mode==3:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency_max, merge_time + transfer_time)
                                    else :
                                        compute_time = temp_tile_latency_max + merge_time + transfer_time
                                begin_time = 0
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                        temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                max_time[m] = compute_time
                            elif j == 0:
                                if mode == 0:
                                    if cur_multiple == 1:
                                        indata = input_channel_PE * (input_size[1]*(stride-1)+max(kernelsize-padding,0)) * inputbit / 8
                                    elif m == 0:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=padding/2, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize*(stride-1)+max(kernelsize - padding, 0)) * inputbit / 8
                                    elif m == cur_multiple-1:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=padding/2, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize * (stride-1) + kernelsize) * inputbit / 8
                                    else:
                                        temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=0, stride=stride,
                                                                           outputsize=split_size[m]) # only one padding column
                                        indata = input_channel_PE * (temp_insize * (stride-1) + kernelsize) * inputbit / 8
                                else:
                                    if cur_multiple == 1:
                                        indata = input_channel_PE * stride * max(kernelsize-padding,0) * inputbit / 8
                                    elif m == 0:
                                        indata = input_channel_PE * stride * max(kernelsize-padding,0) * inputbit /8
                                    else:
                                        indata = input_channel_PE * stride * kernelsize * inputbit / 8
                                rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                # TODO: Check
                                begin_time = self.pre_max_time
                                if mix_mode==1 or mix_mode==4:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                                elif mix_mode==2 or mix_mode==3:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                                #change to temp_tile_latency_max
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                        temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                max_time[m] = compute_time
                            else:
                                # ignore the last several columns with padding
                                if mode == 0:
                                    indata = input_channel_PE * stride * inputbit /8
                                else:
                                    if i == 0:
                                        indata = input_channel_PE * stride * kernelsize * inputbit / 8
                                    else:
                                        indata = input_channel_PE * stride **2 * inputbit / 8
                                rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                                temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                temp_tile_latency_max=temp_tile_latency.tile_latency
                                begin_time = self.pre_max_time
                                if mix_mode==1 or mix_mode==4:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                                elif mix_mode==2 or mix_mode==3:
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                                self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                        temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                max_time[m] = compute_time
            else:
                if layer_dict['type'] == 'conv':
                    self.layer_latency_initial()
                    output_size = list(map(int, layer_dict['Outputsize']))
                    input_size = list(map(int, layer_dict['Inputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    inputchannel = int(layer_dict['Inputchannel'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['outputbit'])
                    Inputindex_list = list(map(int, layer_dict['Inputindex']))
                    inputindex = Inputindex_list[0]
                    input_channel_PE = self.graph.layer_tileinfo[layer_id]['max_row'] / (kernelsize ** 2)
                    # the input channel number each PE processes
                    if mix_mode==1:
                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                                read_column=self.graph.layer_tileinfo[layer_id][
                                                                    'max_column'],
                                                                indata=0, rdata=0, inprecision=inputbit,
                                                                PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size,
                                                                
                                                                )
                    #linqiushi modified
                    if mix_mode==2:
                        temp_device_type=[]
                        temp_PE_num=[]
                        temp_xbar_size=[]
                        temp_pos=[]
                        for i in range(self.graph.layer_tileinfo[0]['tile_num_mix'][0]):
                            for j in range(self.graph.layer_tileinfo[0]['tile_num_mix'][1]):
                               
                                if self.graph.auto_layer_mapping==0:
                                    if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                                        pass
                                    elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                                        
                                        temp_pos.append([i,j])
                                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                        temp_PE_num.append((self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])**2)
                                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                                else:
                                    if self.graph.mapping_result[i][j]==layer_id:
                                        temp_pos.append([i,j])
                                        temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                        temp_PE_num.append(self.graph.layer_tileinfo[layer_id]['max_PE'][i][j])
                                        temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                        
                        temp_tile_latency_max=0
                        for i in range(len(temp_device_type)):
                            #max_PE,row,column待覆盖
                            temp_tile_latency0 = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                read_row=self.graph.layer_tileinfo[layer_id]['max_row_mix'][i],
                                                            read_column=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i],
                                                                indata=0, rdata=0, inprecision=inputbit,
                                                                PE_num=temp_PE_num[i],
                                                                default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size,
                                                                device_type=temp_device_type[i],
                                                                xbar_size=[int(temp_xbar_size[i]),int(temp_xbar_size[i])],
                                                                mix_mode=mix_mode
                                                                )
                            if temp_tile_latency0.tile_latency>temp_tile_latency_max:
                                temp_tile_latency_max=temp_tile_latency0.tile_latency
                                temp_tile_latency = temp_tile_latency0
                                
                    elif mix_mode==3:
                    
                        temp_tile_latency_max=0
                        for i in range(len(self.graph.layer_tileinfo[layer_id]['tile_max_row'])):
                            
                            temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                read_row=self.graph.layer_tileinfo[layer_id]['tile_max_row'][i],
                                                                read_column=self.graph.layer_tileinfo[layer_id]['tile_max_column'][i],
                                                                indata=0, rdata=0, inprecision=inputbit,
                                                                PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size,
                                                                device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                                xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i]),int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i])],
                                                                mix_mode=mix_mode
                                                                )  
                            if temp_tile_latency.tile_latency>temp_tile_latency_max:
                                temp_tile_latency_max=temp_tile_latency.tile_latency 
                    elif mix_mode==4:
                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                            read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                            read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                            indata=0, rdata=0, inprecision=inputbit,
                                                            device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                            PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                            default_inbuf_size=self.graph.max_inbuf_size,
                                                            default_outbuf_size=self.graph.max_outbuf_size,
                                                            xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size']),int(self.graph.layer_tileinfo[layer_id]['xbar_size'])],
                                                            mix_mode=mix_mode
                                                            ) 
                    #assert 0
                    #linqiushi above
                    if mix_mode==3:
                        temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['tile_max_column'][0]*
                                                                                outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                        if self.Booksim_Flag == 0 :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                        else :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                    elif mix_mode==2:
                    
                        MAX=0
                        for i in range(len(temp_pos)):
                            j=temp_pos[i][0]
                            k=temp_pos[i][1]
                            if MAX<self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]:
                                MAX=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]
                        temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (MAX*outputbit/8))
                        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                        if self.Booksim_Flag == 0 :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth)
                        else :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq
                        
                    elif mix_mode==1 or mix_mode==4:
                        temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['max_column']*
                                                                                    outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                        if self.Booksim_Flag == 0 :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                        (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                        self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                        else :
                            merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                        (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                        self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq
                    
                        
                    # Todo: update merge time (adder tree) and transfer data volume
                    if self.Booksim_Flag == 0 :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                    else :
                        transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                    
                    ''' get the multiple for the conv layer '''
                    cur_multiple = self.multiple[layer_id]
                    split_size = Split_map(padding=padding, outputsize=output_size[1], multiple=cur_multiple)
                    self.layer_split.append(split_size)
                    max_time = [0] * cur_multiple
                   
                    for i in range(output_size[0]):
                        for m in range(cur_multiple):
                            for j in range(split_size[m]):
                                self.pre_max_time = max_time[m]
                                if kernelsize > 1:
                                    last_layer_pos = (min(max(kernelsize-padding,1) + stride * i, input_size[0]) - 1) * \
                                                 input_size[1] + min(max(kernelsize-padding,1) + stride * j, input_size[1]) - 1
                                    
                                else:
                                    last_layer_pos = i*stride*input_size[1]+j*stride
                                # if last_layer_pos > len(self.finish_time[layer_id - 1]) - 1:
                                #     print("pos error", i, j)
                                if (i == 0) & (j == 0):
                                    ''' the first output '''
                                    if mode == 0:
                                        if cur_multiple == 1:
                                            indata = input_channel_PE * (input_size[1] * max(kernelsize - padding - 1, 0) +
                                                        max(kernelsize - padding, 0)) * inputbit / 8
                                        elif m == 0:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize,padding=padding / 2, stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                        max(kernelsize - padding, 0)) * inputbit / 8
                                        elif m == cur_multiple - 1:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize,padding=padding / 2, stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                        kernelsize) * inputbit / 8
                                        else:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=0,stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * max(kernelsize - padding - 1, 0) +
                                                        kernelsize) * inputbit / 8
                                    else:
                                        if cur_multiple == 1:
                                            indata = input_channel_PE * (
                                                        max(kernelsize - padding, 0) ** 2) * inputbit / 8
                                        elif m == 0:
                                            indata = input_channel_PE * (
                                                        max(kernelsize - padding, 0) ** 2) * inputbit / 8
                                        else:
                                            indata = input_channel_PE * (
                                                        max(kernelsize - padding, 0) * kernelsize) * inputbit / 8
                                    rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                    temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                    temp_tile_latency_max=temp_tile_latency.tile_latency
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    # the maximum time of the required input data (in all input layers)
                                    
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                            #add the weight_transfer latency and write latency
                                            weight_transfer_time=self.graph.weight_transfer_time(layer_id)
                                            write_latency=self.graph.calculate_write_latency
                                    else:
                                        for idx in temp_Inputindex:
                                            if cur_multiple == 1:
                                                tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                                
                                            else:
                                                updated_last_layer_pos = self.Judge(last_layer_id=(layer_id+idx),last_layer_pos=last_layer_pos,current_layer_id=layer_id)
                                                tmp_time = self.finish_time[layer_id + idx][updated_last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    
                                    if mix_mode==1 or mix_mode==4:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                                    elif mix_mode==2 or mix_mode==3:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                                   
                                    self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                    max_time[m] = compute_time
                                elif j == 0:
                                    if mode == 0:
                                        if cur_multiple == 1:
                                            indata = input_channel_PE * (input_size[1] * (stride - 1) + max(kernelsize - padding,0)) * inputbit / 8
                                        elif m == 0:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize,padding=padding / 2, stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * (stride - 1) + max(kernelsize - padding, 0)) * inputbit / 8
                                        elif m == cur_multiple - 1:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize,padding=padding / 2, stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * (stride - 1) + kernelsize) * inputbit / 8
                                        else:
                                            temp_insize = inoutsize_conversion(kernelsize=kernelsize, padding=0,stride=stride,
                                                                               outputsize=split_size[m])  # only one padding column
                                            indata = input_channel_PE * (temp_insize * (stride - 1) + kernelsize) * inputbit / 8
                                    else:
                                        if cur_multiple == 1:
                                            indata = input_channel_PE * stride * max(kernelsize - padding,0) * inputbit / 8
                                        elif m == 0:
                                            indata = input_channel_PE * stride * max(kernelsize - padding,0) * inputbit / 8
                                        else:
                                            indata = input_channel_PE * stride * kernelsize * inputbit / 8
                                    rdata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                                    temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                    temp_tile_latency_max=temp_tile_latency.tile_latency
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    # the maximum time of the required input data (in all input layers)
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                    else:
                                        for idx in temp_Inputindex:
                                            if cur_multiple == 1:
                                                tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                            else:
                                                updated_last_layer_pos = self.Judge(last_layer_id=(layer_id + idx),
                                                                                    last_layer_pos=last_layer_pos,
                                                                                    current_layer_id=layer_id)
                                                tmp_time = self.finish_time[layer_id + idx][updated_last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    if mix_mode==1 or mix_mode==4:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                                    elif mix_mode==2 or mix_mode==3:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                                    self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                    max_time[m] = compute_time
                                    
                                else:
                                    if mode == 0:
                                        indata = input_channel_PE * stride * inputbit / 8
                                    else:
                                        if i ==0:
                                            indata = input_channel_PE * stride * kernelsize * inputbit / 8
                                        else:
                                            indata = input_channel_PE * stride**2 * inputbit / 8
                                    rdata = stride * kernelsize * input_channel_PE * inputbit / 8
                                    temp_tile_latency.update_tile_latency(indata=indata, rdata=rdata)
                                    temp_tile_latency_max=temp_tile_latency.tile_latency
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    # the maximum time of the required input data (in all input layers)
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                    else:
                                        for idx in temp_Inputindex:
                                            #print("前面层的类型：",self.NetStruct[layer_id+idx][0][0]["type"],"自己类型：",layer_dict["type"])
                                            if cur_multiple == 1:
                                                #if(layer_id+idx>0):
                                                    #print("last_layer_pos:",last_layer_pos,"layer_id+idx:",layer_id+idx,temp_Inputindex)
                                                tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                            else:
                                                updated_last_layer_pos = self.Judge(last_layer_id=(layer_id + idx),
                                                                                    last_layer_pos=last_layer_pos,
                                                                                    current_layer_id=layer_id)
                                                tmp_time = self.finish_time[layer_id + idx][updated_last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    if mix_mode==1 or mix_mode==4:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                                    elif mix_mode==2 or mix_mode==3:
                                        if self.Pipe_flag == 1 :
                                            compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                                        else :
                                            compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                                    self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                                    max_time[m] = compute_time
                                    
                    
                else:
                    cur_multiple = self.multiple[layer_id]
                    assert cur_multiple == 1, "Only the conv layer can be multipled"
                    if layer_dict['type'] == 'fc':
                        output_size = int(layer_dict['Outfeature'])
                        input_size = int(layer_dict['Infeature'])
                        self.layer_split.append([input_size])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['outputbit'])
                        self.layer_latency_initial()
                        indata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                        rdata = indata
                        if mix_mode==1:
                            temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                                    read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                                    indata=indata, rdata=rdata, inprecision=inputbit,
                                                                    PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                    default_inbuf_size=self.graph.max_inbuf_size,
                                                                    default_outbuf_size=self.graph.max_outbuf_size
                                                                    )
                        #linqiushi modified
                        if mix_mode==2:
                            temp_device_type=[]
                            temp_PE_num=[]
                            temp_xbar_size=[]
                            temp_pos=[]
                            for i in range(self.graph.layer_tileinfo[0]['tile_num_mix'][0]):
                                for j in range(self.graph.layer_tileinfo[0]['tile_num_mix'][1]):
                                    
                                    if self.graph.auto_layer_mapping==0:
                                        if self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j]=='no':
                                            pass
                                        elif int(self.graph.layer_tileinfo[0]['layer_mapping_mix'][i][j])==layer_id:
                                            
                                            temp_pos.append([i,j])
                                            temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                            temp_PE_num.append((self.graph.layer_tileinfo[0]['PE_num_mix'][i][j])**2)
                                            temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                                    else:
                                        if self.graph.mapping_result[i][j]==layer_id:
                                            temp_pos.append([i,j])
                                            temp_device_type.append(self.graph.layer_tileinfo[0]['device_type_mix'][i][j])
                                            temp_PE_num.append(self.graph.layer_tileinfo[layer_id]['max_PE'][i][j])
                                            temp_xbar_size.append(self.graph.layer_tileinfo[0]['xbar_size_mix'][i][j])
                                        
                            
                            temp_tile_latency_max=0
                            for i in range(len(temp_device_type)):
                                #max_PE,row,column待覆盖
                                temp_tile_latency0 = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    read_row=self.graph.layer_tileinfo[layer_id]['max_row_mix'][i],
                                                                read_column=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i],
                                                                    indata=0, rdata=0, inprecision=inputbit,
                                                                    PE_num=temp_PE_num[i],
                                                                    default_inbuf_size=self.graph.max_inbuf_size,
                                                                    default_outbuf_size=self.graph.max_outbuf_size,
                                                                    device_type=temp_device_type[i],
                                                                    xbar_size=[int(temp_xbar_size[i]),int(temp_xbar_size[i])],
                                                                    mix_mode=mix_mode
                                                                    )
                                if temp_tile_latency0.tile_latency>temp_tile_latency_max:
                                    temp_tile_latency_max=temp_tile_latency0.tile_latency
                                    temp_tile_latency = temp_tile_latency0
                                    
                        elif mix_mode==3:
                    
                            temp_tile_latency_max=0
                            for i in range(len(self.graph.layer_tileinfo[layer_id]['tile_max_row'])):
                                
                                temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                    read_row=self.graph.layer_tileinfo[layer_id]['tile_max_row'][i],
                                                                    read_column=self.graph.layer_tileinfo[layer_id]['tile_max_column'][i],
                                                                    indata=0, rdata=0, inprecision=inputbit,
                                                                    PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                    default_inbuf_size=self.graph.max_inbuf_size,
                                                                    default_outbuf_size=self.graph.max_outbuf_size,
                                                                    device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                                    xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i]),int(self.graph.layer_tileinfo[layer_id]['xbar_size'][i])],
                                                                    mix_mode=mix_mode
                                                                    )  
                                if temp_tile_latency.tile_latency>temp_tile_latency_max:
                                    temp_tile_latency_max=temp_tile_latency.tile_latency 
                        elif mix_mode==4:
                            temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                read_row=self.graph.layer_tileinfo[layer_id]['max_row'],
                                                                read_column=self.graph.layer_tileinfo[layer_id]['max_column'],
                                                                indata=0, rdata=0, inprecision=inputbit,
                                                                device_type=self.graph.layer_tileinfo[layer_id]['device_type'],
                                                                PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'],
                                                                default_inbuf_size=self.graph.max_inbuf_size,
                                                                default_outbuf_size=self.graph.max_outbuf_size,
                                                                xbar_size=[int(self.graph.layer_tileinfo[layer_id]['xbar_size']),int(self.graph.layer_tileinfo[layer_id]['xbar_size'])],
                                                                mix_mode=mix_mode
                                                                ) 
                        #assert 0
                        #linqiushi above
                        if mix_mode==3:
                            temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['tile_max_column'][0]*
                                                                                    outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                            temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                            if self.Booksim_Flag == 0 :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                            else :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                    (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['tile_max_column'][0] *
                                    self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                        elif mix_mode==2:
                    
                            MAX=0
                            for i in range(len(temp_pos)):
                                j=temp_pos[i][0]
                                k=temp_pos[i][1]
                                if MAX<self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]:
                                    MAX=self.graph.layer_tileinfo[layer_id]['max_column_mix'][i]*self.graph.layer_tileinfo[layer_id]['max_PE'][j][k]
                            temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (MAX*outputbit/8))
                            temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                            if self.Booksim_Flag == 0 :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                        (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth)
                            else :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                        (temp_tile_latency.digital_period +MAX * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                            
                        elif mix_mode==1 or mix_mode==4:
                            temp_tile_latency.outbuf.calculate_buf_read_latency(rdata = (self.graph.layer_tileinfo[layer_id]['max_column']*
                                                                                        outputbit*self.graph.layer_tileinfo[layer_id]['max_PE']/8))
                            temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf.buf_rlatency
                            if self.Booksim_Flag == 0 :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                            (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                            self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth)
                            else :
                                merge_time = temp_tile_latency.tile_buf_rlatency+self.graph.inLayer_distance[0][layer_id] * \
                                            (temp_tile_latency.digital_period +self.graph.layer_tileinfo[layer_id]['max_column'] *
                                            self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth) + self.Merge_Latency[layer_id] *  1000 / self.freq

                            
                        # Todo: update merge time (adder tree) and transfer data volume
                        if self.Booksim_Flag == 0 :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                        else :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                        temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                        max_prelayer_time = 0
                        #fc: no_pipeline
                        for idx in temp_Inputindex:
                            tmp_time = self.finish_time[layer_id+idx][-1]
                            if tmp_time > max_prelayer_time:
                                max_prelayer_time = tmp_time
                        begin_time = max_prelayer_time
                        if mix_mode==1 or mix_mode==4:
                            if self.Pipe_flag == 1 :
                                compute_time = max(temp_tile_latency.tile_latency, merge_time + transfer_time) + begin_time
                            else :
                                compute_time = temp_tile_latency.tile_latency + merge_time + transfer_time + begin_time 
                        elif mix_mode==2 or mix_mode==3:
                            if self.Pipe_flag == 1 :
                                compute_time = max(temp_tile_latency_max, merge_time + transfer_time) + begin_time
                            else :
                                compute_time = temp_tile_latency_max + merge_time + transfer_time + begin_time
                        
                        self.pipe_result_update(layer_type='fc', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time, output_size=output_size)
                    elif layer_dict['type'] == 'pooling':
                        self.layer_latency_initial()
                        output_size = list(map(int, layer_dict['Outputsize']))
                        input_size = list(map(int, layer_dict['Inputsize']))
                        self.layer_split.append([input_size[1]])
                        kernelsize = int(layer_dict['Kernelsize'])
                        stride = int(layer_dict['Stride'])
                        inputchannel = int(layer_dict['Inputchannel'])
                        outputchannel = int(layer_dict['Outputchannel'])
                        padding = int(layer_dict['Padding'])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['outputbit'])
                        temp_pooling_latency = pooling_latency_analysis(SimConfig_path=self.SimConfig_path,
                                                                        indata=0, rdata=0, outprecision = outputbit,
                                                                        default_inbuf_size = self.graph.max_inbuf_size,
                                                                        default_outbuf_size = self.graph.max_outbuf_size,
                                                                        default_inchannel = inputchannel, default_size = (kernelsize**2))
                        temp_pooling_latency.outbuf.calculate_buf_read_latency(rdata=(outputchannel*outputbit/8))
                        temp_pooling_latency.outbuf_rlatency = temp_pooling_latency.outbuf.buf_rlatency
                        merge_time = temp_pooling_latency.outbuf_rlatency
                        # Todo: update merge time of pooling tile
                        if self.Booksim_Flag == 0 :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                        else :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                        # Todo: update transfer data volume
                        self.pre_max_time = 0
                        for i in range(output_size[0]):
                            for j in range(output_size[1]):
                                last_layer_pos = (min(max(kernelsize - padding, 1) + stride * i, input_size[0]) - 1) * \
                                                 input_size[1] + min(max(kernelsize - padding, 1) + stride * j, input_size[1]) - 1
                                if (i==0) & (j==0):
                                    if mode == 0:
                                        indata = inputchannel * (input_size[1] * max(kernelsize-padding-1,0)+max(kernelsize-padding,0))*inputbit/8
                                    else:
                                        indata = inputchannel * (max(kernelsize-padding,0)**2)*inputbit/8
                                    rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata,rdata=rdata)
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    # the maximum time of the required input data (in all input layers)
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                    else:
                                        for idx in temp_Inputindex:
                                            tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_pooling_latency.pooling_latency , merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                    self.pre_max_time = compute_time
                                    self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_pooling_latency=temp_pooling_latency,merge_time=merge_time,transfer_time=transfer_time)
                                elif j==0:
                                    if mode == 0:
                                        indata = inputchannel * (input_size[1] * (stride - 1) + max(kernelsize - padding, 0)) * inputbit/8
                                    else:
                                        indata = inputchannel * stride * max(kernelsize - padding, 0) * inputbit / 8
                                    rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                    else:            
                                        for idx in temp_Inputindex:
                                            tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_pooling_latency.pooling_latency , merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                    self.pre_max_time = compute_time
                                    self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_pooling_latency=temp_pooling_latency,merge_time=merge_time, transfer_time=transfer_time)
                                else:
                                    if mode == 0:
                                        indata = inputchannel * stride * inputbit / 8
                                    else:
                                        indata = inputchannel * stride **2 * inputbit / 8
                                    rdata = stride * kernelsize * inputchannel * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)
                                    temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                    max_prelayer_time = 0
                                    #linqiushi modified
                                    if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                    else:
                                        for idx in temp_Inputindex:
                                            tmp_time = self.finish_time[layer_id + idx][last_layer_pos]
                                            if tmp_time > max_prelayer_time:
                                                max_prelayer_time = tmp_time
                                    #linqiushi above
                                    begin_time = max(max_prelayer_time, self.pre_max_time)
                                    if self.Pipe_flag == 1 :
                                        compute_time = max(temp_pooling_latency.pooling_latency , merge_time + transfer_time) + begin_time
                                    else :
                                        compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                    self.pre_max_time = compute_time
                                    self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                            temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)
                    elif layer_dict['type'] == 'element_sum':
                        self.layer_latency_initial()
                        Inputindex_list = list(map(int, layer_dict['Inputindex']))
                        assert len(Inputindex_list) > 1, "the number of element_sum's previous layers must > 1"
                        idx = 0
                        previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[0]][0][0]
                        while previous_layer_dict['type'] == 'element_sum':
                            
                            idx = idx + 1
                            previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[idx]][0][0]
                        output_size = list(map(int, previous_layer_dict['Outputsize']))
                        input_size = list(map(int, previous_layer_dict['Outputsize']))
                        
                        self.layer_split.append([input_size[1]])
                        kernelsize = int(previous_layer_dict['Kernelsize'])
                        inputchannel = int(previous_layer_dict['Outputchannel'])
                        outputchannel = int(previous_layer_dict['Outputchannel'])
                        inputbit = int(previous_layer_dict['outputbit'])
                        outputbit = int(previous_layer_dict['outputbit'])
                        merge_time = 0
                        if self.Booksim_Flag == 0 :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                        else :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                        global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.global_buf_size)
                        global_buf.calculate_buf_read_latency(rdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                        global_buf.calculate_buf_write_latency(wdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                        self.pre_max_time = 0
                        for i in range(output_size[0]):
                            for j in range(output_size[1]):
                                max_prelayer_time = 0
                                # the maximum time of the required input data (in all input layers)
                                #linqiushi modified
                                if rewrite_mode==1 and whether_rewrite==1:
                                        for idx in temp_Inputindex:
                                            tmp_time = max(self.finish_time[layer_id + idx])
                                            if tmp_time>max_prelayer_time:
                                                max_prelayer_time=tmp_time
                                            
                                else:
                                    for idx in Inputindex_list:
                                        tmp_time = self.finish_time[layer_id+idx][i*input_size[1]+j]
                                        if tmp_time > max_prelayer_time:
                                            max_prelayer_time = tmp_time
                                #linqiushi above
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                
                                compute_time = 10+merge_time+transfer_time+begin_time+global_buf.buf_rlatency+global_buf.buf_wlatency
                                
                                
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='element_sum', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                        global_buf=global_buf, merge_time=merge_time, transfer_time=transfer_time)
                    elif layer_dict['type'] == 'element_multiply':
                        self.layer_latency_initial()
                        Inputindex_list = list(map(int, layer_dict['Inputindex']))
                        assert len(Inputindex_list) > 1, "the number of element_multiply's previous layers must > 1"
                        idx = 0
                        max_previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[0]][0][0]
                        #find the inputlayer with the max input size
                        for i in range(len(Inputindex_list)):
                            if self.NetStruct[layer_id + Inputindex_list[i]][0][0]['Outputsize']>max_previous_layer_dict['Outputsize']:
                                max_previous_layer_dict=self.NetStruct[layer_id + Inputindex_list[i]][0][0]
    
                        while max_previous_layer_dict['type'] == 'element_multiply':
                            idx = idx + 1
                            max_previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[idx]][0][0]
                        output_size = list(map(int, max_previous_layer_dict['Outputsize']))
                        input_size = list(map(int, max_previous_layer_dict['Outputsize']))
                        self.layer_split.append([input_size[1]])
                        inputchannel = int(max_previous_layer_dict['Outputchannel'])
                        outputchannel = int(max_previous_layer_dict['Outputchannel'])
                        inputbit = int(max_previous_layer_dict['outputbit'])
                        outputbit = int(max_previous_layer_dict['outputbit'])
                        merge_time = 0
                        if self.Booksim_Flag == 0 :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth)
                        else :
                            transfer_time = self.graph.transLayer_distance[0][layer_id] * (outputchannel * outputbit / self.inter_tile_bandwidth) + (self.Trans_Latency[layer_id] ) *  1000 / self.freq

                        global_buf = buffer(SimConfig_path=self.SimConfig_path,buf_level=2,default_buf_size=self.graph.global_buf_size)
                        global_buf.calculate_buf_read_latency(rdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                        global_buf.calculate_buf_write_latency(wdata=(len(Inputindex_list)*inputbit*inputchannel/8))
                        self.pre_max_time = 0
                        for i in range(output_size[0]):
                            for j in range(output_size[1]):
                                max_prelayer_time = 0
                                # the maximum time of the required input data (in all input layers)
                                #linqiushi modified
                                if rewrite_mode==1 and whether_rewrite==1:
                                    for idx in temp_Inputindex:
                                        tmp_time = max(self.finish_time[layer_id + idx])
                                        if tmp_time>max_prelayer_time:
                                            max_prelayer_time=tmp_time
                                else:
                                    for idx in Inputindex_list:
                                        tmp_time = self.finish_time[layer_id+idx][i*input_size[1]+j]
                                        if tmp_time > max_prelayer_time:
                                            max_prelayer_time = tmp_time
                                #linqiushi above
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                compute_time = 10+merge_time+transfer_time+begin_time+global_buf.buf_rlatency+global_buf.buf_wlatency
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='element_multiply', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id,
                                                        global_buf=global_buf, merge_time=merge_time, transfer_time=transfer_time)
            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1] - self.compute_interval[layer_id][l][0])
            self.occupancy.append(temp_runtime / (max(self.finish_time[layer_id]) - min(self.begin_time[layer_id])))
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_inter_tile_latency.append(sum(self.inter_tile_latency[layer_id]))
            self.total_intra_tile_latency.append(sum(self.intra_tile_latency[layer_id]))
            self.total_tile_merge_latency.append(sum(self.tile_merge_latency[layer_id]))
            self.total_tile_transfer_latency.append(sum(self.tile_transfer_latency[layer_id]))
            self.total_iReg_latency.append(sum(self.iReg_latency[layer_id]))
            self.total_oReg_latency.append(sum(self.oReg_latency[layer_id]))
            self.total_input_demux_latency.append(sum(self.input_demux_latency[layer_id]))
            self.total_output_mux_latency.append(sum(self.output_mux_latency[layer_id]))
            self.total_shiftreg_latency.append(sum(self.shiftreg_latency[layer_id]))
            self.total_adder_latency.append(sum(self.adder_latency[layer_id]))
            self.total_jointmodule_latency.append(sum(self.jointmodule_latency[layer_id]))
            self.total_pooling_latency.append(sum(self.pooling_latency[layer_id]))
            self.total_buffer_r_latency.append(sum(self.buffer_r_latency[layer_id]))
            self.total_buffer_w_latency.append(sum(self.buffer_w_latency[layer_id]))

    def booksim(self):
        filename = '../booksim2/runfiles/nnmeshconfig'
        try:
            with open(filename, 'r') as file:
                content = file.readlines()
        except FileNotFoundError:
            print(f"The file {filename} was not found.")
            exit()
        new_content = []
        k_value = self.tile_num[0]
        for line in content:
            if 'k  =' in line:
                new_line = f'k  = {k_value};\n'
                new_content.append(new_line)
            else:
                new_content.append(line)
        try:
            with open(filename, 'w') as file:
                file.writelines(new_content)
            print(f'The value of k has been updated to {k_value}')
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
        program_path = '../booksim2/src/booksim'
        args = ['../booksim2/runfiles/nnmeshconfig']

        try:
            process = subprocess.Popen([program_path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if stdout:
                print("程序输出：")
                print(stdout)
            if stderr:
                print("程序错误输出：")
                print(stderr)
            if process.returncode != 0:
                print(f"程序退出状态码：{process.returncode}")
            else:
                print("程序成功执行完成。")
        except FileNotFoundError:
            print(f"未找到程序: {program_path}")
        except Exception as e:
            print(f"执行程序时发生错误: {e}")
        print("继续执行Python代码...")
        for line in stdout.splitlines():
            line = line.strip()
            if "Total Area" in line:
                area = line.split(':')[1].strip()
                NoC_area = float(area)
                print(f"Final Total Area: {NoC_area} um^2\n")
            if "Total Power" in line:
                power = line.split(':')[1].strip()
                NoC_power = float(power)/1000
                print(f"Final Total Power: {NoC_power} W\n")

        with open('../booksim2/runfiles/layerlatency_table.txt') as file:
                for line in file:
                    line = line.strip()
                    if "MergeLatency for Layer" in line:
                        latency = line.split(':')[1].strip()
                        self.Merge_Latency.append(float(latency))
                    if "TransLatency between Layer" in line:
                        latency = line.split(':')[1].strip()
                        self.Trans_Latency.append(float(latency))

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "alexnet_params.pth")

    __TestInterface = TrainTestInterface('alexnet', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path)
    structure_file = __TestInterface.get_structure()
    test = Model_latency(structure_file, test_SimConfig_path)

    tile = 0
    test.calculate_model_latency(mode=2)
    test.model_latency_output()
