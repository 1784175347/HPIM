#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import argparse
import numpy as np
import torch
import collections
import configparser
import time
from importlib import import_module
from MNSIM.Interface.interface import *
from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Mapping_Model.Behavior_mapping import behavior_mapping
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy
from MNSIM.mixing.mixtile import mixtile
from IPython import embed

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    start_time = time.time()
    home_path = os.getcwd()
    # print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig_noNoC.ini")
    #linqiushi modified
    mix_tile_path=os.path.join(home_path,"mix_tileinfo_cifar10_resnet18.ini")
    weights_file_path = os.path.join(home_path, "cifar10_resnet18_params.pth")
    # print(SimConfig_path)
    parser = argparse.ArgumentParser(description='MNSIM example')
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
        help="Whether delete the unnecessary files automatically")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
        help="Hardware description file location & name, default:/MNSIM_Python/SimConfig.ini")
    parser.add_argument("-Weights", "--weights", default=weights_file_path,
        help="NN model weights file location & name, default:/MNSIM_Python/cifar10_vgg8_params.pth")
    parser.add_argument("-Dataset", "--dataset", default='cifar10',
        help="Dataset description (name), default: cifar10")
    parser.add_argument("-NN", "--NN", default='resnet18',
        help="NN model description (name), default: vgg8")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
        help="Disable hardware modeling, default: false")
    parser.add_argument("-DisAccu", "--disable_accuracy_simulation", action='store_true', default=False,
        help="Disable accuracy simulation, default: false")
    parser.add_argument("-SAF", "--enable_SAF", action='store_true', default=True,
        help="Enable simulate SAF, default: false")
    parser.add_argument("-Var", "--enable_variation", action='store_true', default=False,
        help="Enable simulate variation, default: false")
    parser.add_argument("-Rratio", "--enable_R_ratio", action='store_true', default=False,
        help="Enable simulate the effect of R ratio, default: false")
    parser.add_argument("-FixRange", "--enable_fixed_Qrange", action='store_true', default=False,
        help="Enable fixed quantization range (max value), default: false")
    parser.add_argument("-DisPipe", "--disable_inner_pipeline", action='store_true', default=False,
        help="Disable inner layer pipeline in latency modeling, default: false")
    parser.add_argument("-D", "--device", default=0,
        help="Determine hardware device (CPU or GPU-id) for simulation, default: CPU")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
        help="Disable module simulation results output, default: false")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
        help="Disable layer-wise simulation results output, default: false")
    #linqiushi modified
    parser.add_argument("-mix_mode","--mix_mode", default=2, type=int, help="1:no mix, 2:mix among tile, 3:mix among tile(easy)")
    parser.add_argument("-mix_tile", "--mix_tileinfo", default=mix_tile_path, help="the exact tile info used in the mixing")
    #linqiushi above
    args = parser.parse_args()
    print("Hardware description file location:", args.hardware_description)
    print("Software model file location:", args.weights)
    print("Whether perform hardware simulation:", not (args.disable_hardware_modeling))
    print("Whether perform accuracy simulation:", not (args.disable_accuracy_simulation))
    print("Whether consider SAFs:", args.enable_SAF)
    print("Whether consider variations:", args.enable_variation)
    if args.enable_fixed_Qrange:
        print("Quantization range: fixed range (depends on the maximum value)")
    else:
        print("Quantization range: dynamic range (depends on the data distribution)")
    #linqiushi modified
    if args.mix_mode==1:
        print("RRAM_SRAM mix mode: no mix")
    elif args.mix_mode==2:
        print("RRAM_SRAM mix mode: mix among tile")
        
    elif args.mix_mode==3:
        print("RRAM_SRAM mix mode: mix among tile(easy)")
    elif args.mix_mode==4:
        print("RRAM_SRAM mix mode: mix among tile(layer same)")
    else:
        assert 0, f'mix mode should in {1,2,3,4}'
    #linqiushi above
    
    mapping_start_time = time.time()

    #linqiiushi modified
    #add parser:mix_mode mix_ratio
    #cifar10/cifar100/Imagenet
    __TestInterface = TrainTestInterface(network_module=args.NN, dataset_module=f"MNSIM.Interface.{args.dataset}",  
        SimConfig_path=args.hardware_description, weights_file=args.weights, device=args.device)
   
    structure_file = __TestInterface.get_structure()
    #linqiushi modified
    if args.mix_mode==2 or args.mix_mode==4:
        mix_tile=mixtile(args.mix_tileinfo)
    else:
        mix_tile=None
    TCG_mapping = TCG(structure_file, args.hardware_description,mix_mode=args.mix_mode,mix_tile=mix_tile)

    # print(TCG_mapping.max_inbuf_size)
    # print(TCG_mapping.max_outbuf_size)
    
    #linqiushi modified
    if args.mix_mode==2 :
        print("start to mix among tiles")
        TCG_mapping=mix_tile.read_tileinfo(SimConfig_path=SimConfig_path,TCG_mapping=TCG_mapping,mix_mode=args.mix_mode)
        
        #assert 0
    
    mapping_end_time = time.time()
    if not (args.disable_hardware_modeling):
        hardware_modeling_start_time = time.time()
        __area = Model_area(NetStruct=structure_file, SimConfig_path=args.hardware_description, TCG_mapping=TCG_mapping,mix_mode=args.mix_mode)
        
        print("========================Area Results=================================")
        area_MNSIM=__area.model_area_output(not (args.disable_module_output), not (args.disable_layer_output))
        __latency = Model_latency(NetStruct=structure_file, SimConfig_path=args.hardware_description, TCG_mapping=TCG_mapping,mix_mode=args.mix_mode,mix_tile=mix_tile,area_MNSIM=area_MNSIM)
        if not (args.disable_inner_pipeline):
            if TCG_mapping.rewrite_mode==2:
                __latency.calculate_model_latency_LLM(mode=1,mix_mode=args.mix_mode)
            else:
                __latency.calculate_model_latency(mode=1,mix_mode=args.mix_mode)
                #__latency.calculate_model_latency_nopipe()
            
        else:
            __latency.calculate_model_latency_nopipe()
        hardware_modeling_end_time = time.time()
        
        
        print("========================Latency Results=================================")
        if TCG_mapping.rewrite_mode==2:
            __latency.model_latency_output_LLM(not (args.disable_module_output), not (args.disable_layer_output),TCG_mapping.rewrite_layer_list)
        else:
            __latency.model_latency_output(not (args.disable_module_output), not (args.disable_layer_output))
        
        __power = Model_inference_power(NetStruct=structure_file, SimConfig_path=args.hardware_description,
                                        TCG_mapping=TCG_mapping,mix_mode=args.mix_mode)
        print("========================Power Results=================================")
        if TCG_mapping.rewrite_mode==2:
            __power.model_power_output_rewrite(not (args.disable_module_output), not (args.disable_layer_output))
        else:
            __power.model_power_output(not (args.disable_module_output), not (args.disable_layer_output))
        __energy = Model_energy(NetStruct=structure_file, SimConfig_path=args.hardware_description,
                                TCG_mapping=TCG_mapping,
                                model_latency=__latency, model_power=__power)
        if args.mix_mode!=2 or mix_tile.auto_layer_mapping!=0:
            print("========================Energy Results=================================")
            __energy.model_energy_output(not (args.disable_module_output), not (args.disable_layer_output))
    '''
    if not (args.disable_accuracy_simulation):
        print("======================================")
        print("Accuracy simulation will take a few minutes on GPU")
        accuracy_modeling_start_time = time.time()
        weight = __TestInterface.get_net_bits()
        weight_2 = weight_update(args.hardware_description, weight,
                                 is_Variation=args.enable_variation, is_SAF=args.enable_SAF, is_Rratio=args.enable_R_ratio)
        if not (args.enable_fixed_Qrange):
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='SCALE'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='SCALE'))
        else:
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='FIX'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='FIX'))
        accuracy_modeling_end_time = time.time()
    '''
    mapping_time = mapping_end_time - mapping_start_time
    
    '''
    print("Mapping time:", mapping_time)
    
    if not (args.disable_hardware_modeling):
        hardware_modeling_time = hardware_modeling_end_time - hardware_modeling_start_time
        print("Hardware modeling time:", hardware_modeling_time)
    else:
        hardware_modeling_time = 0
    if not (args.disable_accuracy_simulation):
        accuracy_modeling_time = accuracy_modeling_end_time - accuracy_modeling_start_time
        print("Accuracy modeling time:", accuracy_modeling_time)
    else:
        accuracy_modeling_time = 0
    print("Total simulation time:", mapping_time+hardware_modeling_time+accuracy_modeling_time)
    '''
    # print(structure_file)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"程序运行时间：{elapsed_time}秒")

if __name__ == '__main__':
    # Data_clean()
    main()
