#include <string>
#include <iostream>
#include <random> 
#include <stdio.h>
#include <stdlib.h>
#include "CNN.hpp"
#include "misc_utils.hpp"

CNNNetwork::CNNNetwork(const Configuration & config) {
    _layer = config.GetInt("cnn_layer_num");
    _bitwidth = config.GetInt("bitwidth");
    _FPS = config.GetInt("FPS");
    _channelbitwidth = config.GetInt("channelbitwidth");
    _freq = config.GetInt("freq");
    _latency_filename = config.GetStr("latency_table");
    _injection_filename = config.GetStr("fixed_injection_table");
    _dest_filename = config.GetStr("fixed_dest_table");
    _layerlatency_filename = config.GetStr("layerlatency_table");
    
    
    _k = config.GetInt("k");
    _n = config.GetInt("n");
    _netnodes = powi( _k, _n);
    
    _distribution_pattern = config.GetStr("distribution_pattern");
    _routing_planning_pattern = config.GetStr("routing_planning_pattern");
    
    _netCIMHeight = config.GetIntArray("netCIMHeight"); 
    if(_netCIMHeight.empty()) {
        _netCIMHeight.push_back(config.GetInt("netCIMHeight"));
    } 

    _netCIMWidth = config.GetIntArray("netCIMWidth"); 
    if(_netCIMWidth.empty()) {
        _netCIMWidth.push_back(config.GetInt("netCIMWidth"));
    }

    if (_layer == 0) return;

    _inputChannels = config.GetIntArray("inputChannels"); 
    if(_inputChannels.empty()) {
        _inputChannels.push_back(config.GetInt("inputChannels"));
    } 

    _outputChannels = config.GetIntArray("outputChannels"); 
    if(_outputChannels.empty()) {
        _outputChannels.push_back(config.GetInt("outputChannels"));
    }

    _kernelHeight = config.GetIntArray("kernelHeight"); 
    if(_kernelHeight.empty()) {
        _kernelHeight.push_back(config.GetInt("kernelHeight"));
    }

    _kernelWidth = config.GetIntArray("kernelWidth"); 
    if(_kernelWidth.empty()) {
        _kernelWidth.push_back(config.GetInt("kernelWidth"));
    } 

    _stride = config.GetIntArray("stride"); 
    if(_stride.empty()) {
        _stride.push_back(config.GetInt("stride"));
    }

    _padding = config.GetIntArray("padding"); 
    if(_padding.empty()) {
        _padding.push_back(config.GetInt("padding"));
    }   

    _hasPooling = config.GetIntArray("hasPooling"); 
    if(_hasPooling.empty()) {
        _hasPooling.push_back(config.GetInt("hasPooling"));
    } 

    _poolSize = config.GetIntArray("poolSize"); 
    if(_poolSize.empty()) {
        _poolSize.push_back(config.GetInt("poolSize"));
    }

    _poolStride = config.GetIntArray("poolStride"); 
    if(_poolStride.empty()) {
        _poolStride.push_back(config.GetInt("poolStride"));
    }

    _isDense = config.GetIntArray("isDense"); 
    if(_isDense.empty()) {
        _isDense.push_back(config.GetInt("isDense"));
    } 

    _inputSizeHeight = config.GetIntArray("inputSizeHeight"); 
    if(_inputSizeHeight.empty()) {
        _inputSizeHeight.push_back(config.GetInt("inputSizeHeight"));
    }

    _inputSizeWidth = config.GetIntArray("inputSizeWidth"); 
    if(_inputSizeWidth.empty()) {
        _inputSizeWidth.push_back(config.GetInt("inputSizeWidth"));
    }  

    for(int i = 0; i < _layer; ++i){
        ConvLayerScale scale(_inputChannels[i],_outputChannels[i],_kernelHeight[i],_kernelWidth[i],
                             _stride[i],_padding[i],bool(_hasPooling[i]),_poolSize[i],
                             _poolStride[i],_inputSizeHeight[i],_inputSizeWidth[i],bool(_isDense[i]));
        addLayerScale(scale);
        _outputSizeHeight.push_back(scale.outputSize.first);
        _outputSizeWidth.push_back(scale.outputSize.second);
    }

    printNetworkScale();
    GetInjectionrate(_netnodes);

}

void CNNNetwork::checkCompatibilityBetween(size_t currentLayerIndex, const ConvLayerScale& newLayer) {
        assert(currentLayerIndex <= layers.size() && "Invalid layer index for compatibility check.");

        // 如果网络中已有层，检查新层与上一层的兼容性
        if (currentLayerIndex > 0) {
            const auto& previousLayer = layers[currentLayerIndex - 1];
            // 如果新层是全连接层，确保上一层的摊平后的大小与新层的输入通道数匹配
            if (newLayer.isDense){
                int flattenedSize = previousLayer.outputChannels;
                if (!previousLayer.isDense) {
                    flattenedSize *= previousLayer.outputSize.first * previousLayer.outputSize.second;
                }
                assert(flattenedSize == newLayer.inputChannels &&
                       "Flattened size of the previous layer must match the input channel number of the dense layer.");
            }
            else{
                int flattenedSize = previousLayer.outputChannels;
                if (previousLayer.isDense) {
                    flattenedSize /= newLayer.inputSize.first * newLayer.inputSize.second;
                }
                // 检查通道数是否匹配
                assert(flattenedSize == newLayer.inputChannels &&
                       "Output channel number of the previous layer must match the input channel number of the new layer.");
    
                // 检查空间尺寸是否匹配
                assert((previousLayer.isDense || (previousLayer.outputSize == newLayer.inputSize)) &&
                       "Spatial dimensions of the previous layer must match the input size of the new layer.");
            }
            
        }
    }


void CNNNetwork::addLayerScale(const ConvLayerScale& scale) {
        // 在添加新层之前，检查与上一层的兼容性
        checkCompatibilityBetween(layers.size(), scale);

        // 添加新层
        layers.push_back(scale);
        
    }

void CNNNetwork::printNetworkScale() const {
    std::cout << "Layers: " << layers.size() << ",Bitwidth: " << _bitwidth << ",FPS: " << _FPS<< ",Channelbitwidth: " << _channelbitwidth<< ",Freq: " << _freq<< std::endl;
    for (const auto& layer : layers) {
        std::cout << "  - Layer Type: " << (layer.isDense ? "Dense" : "Convolutional")
                  << ", Input Channels: " << layer.inputChannels
                  << ", Output Channels: " << layer.outputChannels
                  << ", Kernel Size: (" << layer.kernelHeight << "x" << layer.kernelWidth << ")"
                  << ", Stride: " << layer.stride
                  << ", Padding: " << layer.padding
                  << ", Has Pooling: " << (layer.hasPooling ? "Yes" : "No")
                  << ", Pool Size: " << (layer.hasPooling ? std::to_string(layer.poolSize) : "N/A")
                  << ", Pool Stride: " << (layer.hasPooling ? std::to_string(layer.poolStride) : "N/A")
                  <<", Input Size: (" << layer.inputSize.first << "x" << layer.inputSize.second << ")"
                  <<", Output Size: (" <<layer.outputSize.first << "x" << layer.outputSize.second << ")"
                  << std::endl;
    }
    std::cout << "Netnodes: " << _netnodes<< std::endl;//<< ",Bitwidth: " << _bitwidth << ",FPS: " << _FPS<< std::endl;
    for (int i = 0;i < _netnodes; i++){
        std::cout<< "  - CIM " << i<<" Size: (" << _netCIMHeight[i] << "x" << _netCIMWidth[i] << ")"<<endl;
    }
}

void CNNNetwork::GetInjectionrate(size_t count) {
    
    Dest.resize(count);
    Layer_net.resize(_layer);
    Layer_net_num.resize(count,0);
    Net_layer.resize(count,0);
    Distribution.resize(count,0);

    int restnet = count;


    if (_distribution_pattern == "sequential"){
        for (size_t i=0;i<count;i++)
        {
            Distribution[i]=i;
        }
    }
    else if (_distribution_pattern == "S_shaped"){
        for (size_t i=0;i<count;i++)
        {
            if(i/_k%2 == 0)
            Distribution[i]=i;
            else
            Distribution[i]=_k - i%_k - 1 + i/_k*_k;
        }
    }
    else if (_distribution_pattern == "diagonal_S_shaped"){
        int node_conut=0;
        for (int i=0;i<=2*(_k-1);i++)
        {
            if (i%2 == 1)
            {
                for (int j=0;j<=i;j++)
                {
                    if ((j>=_k) || (i-j>=_k)) continue;
                    Distribution[node_conut]=i - j + j*_k;
                    node_conut++;
                }
            }
            else
            {
                for (int j=i;j>=0;j--)
                {
                    if ((j>=_k) || (i-j>=_k)) continue;
                    Distribution[node_conut]=i - j + j*_k;
                    node_conut++;
                }
            }
        }
    }else {
    cout << "Invalid Distribution pattern: " << _distribution_pattern << endl;
    exit(-1);
  }

    for (int i = 0;i < _layer; i++){
        
        int KernelperNet;
        int usedNet;
        KernelperNet = _netCIMHeight[0]/(layers[i].inputChannels * layers[i].kernelHeight * layers[i].kernelWidth);
        if (KernelperNet >= layers[i].outputChannels)
        {
            Layer_net[i].push_back(Distribution[count - restnet]);
            Layer_net_num[i] = Layer_net_num[i] + 1;
            Net_layer[Distribution[count - restnet]] = i + 1;
            restnet = restnet - 1;
            assert(restnet >= 0 && "Weight Overflow");
            continue;
        }
        else
        {
            int j = KernelperNet;
            for (;j>0;j--) if (layers[i].outputChannels % j == 0)  break;
            usedNet = layers[i].outputChannels / j;
            for (int k = 0;k < usedNet; k++)
            {
                Layer_net[i].push_back(Distribution[count - restnet]);
                Layer_net_num[i] = Layer_net_num[i] + 1;
                Net_layer[Distribution[count - restnet]] = i + 1;
                restnet = restnet - 1;
                assert(restnet >= 0 && "Weight Overflow");
            }
        }
    }     


    for (int i = 0;i < _netnodes; i++){
        int layer_id = Net_layer[i];
        if (layer_id == 0 || layer_id == _layer)
        {
            Dest[i].push_back(0);
        }
        else
        {
            Dest[i].push_back(Layer_net[layer_id].size());
            for(int j = 0; j < Layer_net_num[layer_id]; j++)
            {
                Dest[i].push_back(Layer_net[layer_id][j]);
            }
        }
    } 


    if (_routing_planning_pattern == "parallel"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            double Injectionrate_layer = double(_outputChannels[layer_id] * _outputSizeHeight[layer_id] * _outputSizeWidth[layer_id] * _bitwidth * _FPS) / (Layer_net_num[layer_id-1] *_channelbitwidth * _freq);
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                Injectionrate.push_back({double(i),double(Layer_net[layer_id][j]),Injectionrate_layer});
            }
        }   
    }
    else if (_routing_planning_pattern == "serial"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            double Injectionrate_layer = double(_outputChannels[layer_id] * _outputSizeHeight[layer_id] * _outputSizeWidth[layer_id] * _bitwidth * _FPS) / (Layer_net_num[layer_id-1] *_channelbitwidth * _freq);
            int M_distance_0,M_distance_1;
            int fisrt_node,last_node;
            fisrt_node = Layer_net[layer_id][0];
            last_node = Layer_net[layer_id][Layer_net_num[layer_id]-1];
            M_distance_0 = abs(fisrt_node/_k - i/_k) + abs(fisrt_node%_k - i%_k);
            M_distance_1= abs(last_node/_k - i/_k) + abs(last_node%_k - i%_k);
            if (M_distance_0 <= M_distance_1)
            {
                int node_now = i;
                for (int j=0;j<Layer_net_num[layer_id];j++)
                {
                    Injectionrate.push_back({double(node_now),double(Layer_net[layer_id][j]),Injectionrate_layer});
                    node_now = Layer_net[layer_id][j];
                }
            }
            else
            {
                int node_now = i;
                for (int j=Layer_net_num[layer_id]-1;j>=0;j--)
                {
                    Injectionrate.push_back({double(node_now),double(Layer_net[layer_id][j]),Injectionrate_layer});
                    node_now = Layer_net[layer_id][j];
                }
            }
        } 
    }
    else if (_routing_planning_pattern == "BFS"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            double Injectionrate_layer = double(_outputChannels[layer_id] * _outputSizeHeight[layer_id] * _outputSizeWidth[layer_id] * _bitwidth * _FPS) / (Layer_net_num[layer_id-1] *_channelbitwidth * _freq);
            std::vector<int> M_distance;
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                int distance = abs(Layer_net[layer_id][j]/_k - i/_k) + abs(Layer_net[layer_id][j]%_k - i%_k);
                M_distance.push_back(distance);
            }
            int max_value = M_distance[0];
            
            for (int value : M_distance) {
                if (value > max_value) {
                    max_value = value;
                }
            }
            std::vector<int> M_distance_num(max_value,0);
            std::vector<vector<int>> M_distance_name(max_value);
            //cout<<"nodes == "<<i<<endl;
            for (int j=1;j<=max_value;j++)
            {
                for (int k=0;k<Layer_net_num[layer_id];k++) {
                    int value = M_distance[k];
                    if (value == j) {
                        M_distance_num[j-1] += 1;
                        M_distance_name[j-1].push_back(Layer_net[layer_id][k]);
                    }
                }
                /*cout<<"M_distance == "<<j<<endl;
                for (int k=0;k<M_distance_num[j-1];k++) {
                    cout<<M_distance_name[j-1][k]<<endl;
                }
                */
            }

            for (int j=1;j<=max_value;j++)
            {
                if (M_distance_num[j-1] == 0) continue;
                for (int k=0;k<M_distance_num[j-1];k++)
                {
                    bool ifbuildpath = false;
                    for (int m=j-1;m>0;m--)
                    {
                        for (int n=0;n<M_distance_num[m-1];n++)
                        {
                            int distance = abs(M_distance_name[m-1][n]/_k - M_distance_name[j-1][k]/_k) + abs(M_distance_name[m-1][n]%_k - M_distance_name[j-1][k]%_k);
                            if (distance == j-m)
                            {
                                ifbuildpath = true;
                                Injectionrate.push_back({double(M_distance_name[m-1][n]),double(M_distance_name[j-1][k]),Injectionrate_layer});
                                break;
                            }
                        }
                        if (ifbuildpath) break;
                    } 
                    if (!ifbuildpath) Injectionrate.push_back({double(i),double(M_distance_name[j-1][k]),Injectionrate_layer});
                }  
                
                
            }
            /*
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                Injectionrate.push_back({double(i),double(Layer_net[layer_id][j]),Injectionrate_layer});
            }
            */
        } 
    }else {
    cout << "Invalid Routing Planning pattern: " << _routing_planning_pattern << endl;
    exit(-1);
  }


    //保存注入率
    std::string filename = _injection_filename;
    if (filename == "none") return ;
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return ; 
    }


    for(size_t i = 0; i < Injectionrate.size(); i++){
        outfile <<  Injectionrate[i][0]<< " "<< Injectionrate[i][1]<<  " "<< Injectionrate[i][2]<< std::endl;
    }
    outfile.close();
    filename = _dest_filename;
    if (filename == "none") return ;
    outfile.open(filename);

    if (!outfile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return ; 
    }


    for(int i = 0; i < _netnodes; i++){
        outfile <<  i<< " "<< Dest[i][0];
        for(size_t j = 1; j < Dest[i].size(); j++)
        {
            outfile << " "<< Dest[i][j];
        } 
        outfile << std::endl;
    }
    outfile.close();

    return ;
}
std::map<int, std::map<int, double>> fixedLatencyTable;
void CNNNetwork::GetLatency(){
    
    Latency_max.resize(_layer,0.0);

    std::string filename = _latency_filename;
    if (filename == "none") return;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    int src, dest;
    double latency;
    while (infile >> src >> dest >> latency) {
                fixedLatencyTable[src][dest] = latency;
    }

    
    
    if (_routing_planning_pattern == "parallel"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            double latency_parallel;
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                latency_parallel = fixedLatencyTable[i][Layer_net[layer_id][j]];
                if (latency_parallel > Latency_max[layer_id-1]) Latency_max[layer_id-1] = latency_parallel;
            }
        }   
    }
    else if (_routing_planning_pattern == "serial"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            double latency_serial = 0.0;
            int M_distance_0,M_distance_1;
            int fisrt_node,last_node;
            fisrt_node = Layer_net[layer_id][0];
            last_node = Layer_net[layer_id][Layer_net_num[layer_id]-1];
            M_distance_0 = abs(fisrt_node/_k - i/_k) + abs(fisrt_node%_k - i%_k);
            M_distance_1= abs(last_node/_k - i/_k) + abs(last_node%_k - i%_k);
            if (M_distance_0 <= M_distance_1)
            {
                int node_now = i;
                for (int j=0;j<Layer_net_num[layer_id];j++)
                {
                    latency_serial += fixedLatencyTable[node_now][Layer_net[layer_id][j]];
                    node_now = Layer_net[layer_id][j];
                }
            }
            else
            {
                int node_now = i;
                for (int j=Layer_net_num[layer_id]-1;j>=0;j--)
                {
                    latency_serial += fixedLatencyTable[node_now][Layer_net[layer_id][j]];
                    node_now = Layer_net[layer_id][j];
                }
            }
            if (latency_serial > Latency_max[layer_id-1]) Latency_max[layer_id-1] = latency_serial;
        } 
    }
    else if (_routing_planning_pattern == "BFS"){
        for (int i = 0;i < _netnodes; i++){
            int layer_id = Net_layer[i];
            if (layer_id == 0 || layer_id == _layer) continue;
            std::vector<double> latency_BFS(_netnodes,0.0);
            //double Injectionrate_layer = double(_outputChannels[layer_id] * _outputSizeHeight[layer_id] * _outputSizeWidth[layer_id] * _bitwidth * _FPS) / (Layer_net_num[layer_id-1] *_channelbitwidth * _freq);
            std::vector<int> M_distance;
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                int distance = abs(Layer_net[layer_id][j]/_k - i/_k) + abs(Layer_net[layer_id][j]%_k - i%_k);
                M_distance.push_back(distance);
            }
            int max_value = M_distance[0];
            
            for (int value : M_distance) {
                if (value > max_value) {
                    max_value = value;
                }
            }
            std::vector<int> M_distance_num(max_value,0);
            std::vector<vector<int>> M_distance_name(max_value);
            //cout<<"nodes == "<<i<<endl;
            for (int j=1;j<=max_value;j++)
            {
                for (int k=0;k<Layer_net_num[layer_id];k++) {
                    int value = M_distance[k];
                    if (value == j) {
                        M_distance_num[j-1] += 1;
                        M_distance_name[j-1].push_back(Layer_net[layer_id][k]);
                    }
                }
                /*cout<<"M_distance == "<<j<<endl;
                for (int k=0;k<M_distance_num[j-1];k++) {
                    cout<<M_distance_name[j-1][k]<<endl;
                }
                */
            }

            for (int j=1;j<=max_value;j++)
            {
                if (M_distance_num[j-1] == 0) continue;
                for (int k=0;k<M_distance_num[j-1];k++)
                {
                    bool ifbuildpath = false;
                    for (int m=j-1;m>0;m--)
                    {
                        for (int n=0;n<M_distance_num[m-1];n++)
                        {
                            int distance = abs(M_distance_name[m-1][n]/_k - M_distance_name[j-1][k]/_k) + abs(M_distance_name[m-1][n]%_k - M_distance_name[j-1][k]%_k);
                            if (distance == j-m)
                            {
                                ifbuildpath = true;
                                //Injectionrate.push_back({double(M_distance_name[m-1][n]),double(M_distance_name[j-1][k]),Injectionrate_layer});
                                latency_BFS[M_distance_name[j-1][k]] = latency_BFS[M_distance_name[m-1][n]] + fixedLatencyTable[M_distance_name[m-1][n]][M_distance_name[j-1][k]];
                                if (latency_BFS[M_distance_name[j-1][k]] > Latency_max[layer_id-1]) Latency_max[layer_id-1] = latency_BFS[M_distance_name[j-1][k]];
                                break;
                            }
                        }
                        if (ifbuildpath) break;
                    } 
                    if (!ifbuildpath){
                        //Injectionrate.push_back({double(i),double(M_distance_name[j-1][k]),Injectionrate_layer});
                        latency_BFS[M_distance_name[j-1][k]] = latency_BFS[i] + fixedLatencyTable[i][M_distance_name[j-1][k]];
                        if (latency_BFS[M_distance_name[j-1][k]] > Latency_max[layer_id-1]) Latency_max[layer_id-1] = latency_BFS[M_distance_name[j-1][k]];
                    } 
                }  
                
                
            }
            /*
            for (int j=0;j<Layer_net_num[layer_id];j++)
            {
                Injectionrate.push_back({double(i),double(Layer_net[layer_id][j]),Injectionrate_layer});
            }
            */
        } 
    }else {
    cout << "Invalid Routing Planning pattern: " << _routing_planning_pattern << endl;
    exit(-1);
    }

    filename = _layerlatency_filename;
    if (filename == "none") return ;
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return ; 
    }

    for (int i=0;i<_layer-1;i++)
    {
     outfile<<"Latency between Layer "<<i<<" to "<<i+1<<" : "<<Latency_max[i]<<endl;
     cout<<"Latency between Layer "<<i<<" to "<<i+1<<" : "<<Latency_max[i]<<endl;
    } 
    
    outfile.close();
    
    return ;
}