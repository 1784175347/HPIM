#ifndef _CNN_HPP_
#define _CNN_HPP_

#include <vector>
#include <utility>
#include <cassert>
#include <fstream>
#include "booksim_config.hpp"

struct ConvLayerScale {
    // 成员变量声明
    int inputChannels;
    int outputChannels;
    int kernelHeight;
    int kernelWidth;
    int stride;
    int padding;
    bool hasPooling;
    int poolSize;
    int poolStride;
    bool isDense;
    std::pair<int, int> inputSize;
    std::pair<int, int> outputSize;

    // 构造函数
    ConvLayerScale(int inCh, int outCh, int kH, int kW, int st, int pad, bool hasP, int pS, int pSt,
                   int inW, int inH, bool isDen)
        : inputChannels(inCh), outputChannels(outCh), kernelHeight(kH), kernelWidth(kW),
          stride(st), padding(pad), hasPooling(hasP), poolSize(hasP ? pS : 1), poolStride(hasP ? pSt : 1),
          isDense(isDen), inputSize({inW, inH})// 确保inputSize在isDense之后初始化
    {
        
        // 计算输出尺寸
        outputSize = calculateOutputSize();
    }

private:
    // 私有方法，用于计算输出尺寸
    std::pair<int, int> calculateOutputSize() const {
        int outputWidth = (inputSize.first - kernelWidth + 2 * padding) / stride + 1;
        int outputHeight = (inputSize.second - kernelHeight + 2 * padding) / stride + 1;

        if (hasPooling) {
            outputWidth = (outputWidth - poolSize) / poolStride + 1;
            outputHeight = (outputHeight - poolSize) / poolStride + 1;
        }

        // 对于全连接层，输出尺寸是1x1
        if (isDense) {
            outputWidth = 1;
            outputHeight = 1;
        }

        return {outputWidth, outputHeight};
    }
};



class CNNNetwork {
private:
    std::vector<ConvLayerScale> layers;
    void checkCompatibilityBetween(size_t currentLayerIndex, const ConvLayerScale& newLayer);
    int _layer;
    int _bitwidth;
    int _FPS;
    int _k;
    int _n;
    int _netnodes;
    int _channelbitwidth;
    int _freq;
    std::string _latency_filename;
    std::string _injection_filename;
    std::string _dest_filename;
    std::string _layerlatency_filename;

    string _distribution_pattern;
    string _routing_planning_pattern;

    vector<int> _netCIMHeight;
    vector<int> _netCIMWidth;

    vector<int> _inputChannels;
    vector<int> _outputChannels;
    vector<int> _kernelHeight;
    vector<int> _kernelWidth;
    vector<int> _stride;
    vector<int> _padding;
    vector<int> _hasPooling;
    vector<int> _poolSize;
    vector<int> _poolStride;
    vector<int> _isDense;
    vector<int> _inputSizeHeight;
    vector<int> _inputSizeWidth;
    vector<int> _outputSizeHeight;
    vector<int> _outputSizeWidth;


    std::vector<vector<double>> Injectionrate;
    std::vector<vector<int>> Dest;
    std::vector<vector<int>> Layer_net;
    std::vector<int> Layer_net_num;
    std::vector<int> Net_layer;
    std::vector<int> Distribution;
    std::vector<double> Latency_max;

public:
    CNNNetwork(const Configuration & config);
    void addLayerScale(const ConvLayerScale& scale);
    void printNetworkScale() const;
    void GetInjectionrate(size_t count);
    void GetLatency();
};

#endif