# HPIM-NoC V1.0
HPIM异构存算NoC仿真器设计代码

--|设计代码

&emsp;&emsp;--| booksim2

&emsp;&emsp;&emsp;&emsp;包含NoC仿真器——booksim2的设计代码（C++）及其配置文件

&emsp;&emsp;&emsp;&emsp;--| doc

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;包含Makefile与菜单文件

&emsp;&emsp;&emsp;&emsp;--| runfiles

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;包含booksim的配置文件与输出文件

&emsp;&emsp;&emsp;&emsp;--| src

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;包含booksim的可执行文件与设计代码

&emsp;&emsp;&emsp;&emsp;--| utils

&emsp;&emsp;--| MNSIM3.0

&emsp;&emsp;&emsp;&emsp;包含存算仿真器——MNSIM3.0的设计代码（python）及其配置文件

&emsp;&emsp;&emsp;&emsp;--| MNSIM

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;包含MNSIM3.0的设计代码

&emsp;&emsp;&emsp;&emsp;--| main.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;MNSIM的主函数

&emsp;&emsp;&emsp;&emsp;--| SA.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;搜索框架的主函数

&emsp;&emsp;&emsp;&emsp;--| SA_cifar10_resnet18_111.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Resnet18下(α,β,γ)=(1,1,1)搜索

&emsp;&emsp;&emsp;&emsp;--| SA_cifar10_alexnet_111.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Alexnet下(α,β,γ)=(1,1,1)搜索

&emsp;&emsp;&emsp;&emsp;--| SA_edp.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;EDP仿真与搜索

&emsp;&emsp;&emsp;&emsp;--| Simulator_cifar10_resnet18.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Resnet18下框架仿真器（启用NoC仿真）

&emsp;&emsp;&emsp;&emsp;--| Simulator_cifar10_resnet18_noNoC.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Resnet18下框架仿真器（禁用NoC仿真）

&emsp;&emsp;&emsp;&emsp;--| Simulator_cifar10_alexnet.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Alexnet下框架仿真器（启用NoC仿真）

&emsp;&emsp;&emsp;&emsp;--| Simulator_cifar10_alexnet_noNoC.py

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Cifar10-Alexnet下框架仿真器（禁用NoC仿真）

&emsp;&emsp;&emsp;&emsp;--| 其余文件

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;MNSIM的配置文件与网络权重文件

&emsp;&emsp;--| README.md

&emsp;&emsp;&emsp;&emsp;代码说明与可能问题

请按如下步骤操作

+ 1.cd HPIM/booksim/src

+ 2.make

+ 3.cd ../../MNSIM3.0

+ 4.python SA.py（注：不同需求请运行不同文件或对文件做参数修改）

可能存在的问题1————booksim2的可执行文件未编译

+ 到文件夹HPIM/booksim2/src下运行make编译booksim

可能存在的问题2————以下文件中的相对地址请根据实际情况修改

+ HPIM/booksim2/runfiles/nnmeshconfig:83-93

+ HPIM/MNSIM3.0/MNSIM\Latency_Model\Model_latency.py:2012:2034

配置文件与部分仿真流程相关的参数

+ 1.SimConfig.ini

&emsp;&emsp;&emsp;Tile内部电路参数：可修改Booksim_Flag以调整是否调用booksim; 1：调用

+ 2.nnmeshconfig

&emsp;&emsp;&emsp;NoC网络参数：可调整sample_period以增加仿真时间避免数据包不生成

+ 3.mix_tileinfo.ini

&emsp;&emsp;&emsp;异构网络节点配置参数，可根据需要修改，注意格式一致