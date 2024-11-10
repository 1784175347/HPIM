import torch
import collections
import re
pthfile = r'/share/linqiushi-local/linqiushi/MNSIM-3.0/cifar10_alexnet_99qbitpim_params.pth'
net = torch.load(pthfile, map_location='cpu')
net = dict(net)
new_net=collections.OrderedDict()
for key,value in net.items():
    print(key)
    key=re.sub('module.', '', key)
    new_net[key]=value
    
torch.save(new_net,pthfile)    

