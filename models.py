
import torch
import torch.nn as nn


#if depth or size changes and we need to resample for the residual connection addition.
class Resample_Channels(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Resample_Channels, self).__init__()
    self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
  def forward(self, x):
    return self.layer(x)

#a single residual block with 2 cnn layers. 
class SEED_DEEP_LAYER(nn.Module):
    def __init__(self, in_channels, out_channels, in_d_0=0, in_d_1=0, stride = 1, k=4, do_pool = False, dropout = 0.01, debug = False):
        super(SEED_DEEP_LAYER, self).__init__()
        self.do_pool = do_pool
        self.debug = debug
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = k, stride = stride, padding = 'same'),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        
        
        conv2_layers = [
                        nn.Conv2d(out_channels, out_channels, kernel_size = k, stride = stride, padding = 'same'),
                        nn.BatchNorm2d(out_channels)
                      ]
        self.mp = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = nn.AvgPool2d((2,2))
        #a way to build layers out of a list
        self.conv2 = nn.Sequential(*conv2_layers)
        
        #do we need to do downsampling?
        self.resample_channels = False
        if in_channels != out_channels:
          self.resampler = Resample_Channels(in_channels, out_channels)
          self.resample_channels = True
        self.a = nn.LeakyReLU()
      
        
    def forward(self, x):
        residual = x
        o = self.conv1(x)
        o = self.dropout(o)
        o = self.conv2(o)
        #going from in channels to out channels requires the residual to be resampled. so that it can be added
        if self.resample_channels:
          residual = self.resampler(residual)
        if self.do_pool:
          o = self.mp(o)
          o = o + self.downsample(residual)
        else:
          o = o + residual
        o = self.a(o)
        
        return o


#the model for CNN-LSTM-RES with and without grid.
class SEED_DEEP(nn.Module):
  def __init__(self, do_pool = True, in_channels=1, is_grid = False, grid_size = (200, 62), out_channels=200, num_classes = 3, num_res_layers = 5,  ll1=1024, ll2 = 256, dropoutc=0.01, dropout1= 0.5, dropout2 = 0.5, debug = False):
    super(SEED_DEEP, self).__init__()
    self.is_grid = is_grid
    self.debug = debug
    #must use modulelist vs regular list to keep everythign in GPU and gradients flowing
    self.res_layers = nn.ModuleList()
    c = in_channels
    for r in range(num_res_layers):
      self.res_layers.append(SEED_DEEP_LAYER(in_channels = c, out_channels=out_channels, do_pool = do_pool, dropout = dropoutc, debug = debug))
      c = out_channels
    self.lstm1 = nn.LSTM(ll1, ll1, batch_first=True)
    self.lin1 = nn.Linear(ll1, ll2)
    self.lin2 = nn.Linear(ll2, num_classes)
    self.do1 = nn.Dropout(dropout1)
    self.do2 = nn.Dropout(dropout2)
    self.ldown = None
    if do_pool:
      self.ldown = nn.Linear(300, ll1)
    else:
      self.ldown = nn.Linear(out_channels * grid_size[0]* grid_size[1], ll1)
    self.la = nn.ReLU()
  def forward(self, x):
    # x = x.unsqueeze(1)
    # print(x.shape)
    #if it's not a grid put it in shape (batch_size, 1, 200, 62), otherwise shape is (batch_size, 200, 9, 9)
    if not self.is_grid:
      x = torch.permute(x, (0,2,1))
      x = x.unsqueeze(1)
    o = x
    for i in range(len(self.res_layers)):
      o = self.res_layers[i](o)
    o = o.view(o.shape[0], -1)
    o = self.ldown(o) 
    res = o
    o, _ = self.lstm1(o)
    o = o + res
    o = self.do1(o)
    o = self.la(self.lin1(o))
    o = self.do2(o)
    o = self.lin2(o)
    return o
  
  
class SEED_DEEP_NO_LSTM(nn.Module):
  def __init__(self, do_pool = True, in_channels=1, is_grid = False, grid_size = (200, 62), out_channels=200, num_classes = 3, num_res_layers = 5,  ll1=1024, ll2 = 256, dropoutc=0.01, dropout1= 0.5, dropout2 = 0.5, debug = False):
    super(SEED_DEEP_NO_LSTM, self).__init__()
    self.is_grid = is_grid
    self.debug = debug
    #must use modulelist vs regular list to keep everythign in GPU and gradients flowing
    self.res_layers = nn.ModuleList()
    c = in_channels
    for r in range(num_res_layers):
      self.res_layers.append(SEED_DEEP_LAYER(in_channels = c, out_channels=out_channels, do_pool = do_pool, dropout = dropoutc, debug = debug))
      c = out_channels
    self.lstm1 = nn.LSTM(ll1, ll1, batch_first=True)
    self.lin1 = nn.Linear(ll1, ll2)
    self.lin2 = nn.Linear(ll2, num_classes)
    self.do1 = nn.Dropout(dropout1)
    self.do2 = nn.Dropout(dropout2)
    self.ldown = None
    if do_pool:
      self.ldown = nn.Linear(300, ll1)
    else:
      self.ldown = nn.Linear(out_channels * grid_size[0]* grid_size[1], ll1)
    self.la = nn.ReLU()
  def forward(self, x):
    # x = x.unsqueeze(1)
    # print(x.shape)
    #if it's not a grid put it in shape (batch_size, 1, 200, 62), otherwise shape is (batch_size, 200, 9, 9)
    if not self.is_grid:
      x = torch.permute(x, (0,2,1))
      x = x.unsqueeze(1)
    o = x
    for i in range(len(self.res_layers)):
      o = self.res_layers[i](o)
    o = o.view(o.shape[0], -1)
    o = self.ldown(o) 
    o = self.do1(o)
    o = self.la(self.lin1(o))
    o = self.do2(o)
    o = self.lin2(o)
    return o
  
  
#implementation of model for https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9219701/
class SEED_2D_DIFF_K(nn.Module):
  def __init__(self, in_channels=200,  num_classes = 3, p = 3, l1=8000):
    super(SEED_2D_DIFF_K, self).__init__()
    
    self.cb1_2 = nn.Sequential(
      nn.Conv2d(1, out_channels = 25, padding=(0,p), kernel_size=(5,1),stride=(1,2)),
      nn.LeakyReLU(),
      nn.Dropout2d(0.25),
      nn.Conv2d(25, out_channels = 25, padding=(0,p), kernel_size=(1,3), stride=(1,2)),
      nn.BatchNorm2d(25),
      nn.LeakyReLU(),
      nn.MaxPool2d((2,1))      
    )
    
    self.cb3_4 = nn.Sequential(
      nn.Conv2d(25, out_channels = 50, padding=(0,p),kernel_size=(5,1), stride=(1,2)),
      nn.LeakyReLU(),
      nn.Dropout2d(0.25),
      nn.Conv2d(50, out_channels = 50, padding=(0,p),kernel_size=(1,3), stride=(1,2)),
      nn.BatchNorm2d(50),
      nn.LeakyReLU(),
      nn.MaxPool2d((2,1))      
    )
    
    self.cb5_6 = nn.Sequential(
      nn.Conv2d(50, out_channels = 100, padding=(0,p),kernel_size=(5,1), stride=(1,2)),
      nn.LeakyReLU(),
      nn.Dropout2d(0.25),
      nn.Conv2d(100, out_channels = 100, padding=(0,p),kernel_size=(1,3), stride=(1,2)),
      nn.BatchNorm2d(100),
      nn.LeakyReLU(),
      nn.MaxPool2d((2,1))      
    )
    
    self.cb7_8 = nn.Sequential(
      nn.Conv2d(1, out_channels = 200, padding=(0,p),kernel_size=(5,1), stride=(1,2)),
      nn.LeakyReLU(),
      nn.Dropout2d(0.25),
      nn.Conv2d(200, out_channels = 200, padding=(0,p),kernel_size=(1,3), stride=(1,2)),
      nn.BatchNorm2d(200),
      nn.LeakyReLU(),
      nn.MaxPool2d((2,1))      
    )
    
    self.lin1 = nn.Linear(l1, 256)
    self.ld = nn.Dropout(0.5)
    self.lin2 = nn.Linear(256, num_classes)
    
  def forward(self, x):
    #reshape to fit
    x = torch.permute(x, (0,2,1))
    x = x.unsqueeze(1)
    
    o = self.cb1_2(x)
    o = self.cb3_4(o)
    o = self.cb5_6(o)
    o = self.cb7_8(o)
    o = o.view(o.shape[0], -1)
    o = self.lin1(o)
    o = self.ld(o)
    o = self.lin2(o)
    return o
  
  
