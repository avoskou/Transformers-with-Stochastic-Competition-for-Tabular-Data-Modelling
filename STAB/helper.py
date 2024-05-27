import torch
import torch.nn    as nn
import torch.optim as optim
import  torch.nn.functional as     F
from copy import deepcopy
from STab.LWTA import LWTA
import weakref





#Dropout 
class Dropout(nn.Module):
    # if False => mc dropout
    deterministic = False
    def __init__(self, p=0.5,inference_factor=1.0):
        super(Dropout, self).__init__()
        self.p = p
        
        self.inference_factor=min(max(inference_factor,0.0),1.0)
        
       
        
    def forward(self, x):
        if  self.training or Dropout.deterministic  :
            return F.dropout(x, self.p, self.training )  
        
        return F.dropout(x, self.p, True)  


#Similar to Timedistributed of tf keras
class LocalLinear(nn.Module):
    def __init__(self, in_size, out_sizes,ftnum):
        super().__init__()
        
        self.w = nn.Parameter(torch.randn(ftnum,in_size, out_sizes))
        self.b = nn.Parameter(torch.randn(1,ftnum, out_sizes))
        
    def forward(self, x):
        x = x.transpose(0, 1)  
        x = torch.bmm(x,self.w)
        x = x.transpose(0, 1)  
        y = x+self.b
        
        return y 


#maps Categorical features to integers
class CatMap:
    def __init__(self,X_train_cat):
        self.CatMapping={ c:  { element: index for index, element in enumerate(list( X_train_cat[c].unique())) }    for c in list(X_train_cat.columns) } 
    def __call__(self,df_):
        
        df=deepcopy(df_)
        for c in df.columns:
            df[c]=df[c].map(self.CatMapping[c])
        return df.astype(int)

def MyRegreLoss(kl_w=0.1,M=0.5):
    def custom_loss(output, target):

        
        loss1 = LWTA.KL_loss()
    
        loss2 = F.mse_loss(output, target)

        combined_loss = kl_w * loss1 + (1-kl_w) * loss2

        return combined_loss*M
    return custom_loss


def MyClassLoss(kl_w=0.1,M=0.5):
    def custom_loss(output, target):

        
        loss1 = LWTA.KL_loss()
    
        loss2 = F.cross_entropy(output, target)

        combined_loss = kl_w * loss1 + (1-kl_w) * loss2

        return combined_loss*M
    return custom_loss