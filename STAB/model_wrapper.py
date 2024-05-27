import torch
import torch.nn    as nn
import torch.optim as optim
import  torch.nn.functional as     F



# model wrapper 
class Num_Cat(nn.Module):
    def __init__(self,model,stochastic=True,Sample_size=32,num_number=6,cat_num=6,classes=1):
        
        super(Num_Cat,self).__init__()
        
        #number of clasess
        self.classes=classes
        
        # number of  numerical and categorical features
        self.num_number = num_number
        self.cat_num    = cat_num
        
        
        #main model
        self.model=model
        
        #sample size for inference
        if stochastic:
            self.S=Sample_size
        else:
            self.S=1
        
        
        #number of numerical features
        self.num_number=num_number
    
    #modifies inference sample size (S)
    def reset_Sample_size(self,Sample_size):
        self.S=Sample_size
        
        
    def forward(self,x):
        
        
        # numerical features
        if  self.num_number ==0:
             nx = None
        else:
             nx=x[:,:self.num_number]
        
        
        
        # categorical features
        if  self.cat_num ==0:
            nc = None
        else:
            nc=(x[:,self.num_number:]).to(torch.long)
        
        
        if self.training:
                   
                    return self.model(nc,nx)
                
        s=x.shape
        
        #S sized sample 
        nx=torch.cat( [nx]*self.S)
        nc=torch.cat( [nc]*self.S)
     
        return self.model(nc,nx).view(self.S,s[-2],self.classes).mean(axis=0)
    
    
