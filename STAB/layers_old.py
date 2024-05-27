import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy  as np
import weakref


from  STab.utils import concrete_sample

class  AttentionBias(nn.Module):
    """
    Adds traible bias to attention  scores    
    """
    def __init__(self,heads,size):
        super().__init__()
        
        self.static_bias    =  nn.Parameter(torch.zeros((1,heads, size,size) ))

    def forward(self,x):
         
        return x+self.static_bias

    
    
    
class LWTA(nn.Module):
    """
    A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """
    instances=weakref.WeakSet()
    ID=0
    
    
    # Tracking lw kl loss 
    @staticmethod 
    def KL_loss():
        out=0
        for layer in  LWTA.instances:
            out+=layer.kl_*layer.kl_w
        return out
    
    def __init__(self, inplace = True, deterministic = False, U=2,kl_w=1,T=0.67):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temperature =T
        self.temp_test = 0.01
        self.deterministic = deterministic
        self.kl_ = 0
        self.U = U
        self.temp = nn.Parameter(torch.tensor(self.temperature))
        LWTA.instances.add(self)
        self.ID=LWTA.ID
        LWTA.ID+=1
        self.kl_w=kl_w

    def forward(self, input):
        out, kl = lwta_activation(input, U = self.U, training = self.training,
                                  temperature = self.temp,
                                  deterministic=self.deterministic, temp_test = self.temp_test)
        if self.training:
            self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def lwta_activation(input, U = 2, deterministic = False, training = True,
                    temperature = 0.67, temp_test = 0.01):
    """
    The general LWTA activation function.
    Can be either deterministic or stochastic depending on the input.
    Deals with both FC and Conv Layers.
    """
 
    kl= 0.

    
    if True:
       
    logits = torch.reshape(input, [-1, input.size(-1)//U, U])
        

    if deterministic:
        a = torch.argmax(logits, 2, keepdims = True)
        mask_r = torch.zeros_like(logits).scatter_(2, a, 1.).reshape(input.shape)
    else:
        if training:
            mask = concrete_sample( logits, 0.67)
        else:
            mask = concrete_sample(logits, temp_test  )
            mask_r = mask.reshape(input.shape)
    
    if training and not deterministic:
        q = mask
        log_q = torch.log(q + 1e-8)
        log_p = torch.log(torch.tensor(1.0 / U))

        kl = torch.sum(q * (log_q - log_p), 1)
        kl = torch.mean(kl) / 1000.
    
    input = input*mask_r

    return input, kl











