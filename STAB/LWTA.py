import torch, math 
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from  STab.utils import concrete_sample, kumaraswamy_sample, bin_concrete_sample, kl_divergence_kumaraswamy
import numpy as np
import weakref
import pandas as pd
import math
import torch
import time
from torch import nn, Tensor




class LWTA(nn.Module):
    """
    A simple implementation of the LWTA  to be used as a standalone approach for various models.
    """
    instances=weakref.WeakSet()
    ID=0
    
    @staticmethod 
    def KL_loss():
        out=0
        for layer in  LWTA.instances:
            out+=layer.kl_*layer.kl_w
        return out
    
    def __init__(self, inplace = True, deterministic = False, U=2,kl_w=1,T=0.5,return_mask=False):
        super(LWTA, self).__init__()
        self.inplace = inplace
        self.temperature =T
        self.temp_test = 0.01
        self.deterministic = deterministic
        self.kl_ = 0
        self.U = U
        self.temp = Parameter(torch.tensor(self.temperature))
        LWTA.instances.add(self)
        self.ID=LWTA.ID
        LWTA.ID+=1
        self.kl_w=kl_w
        self.return_mask=return_mask

    def forward(self, input):
        out, kl,mask = lwta_activation(input, U = self.U, training = self.training,
                                  temperature = self.temp,
                                  deterministic=self.deterministic, temp_test = self.temp_test)
        if self.training:
            self.kl_ = kl
        if self.return_mask:
            return out,mask
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
   # out = input.clone()
    kl= 0.
    #return input, 0,input*0+1
    # case of a fully connected layer
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

    return input, kl,mask_r










class Gsoftmax(nn.Module):
    """
    A simple implementation of the LWTA activation to be used as a standalone approach for various models.
    """
    instances=weakref.WeakSet()
    ID=0
    
    @staticmethod 
    def KL_loss():
        out=0
        for layer in  Gsoftmax.instances:
            out+=layer.kl_
        return out
    
    def __init__(self, inplace = True, deterministic = False, U=2):
        super(Gsoftmax, self).__init__()
        self.inplace = inplace
        self.temperature = 1.00
        self.temp_test = 0.01
        self.deterministic = deterministic
        self.kl_ = 0
        self.U = U
        self.temp = Parameter(torch.tensor(self.temperature))
        Gsoftmax.instances.add(self)
        self.ID=Gsoftmax.ID
        Gsoftmax.ID+=1

    def forward(self, input):
        out, kl = Gsoftmax_activation(input, U = self.U, training = self.training,
                                  temperature = self.temp,
                                  deterministic=self.deterministic, temp_test = self.temp_test)
        if self.training:
            self.kl_ = kl

        return out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def Gsoftmax_activation(input, U = 2, deterministic = False, training = True,
                    temperature = 0.67, temp_test = 0.01):
    """
    The general LWTA activation function.
    Can be either deterministic or stochastic depending on the input.
    Deals with both FC and Conv Layers.
    """
    out = input.clone()
    kl= 0.
    U=out.size(1)
    # case of a fully connected layer
    if True:
        logits = torch.reshape(out, [-1, out.size(1), out.size(2)])
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

    input *= mask_r

    return input, kl
