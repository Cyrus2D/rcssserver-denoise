"""Shared causal GRU and three position-only output distributions."""
import math
import torch
from torch import nn
import torch.nn.functional as F

KINDS=['point','gaussian','mixture']

class PositionModel(nn.Module):
    def __init__(self,features,kind,hidden=64,components=3):
        super().__init__();self.kind=kind;self.components=components if kind=='mixture' else 1
        self.embedding=nn.Embedding(12,4)
        self.input=nn.Sequential(nn.Linear(features+4,hidden),nn.SiLU())
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU())
        self.head=nn.Linear(hidden,2 if kind=='point' else 5 if kind=='gaussian' else components*6)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if kind=='mixture':
            with torch.no_grad():
                offsets=[(0.,0.),(.03,.03),(-.03,-.03)]
                for i,(x,y) in enumerate(offsets[:components]):self.head.bias[i*6:i*6+2]=torch.tensor([x,y])

    def forward(self,x,unum):
        emb=self.embedding(unum).unsqueeze(1).expand(-1,x.shape[1],-1)
        h,_=self.gru(self.input(torch.cat([x,emb],-1)))
        z=self.head(self.readout(h[:,-1]));last=x[:,-1]
        age=last[:,10:11]*30;velocity=last[:,2:4]*1.5
        if self.kind=='point':means=velocity*age+torch.tanh(z)*age*1.5;return dict(point=means)
        if self.kind=='gaussian':raw=z[:,None,:];weight=z.new_ones((len(z),1))
        else:
            raw=z.reshape(len(z),self.components,6);weight=raw[:,:,5].softmax(-1)
        means=velocity[:,None,:]*age[:,:,None]+torch.tanh(raw[:,:,:2])*age[:,:,None]*1.5
        scale=(F.softplus(raw[:,:,2:4])*.2*age[:,:,None]+.01).clamp(.01,100.)
        rho=.95*raw[:,:,4].tanh()
        point=(weight[:,:,None]*means).sum(1)
        return dict(point=point,means=means,scale=scale,rho=rho,weight=weight)

def log_prob(output,y):
    residual=(y[:,None,:]-output['means'])/output['scale']
    rho=output['rho'];den=(1-rho.square()).clamp_min(1e-5)
    quad=(residual[:,:,0].square()+residual[:,:,1].square()-2*rho*residual[:,:,0]*residual[:,:,1])/den
    component=-math.log(2*math.pi)-output['scale'].log().sum(-1)-.5*den.log()-.5*quad
    return torch.logsumexp(component+output['weight'].clamp_min(1e-12).log(),-1)

def point_output(output,rule='mean'):
    if rule=='map' and 'means' in output:
        return output['means'][torch.arange(len(output['point']),device=output['point'].device),output['weight'].argmax(-1)]
    return output['point']

def loss(output,y):
    error=(output['point']-y).square().sum(-1).add(1e-4).sqrt().mean()
    return error if 'means' not in output else -log_prob(output,y).mean()+.2*error

def covariance(output):
    """Total mixture covariance includes between-component uncertainty."""
    s=output['scale'];rho=output['rho'];w=output['weight'];mu=output['means']
    diff=mu-output['point'][:,None,:]
    xx=(w*(s[:,:,0].square()+diff[:,:,0].square())).sum(-1)
    yy=(w*(s[:,:,1].square()+diff[:,:,1].square())).sum(-1)
    xy=(w*(rho*s[:,:,0]*s[:,:,1]+diff[:,:,0]*diff[:,:,1])).sum(-1)
    return torch.stack([xx,yy,xy],-1)

def mahalanobis(output,y):
    cov=covariance(output);d=y-output['point'];a,b,c=cov.unbind(-1)
    return (b*d[:,0].square()+a*d[:,1].square()-2*c*d[:,0]*d[:,1])/(a*b-c.square()).clamp_min(1e-8)
