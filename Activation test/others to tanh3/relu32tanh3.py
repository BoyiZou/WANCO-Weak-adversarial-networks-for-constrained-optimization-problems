import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import mpl_toolkits.mplot3d as p3d


class ResNet(torch.nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.params = params

        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        xxx = x[:,0]

        yyy = x[:,1]

        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x = (xxx-0).reshape(-1,1)*(xxx-1).reshape(-1,1)*x.reshape(-1,1)*(yyy-0).reshape(-1,1)*(yyy-1).reshape(-1,1) - 1
        return x


class Block(nn.Module):

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x



class drrnn(nn.Module):

    def __init__(self, in_N, m, out_N, depth=1):
        super(drrnn, self).__init__()
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.Tanh()
        self.stack = nn.ModuleList()

        self.stack.append(nn.Linear(in_N, m))

        for i in range(depth):
            self.stack.append(Block(m, m, m))

        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x




def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainWANCO(model,device,params,optimizer,scheduler,model_2,optimizer_2,scheduler_2):




    Pix = 1000
    x = torch.arange(0,1+1/Pix,1/Pix)
    y = torch.arange(0,1+1/Pix,1/Pix)
    TestData = torch.zeros([(Pix+1)**2,params["width"]]).to(device)
    X,Y = torch.meshgrid(x,y)

    XX = X.reshape(-1,1)
    YY = Y.reshape(-1,1)
    XX = XX.squeeze()
    YY = YY.squeeze()
    print(X.shape)
    TestData[:,0] = XX
    TestData[:,1] = YY

    TestData.requires_grad = True


    xxx_2 = torch.zeros((1,1)).float().to(device)

    model.train()
    model_2.train()
    data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data.requires_grad = True

    FF = open("loss.txt","w")
    GG = open("lambda.txt","w")
    JJ = open("mass.txt","w")

    for step in range(params["trainStep"]):

        uxt = model(data)
        yyy_2 = model_2(xxx_2)
        model.zero_grad()
        model_2.zero_grad()

        dudxy = torch.autograd.grad(uxt,data,grad_outputs=torch.ones_like(uxt),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        grad_u_2norm_2 = dudxy[:,0]**2+dudxy[:,1]**2
        grad_u_2norm_2 = grad_u_2norm_2.reshape(-1,1)

        mass = (torch.mean(uxt)-params["volumn"])
        loss = params["penalty2"]*(1/2*torch.mean(grad_u_2norm_2) + (1/(params["epsilon"]**2))*torch.mean((uxt**2-1)**2)) -yyy_2*mass  + params["penalty"]*(mass)**2

        loss_2 = yyy_2*mass

        if step%params["writeStep"] == params["writeStep"]-1:
          print("Error at Step %s is %s."%(step+1,loss))
          file = open("loss.txt","a")
          file.write(str(loss)+"\n")
          file = open("lambda.txt","a")
          file.write(str(yyy_2.cpu().detach().numpy())+"\n")
          file = open("mass.txt","a")
          file.write(str(mass.cpu().detach().numpy())+"\n")

        if step%params["plotStep"] == params["plotStep"]-1:
          Train_u = model(TestData)
          Train_u = Train_u.cpu().detach().numpy()

          plt.imshow(Train_u.reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          fig = plt.figure()
          ax = p3d.Axes3D(fig)
          ax.plot_wireframe(X,Y,Train_u.reshape(Pix+1,Pix+1))
          plt.show()

        if step%params["sampleStep"] == params["sampleStep"]-1:
          np.random.seed(step)
          data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
          data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data.requires_grad = True


        if step%2 > 0:
          loss_2.backward()

          optimizer_2.step()
          scheduler_2.step()
        else:
          loss.backward()

          optimizer.step()
          scheduler.step()

        params["penalty"] = params["penalty"]*params["alpha"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 

def main():
    # Parameters
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.05
    params["d"] = 50
    params["dd"] = 1
    params["bodyBatch"] = 160000
    params["lr"] = 0.016 
    params["lr2"] = 0.032
    params["width"] = 50 
    params["depth"] = 4 
    params["trainStep"] = 20000
    params["penalty"] = 1000  
    params["alpha"] = 1.0003
    params["volumn"] = -0.5
    params["penalty2"] = 20
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 50
    params["milestone"] = [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,9500]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = ResNet(params).to(device)
    model_2 = drrnn(1, 10, 1).to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(),lr=params["lr2"],weight_decay=params["decay"])
    scheduler_2 = MultiStepLR(optimizer_2,milestones=params["milestone"],gamma=params["gamma"])
    print("Generating network costs %s seconds."%(time.time()-startTime))

    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

    startTime = time.time()
    trainWANCO(model,device,params,optimizer,scheduler,model_2,optimizer_2,scheduler_2)  
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")



if __name__=="__main__":
    main()