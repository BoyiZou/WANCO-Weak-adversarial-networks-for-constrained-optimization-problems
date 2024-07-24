import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os

class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x, params,number_of_pts,device):
        xxx = x[:,0].reshape(-1,1)

        yyy = x[:,1].reshape(-1,1)

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)

        a = torch.zeros([number_of_pts,self.params["dd"]]).float().to(device)
        ms = torch.ones([number_of_pts,self.params["dd"]]).float().to(device)

        a1 = (1-xxx)*(1-yyy)*yyy*0.5*torch.sin(torch.pi*yyy) + xxx*yyy*(1-yyy)*1.5*F.relu(torch.sin((3*yyy-1)*torch.pi))
        a1 = a1.reshape(-1,1)
        a[:,0] = a1.squeeze()

        m = (xxx*(1-xxx)*(1-yyy)*yyy).squeeze()
        ms[:,0] = m
        ms[:,1] = m

        x =  ms*x+a

        return x


class RitzNet_1(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet_1, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width_1"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth_1"]):
            self.linear.append(nn.Linear(self.params["width_1"], self.params["width_1"]))

        self.linearOut = nn.Linear(self.params["width_1"], self.params["dd_1"])

    def forward(self, x):

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x = x - torch.mean(x)

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

def trainnew(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,model_2,optimizer_2,scheduler_2):

    Pix = 1000
    x = torch.arange(0,1+1/Pix,1/Pix)
    y = torch.arange(0,1+1/Pix,1/Pix)
    TestData = torch.zeros([(Pix+1)**2,params["d"]]).to(device)
    X,Y = torch.meshgrid(x,y)

    XX = X.reshape(-1,1)
    YY = Y.reshape(-1,1)
    XX = XX.squeeze()
    YY = YY.squeeze()

    TestData[:,0] = XX
    TestData[:,1] = YY
    TestData.requires_grad = True

    Pix_2 = 20
    x_2 = torch.arange(0,1+1/Pix_2,1/Pix_2)
    y_2 = torch.arange(0,1+1/Pix_2,1/Pix_2)
    TestData_2 = torch.zeros([(Pix_2+1)**2,params["d"]]).to(device)
    XXXX,YYYY = torch.meshgrid(x_2,y_2)
    XXXX = XXXX.reshape(-1,1)
    YYYY = YYYY.reshape(-1,1)
    TestData_2[:,0] = XXXX.squeeze()
    TestData_2[:,1] = YYYY.squeeze()

    model.train()
    model_1.train()
    model_2.train()
    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data.requires_grad = True

    FF = open("loss.txt","w")
    EE = open("lr.txt","w")
    GG = open("lambda.txt","w")
    JJ = open("mass.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")

    for step in range(params["trainStep"]):

        uxt = model(data,params,params["bodyBatch"],device)
        u1 = uxt[:,0].reshape(-1,1)
        u2 = uxt[:,1].reshape(-1,1)
        phi = uxt[:,2].reshape(-1,1)
        p = model_1(data).reshape(-1,1)

        model.zero_grad()
        model_2.zero_grad()
        model_1.zero_grad()

        du1dxy = torch.autograd.grad(u1,data,grad_outputs=torch.ones_like(u1),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du2dxy = torch.autograd.grad(u2,data,grad_outputs=torch.ones_like(u2),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        dphidxy = torch.autograd.grad(phi,data,grad_outputs=torch.ones_like(phi),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

        u_2norm_2 = u1**2+u2**2

        grad_u1_2norm_2 = du1dxy[:,0]**2+du1dxy[:,1]**2
        grad_u1_2norm_2 = grad_u1_2norm_2.reshape(-1,1)
        grad_u2_2norm_2 = du2dxy[:,0]**2+du2dxy[:,1]**2
        grad_u2_2norm_2 = grad_u2_2norm_2.reshape(-1,1)
        grad_phi_2norm_2 = dphidxy[:,0]**2+dphidxy[:,1]**2
        grad_phi_2norm_2 = grad_phi_2norm_2.reshape(-1,1)

        div_u = du1dxy[:,0]+du2dxy[:,1]

        volume = torch.mean(phi)-params["C"]

        J_alpha = 0.5*torch.mean((grad_u1_2norm_2+grad_u2_2norm_2))+0.5*params["alpha0"]*torch.mean((1-phi)**2*u_2norm_2)
        J_epsilon =  params["eta"]*(0.5*params["epsilon"]*torch.mean(grad_phi_2norm_2) + 1/(8*params["epsilon"])*torch.mean(phi**2*(1-phi)**2))
        Lag_p = -torch.sum(div_u*p) + params["beta1"]/2*torch.sum(div_u**2)

        Lag_volume = params["beta2"]/2*volume**2

        loss = J_alpha +J_epsilon + Lag_p + Lag_volume

        loss1 = -Lag_p

        if step%params["writeStep"] == params["writeStep"]-1:
          print("Error: %s , J_alpha: %s, J_epsilon: %s, Lag_p: %s, Lag_volume: %s   at Step %s."\
                %(loss.cpu().detach().numpy(),J_alpha.cpu().detach().numpy(),J_epsilon.cpu().detach().numpy(),Lag_p.cpu().detach().numpy(),Lag_volume.cpu().detach().numpy(),step+1))

          file = open("loss.txt","a")
          file.write(str(loss)+"\n")

          file = open("mass.txt","a")
          file.write(str(volume.cpu().detach().numpy())+"\n")

        if step%params["plotStep"] == params["plotStep"]-1:
          Train_u = model(TestData,params,1002001,device)
          p = model_1(TestData).reshape(-1,1)
          Train_u = Train_u.cpu().detach().numpy()
          Train_p = p.cpu().detach().numpy()

          Train_u1 = Train_u[:,0].reshape(Pix+1,Pix+1)
          Train_u2 = Train_u[:,1].reshape(Pix+1,Pix+1)
          Train_phi = Train_u[:,2].reshape(Pix+1,Pix+1)
          Train_p = Train_p.reshape(Pix+1,Pix+1)

          plt.figure()
          plt.imshow(Train_u1.T,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u2.T,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          plt.imshow(Train_phi.T,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          phase_fluid = (Train_phi>0.5)+0
          plt.figure()
          plt.imshow(phase_fluid.T,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_p.T,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          Train_u = model(TestData_2,params,441,device)
          Train_u = Train_u.cpu().detach().numpy()
          Train_u1 = Train_u[:,0].reshape(Pix_2+1,Pix_2+1)
          Train_u2 = Train_u[:,1].reshape(Pix_2+1,Pix_2+1)

          plt.figure()
          plt.quiver(XXXX,YYYY,Train_u1,Train_u2,width = 0.001,scale = 3)
          plt.show()


        if step%params["writeStep"] == params["writeStep"]-1:
          data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
          data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data.requires_grad = True

        if step%4 > 2:
          loss1.backward()

          optimizer_1.step()
          scheduler_1.step()
        else:
          loss.backward()

          optimizer.step()
          scheduler.step()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 

def main():
    # Parameters
    torch.manual_seed(22)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.01
    params["d"] = 2
    params["dd"] = 3
    params["bodyBatch"] = 20000
    params["lr"] = 0.016
    params["lr1"] = 0.006
    params["width"] = 50
    params["depth"] = 4 
    params["trainStep"] = 30000 
    params["alpha"] = 1.0005
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 10
    params["milestone"] = [400,800,1500,2400,3500,5500,7000,9000,11000,13000,15000,17000,19000,21000,23000,25000,27000,29000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    params["alpha0"] = 1000000
    params["eta"] = 10
    params["beta1"] = 10
    params["beta2"] = 1000
    params["C"] = 0.5
    params["width_1"] = 1
    params["depth_1"] = 1
    params["dd_1"] = 1

    startTime = time.time()
    model = RitzNet(params).to(device)
    model_1 = RitzNet_1(params).to(device)
    model_2 = drrnn(1, 1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_1 = torch.optim.Adam(model_1.parameters(),lr=params["lr1"],weight_decay=params["decay"])
    scheduler_1 = MultiStepLR(optimizer_1,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_2 = torch.optim.Adam(model_2.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler_2 = MultiStepLR(optimizer_2,milestones=params["milestone"],gamma=params["gamma"])

    print("Generating network costs %s seconds."%(time.time()-startTime))
    startTime = time.time()
    trainnew(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,model_2,optimizer_2,scheduler_2)  #trainnew model
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()

    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

if __name__=="__main__":
    main()