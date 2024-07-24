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
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        xxx = x[:,0]

        yyy = x[:,1]

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)

        x = (xxx-0).reshape(-1,1)*(xxx-1).reshape(-1,1)*(yyy-0).reshape(-1,1)*(yyy-1).reshape(-1,1)*F.relu(x)
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

    def __init__(self, in_N, m, out_N, depth=4):
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
    TestData = torch.zeros([(Pix+1)**2,params["d"]]).to(device)
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
    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data.requires_grad = True

    FF = open("loss.txt","w")
    EE = open("lr.txt","w")
    GG = open("lambda.txt","w")
    JJ = open("mass_u1.txt","w")
    DD = open("mass_u2.txt","w")
    ZZ = open("mass_u3.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")

    for step in range(params["trainStep"]):

        uxt = model(data)
        u1 = uxt[:,0]
        u2 = uxt[:,1]
        u3 = uxt[:,2]
        yyy_2 = model_2(xxx_2)
        model.zero_grad()

        du1dxy = torch.autograd.grad(u1,data,grad_outputs=torch.ones_like(u1),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du2dxy = torch.autograd.grad(u2,data,grad_outputs=torch.ones_like(u2),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du3dxy = torch.autograd.grad(u3,data,grad_outputs=torch.ones_like(u3),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

        grad_u1_2norm_2 = du1dxy[:,0]**2+du1dxy[:,1]**2
        grad_u1_2norm_2 = grad_u1_2norm_2.reshape(-1,1)
        grad_u2_2norm_2 = du2dxy[:,0]**2+du2dxy[:,1]**2
        grad_u2_2norm_2 = grad_u2_2norm_2.reshape(-1,1)
        grad_u3_2norm_2 = du3dxy[:,0]**2+du3dxy[:,1]**2
        grad_u3_2norm_2 = grad_u3_2norm_2.reshape(-1,1)


        mass_u1 = (torch.mean(u1**2)-params["volumn"])
        mass_u2 = (torch.mean(u2**2)-params["volumn"])
        mass_u3 = (torch.mean(u3**2)-params["volumn"])
        mass = torch.abs(mass_u1) + torch.abs(mass_u2) + torch.abs(mass_u3)

        loss = params["penalty2"]*(1/2*(torch.mean(grad_u1_2norm_2)+torch.mean(grad_u2_2norm_2)+torch.mean(grad_u3_2norm_2)) + (2/(params["epsilon"]**2))*torch.mean((u1*u2)**2+(u2*u3)**2+(u1*u3)**2)) - yyy_2*(mass_u1+mass_u2+mass_u3)  +  params["penalty"]*(mass_u1**2+mass_u2**2+mass_u3**2)

        loss_2 =  yyy_2*(mass_u1+mass_u2+mass_u3)

        if step%params["writeStep"] == params["writeStep"]-1:
          print("Error at Step %s is %s."%(step+1,loss))
          file = open("loss.txt","a")
          file.write(str(loss)+"\n")
          file = open("lr.txt","a")
          file.write(str(scheduler.get_last_lr())+"\n")
          file = open("lambda.txt","a")
          file.write(str(yyy_2.cpu().detach().numpy())+"\n")
          file = open("mass_u1.txt","a")
          file.write(str(mass_u1.cpu().detach().numpy())+"\n")
          file = open("mass_u2.txt","a")
          file.write(str(mass_u2.cpu().detach().numpy())+"\n")
          file = open("mass_u3.txt","a")
          file.write(str(mass_u3.cpu().detach().numpy())+"\n")
          file = open("beta.txt","a")
          file.write(str(params["penalty"])+"\n")
          file = open("penalty2.txt","a")
          file.write(str(params["penalty2"])+"\n")

        if step%params["plotStep"] == params["plotStep"]-1:
          Train_u = model(TestData)
          Train_u = Train_u.cpu().detach().numpy()

          Train_u1 = Train_u[:,0].reshape(Pix+1,Pix+1)
          Train_u2 = Train_u[:,1].reshape(Pix+1,Pix+1)
          Train_u3 = Train_u[:,2].reshape(Pix+1,Pix+1)

          plt.figure()

          plt.imshow(Train_u[:,0].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          plt.imshow(Train_u[:,1].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u[:,2].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


          plt.figure()
          plt.imshow(Train_u[:,1].reshape(Pix+1,Pix+1)+Train_u[:,0].reshape(Pix+1,Pix+1)+Train_u[:,2].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()



          plt.figure()
          aaa = (Train_u1>Train_u2)+0
          bbb = (Train_u1>Train_u3)+0
          ccc = (Train_u2>Train_u1)+0
          ddd = (Train_u2>Train_u3)+0
          eee = (Train_u3>Train_u1)+0
          fff = (Train_u3>Train_u2)+0


          aaa = torch.from_numpy(aaa)
          bbb = torch.from_numpy(bbb)
          ccc = torch.from_numpy(ccc)
          ddd = torch.from_numpy(ddd)
          eee = torch.from_numpy(eee)
          fff = torch.from_numpy(fff)

          phase_u1 = 1*F.relu(aaa+bbb-1)
          phase_u2 = 2*F.relu(ccc+ddd-1)
          phase_u3 = 3*F.relu(eee+fff-1)

          phase = phase_u1+phase_u2+phase_u3
          phase = phase.cpu().detach().numpy()
          plt.imshow(phase,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()


        if step%params["writeStep"] == params["writeStep"]-1:
          data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 

def main():
    # Parameters
    torch.manual_seed(22)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.05
    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 3
    params["bodyBatch"] = 100000 
    params["lr"] = 0.016 
    params["width"] = 120 
    params["depth"] = 8 
    params["trainStep"] = 40000
    params["penalty"] = 10000000  
    params["alpha"] = 1.0005
    params["volumn"] = 1
    params["penalty2"] = 5000
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 10
    params["milestone"] = [1000,3000,5000,6000,8000,10000,14000,17000,19500,22500,24000,26000,28000,29000,33000,35000,37000,39000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = ResNet(params).to(device)
    model_2 = drrnn(1, 20, 1).to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(),lr=params["lr"],weight_decay=params["decay"])
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