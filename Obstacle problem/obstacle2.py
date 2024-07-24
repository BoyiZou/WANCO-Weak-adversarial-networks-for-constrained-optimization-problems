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
        xx = x[:,0].reshape(-1,1)


        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x = x.reshape(-1,1)
        x  = x*torch.sin(torch.pi*xx)


        return x


class ResNet_1(torch.nn.Module):
    def __init__(self, params):
        super(ResNet_1, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["width"], self.params["width_1"])
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

        x = -F.relu(x)

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

        return -F.relu(x)

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainWANCO(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,phi_fun):


    model.train()
    model_1.train()

    data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
    xx = data[:,0].reshape(-1,1)

    phi = phi_fun(xx)
    data.requires_grad = True

    bdrydatatimeleft = torch.zeros((1,params['width'])).float().to(device)
    bdrydatatimeleft[:,0] = 0
    bdrydatatimeright = torch.zeros((1,params['width'])).float().to(device)
    bdrydatatimeright[:,0] = 1

    FF = open("loss.txt","w")
    EE = open("lr.txt","w")
    GG = open("lambda.txt","w")
    JJ = open("mass.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")


    for step in range(params["trainStep"]):

        u = model(data).reshape(-1,1)
        model.zero_grad()
        model_1.zero_grad()

        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        dudx = dudx[:,0].reshape(-1,1)
        lambda_1 = model_1(data).reshape(-1,1)
        u_left = model(bdrydatatimeleft).reshape(-1,1)
        u_right = model(bdrydatatimeright).reshape(-1,1)

        loss____1 = params["penalty_F"]*torch.mean((dudx**2))
        loss____2 =  - torch.mean(lambda_1*((phi-u))) + params["penalty"]/2*torch.mean((F.relu(phi-u))**2)

        loss =  loss____1 + loss____2
        loss2 = -loss
        loss1 = -loss

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():

                print("Error %s,loss_1: %s, loss_2: %s at Step %s ."%(loss.cpu().detach().numpy(),\
                  loss____1.cpu().detach().numpy(),loss____2.cpu().detach().numpy(),step+1))
                file = open("loss.txt","a")
                file.write(str(loss.cpu().detach().numpy())+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
            data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            xx = data[:,0].reshape(-1,1)
            phi = phi_fun(xx)
            data.requires_grad = True

        if step%params["plotStep"] == 1:
            test(model,model_1,device,params,phi_fun)

        if step%5 > 3:
          loss1.backward()

          optimizer_1.step()
          scheduler_1.step()
        else:
          loss.backward()

          optimizer.step()
          scheduler.step()

        if step < 5000:
          params["penalty"] = params["penalty"]*params["alpha"]



def test(model,model_1,device,params,phi_fun):


    data = torch.zeros((params['numQuad'],params["width"])).float().to(device)
    data[:,0] = torch.from_numpy(np.linspace(0,1,params['numQuad'],endpoint=True))
    xx = data[:,0].reshape(-1,1)
    phi = phi_fun(xx)

    data.requires_grad=True

    u = model(data).reshape(-1,1)
    lambda_1 =model_1(data).reshape(-1,1)

    nexact = exactnew(data,params)

    model.zero_grad()
    model_1.zero_grad()

    dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

    dudx = dudx.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    u = u.cpu().detach().numpy()
    nexact = nexact.cpu().detach().numpy()
    phi = phi.cpu().detach().numpy()
    lambda_1 = lambda_1.cpu().detach().numpy()

    plt.figure()
    plt.scatter(data[:,0],lambda_1,c='y',s=0.05)

    plt.title('Adversarial term/ Lambda')
    plt.show()

    plt.figure()
    plt.scatter(data[:,0],phi,c='y',s=0.05)

    plt.title('Phi\Obstacle')
    plt.show()


    ppp = plt.figure()
    plt.scatter(data[:,0],u,c='y',s=0.05)

    plt.title('training solution')
    plt.show()

    plt.figure()
    plt.scatter(data[:,0],nexact,c='y',s=0.05)
    plt.title('Truth')
    plt.show()


    plt.figure()
    plt.scatter(data[:,0],nexact-u,c='y',s=0.05)

    plt.title('point wise error u_{truth}-u_{train}')
    plt.show()


    plt.figure()
    plt.scatter(data[:,0], u, color='blue', marker='.',s=0.05)
    plt.plot(data[:,0], nexact, color='red',linewidth=1)
    plt.plot(data[:,0], phi, color='yellow',linewidth=1)


    plt.legend(['train', 'truth', 'obstacle'])
    plt.xlabel('x')
    plt.ylabel('u')

    plt.show()

    return 1


def phi_fun(x):

    phi1 = ((0.25-x)>0+0).reshape(-1,1)*10*torch.sin(2*math.pi*x)

    phi2 = ((0.25-x)*(x-0.5)>0+0).reshape(-1,1)*(5*torch.cos((4*x-1)*math.pi)+5)

    phi3 = ((0.5-x)*(x-0.75)>0+0).reshape(-1,1)*(5*torch.cos((-4*x+3)*math.pi)+5)

    phi4 = ((x-0.75)>0+0).reshape(-1,1)*10*torch.sin(2*math.pi*(1-x))

    phi = phi1 + phi2 + phi3 + phi4

    phi = phi.reshape(-1,1)

    return phi


def exactnew(data,params):

    xx = data[:,0].reshape(-1,1)

    nexact1 = ((0.25-xx)>0+0).reshape(-1,1)*10*torch.sin(2*math.pi*xx)

    nexact2 = ((0.25-xx)*(xx-0.75)>0+0).reshape(-1,1)*10

    nexact3 = ((xx-0.75)>0+0).reshape(-1,1)*10*torch.sin(2*math.pi*(1-xx))

    nexact = nexact1 + nexact2 + nexact3

    nexact = nexact.reshape(-1,1)

    return nexact


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) # if p.requires_grad

def main():

    torch.manual_seed(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()

    params["d"] = 10
    params["dd"] = 1
    params["bodyBatch"] = 20000 

    params["lr"] = 0.016 
    params["lr2"] = 0.064 
    params["width"] = 120 
    params["depth"] = 8 
    params["trainStep"] = 20000
    params["penalty"] = 5000 
    params["penalty_F"] = 1
    params["alpha"] = 1.0003
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 10
    params["milestone"] = [10,100,200,500,700,1200,1500,2200,3000,3700,4500,5200,6000,7000,8000,9000,10000,11000,12000,13000,14000,15500,17000,18000,19000,19500,21000,23000,25000,27000,29000,35000,39000,42000,45000,48000,49500]
    params["milestone2"] = [5000,10000,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500]
    params["gamma"] = 0.5
    params["decay"] = 0.0001
    params["beta"] = 5000
    params["width_1"] = 10
    params["depth_1"] = 2
    params["dd_1"] = 1

    startTime = time.time()
    model = ResNet(params).to(device)
    model_1 = ResNet_1(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_1 = torch.optim.Adam(model_1.parameters(),lr=params["lr2"],weight_decay=params["decay"])
    scheduler_1 = MultiStepLR(optimizer_1,milestones=params["milestone"],gamma=params["gamma"])
    print("Generating network costs %s seconds."%(time.time()-startTime))
    startTime = time.time()
    trainWANCO(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,phi_fun)  
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()

    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")



if __name__=="__main__":
    main()