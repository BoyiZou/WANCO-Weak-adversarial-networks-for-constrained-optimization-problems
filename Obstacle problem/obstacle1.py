import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os

# create the class of primal network
class ResNet(torch.nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"]) # input layer
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"])) # hidden layers

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"]) # output layer

    def forward(self, x):
        xx = x[:,0].reshape(-1,1) # record the input coordinates to exactly imposed the boundary (efficient for simple geometry)

        x = self.linearIn(x)  # input layer
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3 # activation $\tanh^3$
            x = x_temp+x # residual connection

        x = self.linearOut(x) # output layer
        x = x.reshape(-1,1)
        x = x*(xx)*(1-xx) # exactly impose the boundary condition

        return x

# create the class of adversarial network
class ResNet_1(torch.nn.Module):
    def __init__(self, params):
        super(ResNet_1, self).__init__()
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

        x = -F.relu(-x) # non-positivity constaint

        return x

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainWANCO(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,phi_fun):

    model.train()
    model_1.train()

    # generate training data
    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    xx = data[:,0].reshape(-1,1)

    phi = phi_fun(xx)
    data.requires_grad = True

    FF = open("loss.txt","w")

    for step in range(params["trainStep"]):

        u = model(data).reshape(-1,1)
        model.zero_grad()
        model_1.zero_grad()
        # compute loss
        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        dudx = dudx[:,0].reshape(-1,1)
        lambda_1 = model_1(data).reshape(-1,1) # adversarial term

        loss____1 = params["penalty_F"]*torch.mean((dudx**2))

        loss____2 =  - torch.mean(lambda_1*((phi-u))) + params["penalty"]/2*torch.mean((F.relu(phi-u))**2)

        loss =  loss____1 + loss____2
 
        loss1 = -loss

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():

                print("Error %s,loss_1: %s, loss_2: %s at Step %s ."%(loss.cpu().detach().numpy(),\
                  loss____1.cpu().detach().numpy(),loss____2.cpu().detach().numpy(),step+1))
                file = open("loss.txt","a")
                file.write(str(loss.cpu().detach().numpy())+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:
            np.random.seed(step)
            data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
            data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            xx = data[:,0].reshape(-1,1)
            phi = phi_fun(xx)
            data.requires_grad = True

        if step%params["plotStep"] == 1:
            test(model,model_1,device,params,phi_fun)
        # adversarial process
        if step%4 > 2:
          loss1.backward()

          optimizer_1.step()
          scheduler_1.step()
        else:
          loss.backward()

          optimizer.step()
          scheduler.step()

        if step < 25000:
          params["penalty"] = params["penalty"]*params["alpha"]

def test(model,model_1,device,params,phi_fun):

    data = torch.zeros((params['numQuad'],params["d"])).float().to(device)
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
    plt.scatter(data[:,0],u-phi,c='y',s=0.05)

    plt.title('point wise error u_{train}-obstacle')
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

# obstacle function
def phi_fun(x):

    phi1 = ((0.25-x)>0+0).reshape(-1,1)*100*x**2

    phi2 = ((0.25-x)*(x-0.75)>0+0).reshape(-1,1)*(100*x*(1-x)-12.5)

    phi3 = ((x-0.75)>0+0).reshape(-1,1)*100*(1-x)**2

    phi = phi1 + phi2 + phi3

    phi = phi.reshape(-1,1)

    return phi

# analytic solution
def exactnew(data,params):

    xx = data[:,0].reshape(-1,1)

    nexact1 = ((1/(2*2**0.5)-xx)>0+0).reshape(-1,1)*(100-50*2**0.5)*xx

    nexact2 = ((1/(2*2**0.5)-xx)*(xx-1+1/(2*2**0.5))>0+0).reshape(-1,1)*(100*xx*(1-xx)-12.5)

    nexact3 = ((xx-1+1/(2*2**0.5))>0+0).reshape(-1,1)*(100-50*2**0.5)*(1-xx)

    nexact = nexact1 + nexact2 + nexact3

    nexact = nexact.reshape(-1,1)

    return nexact


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) # if p.requires_grad

def main():

    torch.manual_seed(2)
    np.random.seed(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()

    params["d"] = 1
    params["dd"] = 1
    params["bodyBatch"] = 5000
    params["lr"] = 0.016 
    params["lr2"] = 0.016
    params["width"] = 120 
    params["depth"] = 8 
    params["trainStep"] = 40000
    params["penalty"] = 100000  
    params["penalty_F"] = 1000
    params["alpha"] = 1.0003
    params["writeStep"] = 50
    params["plotStep"] = 1000
    params["sampleStep"] = 10
    params["milestone"] = [500,1500,2200,3000,3700,4500,6000,8000,10000,12000,14000,17000,19000,21000,23000,25000,27000,29000,35000,39000,42000,45000,48000,49500]
    params["milestone2"] = [2500,5000,10000,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500]
    params["gamma"] = 0.5
    params["decay"] = 0.0001
    params["beta"] = 5000
    params["width_1"] = 40
    params["depth_1"] = 3
    params["dd_1"] = 1

    startTime = time.time()
    model = ResNet(params).to(device)
    model_1 = ResNet_1(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_1 = torch.optim.Adam(model_1.parameters(),lr=params["lr2"],weight_decay=params["decay"])
    scheduler_1 = MultiStepLR(optimizer_1,milestones=params["milestone2"],gamma=params["gamma"])

    print("Generating network costs %s seconds."%(time.time()-startTime))
    startTime = time.time()
    trainWANCO(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,phi_fun)  #trainWANCO model
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()

    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")



if __name__=="__main__":
    main()