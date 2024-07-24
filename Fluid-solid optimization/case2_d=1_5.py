import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os


class ResNet(torch.nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.params = params
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x, params,number_of_pts,device):

        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x[:,2] = torch.relu(x[:,2])

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
        x = x - torch.mean(x)

        return x

class ResNet_3(torch.nn.Module):
    def __init__(self, params):
        super(ResNet_3, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["width"], self.params["width_3"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth_3"]):
            self.linear.append(nn.Linear(self.params["width_3"], self.params["width_3"]))

        self.linearOut = nn.Linear(self.params["width_3"], self.params["dd_3"])

    def forward(self, x):

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)

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

def trainnew(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,model_2,optimizer_2,scheduler_2,model_3,optimizer_3,scheduler_3):
    
    Pix = 1000
    x = params["length"]*torch.arange(0,1+1/Pix,1/Pix)
    y = torch.arange(0,1+1/Pix,1/Pix)
    TestData = torch.zeros([(Pix+1)**2,params["width"]]).to(device)
    X,Y = torch.meshgrid(x,y)

    XX = X.reshape(-1,1)
    YY = Y.reshape(-1,1)
    XX = XX.squeeze()
    YY = YY.squeeze()

    TestData[:,0] = XX
    TestData[:,1] = YY
    TestData.requires_grad = True

    Pix_2 = 20
    x_2 = params["length"]*torch.arange(0,1+1/Pix_2,1/Pix_2)
    y_2 = torch.arange(0,1+1/Pix_2,1/Pix_2)
    TestData_2 = torch.zeros([(Pix_2+1)**2,params["width"]]).to(device)
    XXXX,YYYY = torch.meshgrid(x_2,y_2)
    XXXX = XXXX.reshape(-1,1)
    YYYY = YYYY.reshape(-1,1)
    TestData_2[:,0] = XXXX.squeeze()
    TestData_2[:,1] = YYYY.squeeze()

    model.train()
    model_1.train()
    model_2.train()
    model_3.train()

    data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
    data[:,0] = params["length"]*torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data.requires_grad = True

    bdrydata_left = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
    bdrydata_left[:,0] = torch.from_numpy(np.zeros(params["bdryBatch"]))
    bdrydata_left[:,1] = torch.from_numpy(np.random.rand(params["bdryBatch"]))

    bdrydata_right = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
    bdrydata_right[:,0] = bdrydata_left[:,0] + params["length"]
    bdrydata_right[:,1] = bdrydata_left[:,1]

    bdrydata_up = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
    bdrydata_up[:,0] = params["length"]*torch.from_numpy(np.random.rand(params["bdryBatch"]))
    bdrydata_up[:,1] = torch.from_numpy(np.zeros(params["bdryBatch"])) + 1

    bdrydata_down = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
    bdrydata_down[:,0] = bdrydata_up[:,0]
    bdrydata_down[:,1] = torch.from_numpy(np.zeros(params["bdryBatch"]))

    FF = open("loss.txt","w")
    EE = open("lr.txt","w")
    GG = open("lambda.txt","w")
    JJ = open("mass.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")

    Train_u = model(TestData,params,1002001,device)
    Train_u = Train_u.cpu().detach().numpy()

    Train_u1 = Train_u[:,0].reshape(Pix+1,Pix+1)
    Train_u2 = Train_u[:,1].reshape(Pix+1,Pix+1)
    Train_phi = Train_u[:,2].reshape(Pix+1,Pix+1)

    plt.figure()
    plt.imshow(Train_u1.T,cmap = 'viridis',origin='lower',aspect=2/3)
    cb = plt.colorbar(shrink=0.7)
    plt.show()

    plt.imshow(Train_u2.T,cmap = 'viridis',origin='lower',aspect=2/3)
    cb = plt.colorbar(shrink=0.7)
    plt.show()


    plt.imshow(Train_phi.T,cmap = 'viridis',origin='lower',aspect=2/3)
    cb = plt.colorbar(shrink=0.7)
    plt.show()


    phase_fluid = (Train_phi>0.5)+0
    plt.figure()
    plt.imshow(phase_fluid.T,cmap = 'viridis',origin='lower',aspect=2/3)
    cb = plt.colorbar(shrink=0.7)
    plt.show()

    Train_u = model(TestData_2,params,441,device)
    Train_u = Train_u.cpu().detach().numpy()
    Train_u1 = Train_u[:,0].reshape(Pix_2+1,Pix_2+1)
    Train_u2 = Train_u[:,1].reshape(Pix_2+1,Pix_2+1)

    plt.figure()
    plt.quiver(XXXX,YYYY,Train_u1,Train_u2,width = 0.001,scale = 3)
    plt.show()

    for step in range(params["trainStep"]):

        uxt = model(data,params,params["bodyBatch"],device)
        u1 = uxt[:,0].reshape(-1,1)
        u2 = uxt[:,1].reshape(-1,1)
        phi = uxt[:,2].reshape(-1,1)
        p = model_1(data).reshape(-1,1)

        left = model(bdrydata_left,params,params["bdryBatch"],device)
        u_left = left[:,0].reshape(-1,1)
        u2_left = left[:,1].reshape(-1,1)
        adv_left = model_3(bdrydata_left)
        adv_u_left = adv_left[:,0].reshape(-1,1)
        adv_u2_left = adv_left[:,1].reshape(-1,1)

        right = model(bdrydata_right,params,params["bdryBatch"],device)
        u_right = right[:,0].reshape(-1,1)
        u2_right = right[:,1].reshape(-1,1)
        adv_right = model_3(bdrydata_right)
        adv_u_right = adv_right[:,0].reshape(-1,1)
        adv_u2_right = adv_right[:,1].reshape(-1,1)

        up = model(bdrydata_up,params,params["bdryBatch"],device)
        u_up = up[:,0].reshape(-1,1)
        u2_up = up[:,1].reshape(-1,1)
        adv_up = model_3(bdrydata_up)
        adv_u_up = adv_up[:,0].reshape(-1,1)
        adv_u2_up = adv_up[:,1].reshape(-1,1)

        down = model(bdrydata_down,params,params["bdryBatch"],device)
        u_down = down[:,0].reshape(-1,1)
        u2_down = down[:,1].reshape(-1,1)
        adv_down = model_3(bdrydata_down)
        adv_u_down = adv_down[:,0].reshape(-1,1)
        adv_u2_down = adv_down[:,1].reshape(-1,1)


        u_bdry = torch.relu((1-(12*bdrydata_left[:,1].reshape(-1,1)-3)**2)) + torch.relu((1-(12*bdrydata_right[:,1].reshape(-1,1)-9)**2))\

        u_bdry = params["g_bar"]*u_bdry.reshape(-1,1)

        model.zero_grad()
        model_1.zero_grad()
        model_2.zero_grad()
        model_3.zero_grad()

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

        loss_bdry = torch.mean((u_left-u_bdry)**2)+torch.mean((u_right-u_bdry)**2)+torch.mean((u2_left)**2)+torch.mean((u2_right)**2)\
              + torch.mean((u_up)**2) + torch.mean((u2_up)**2) + torch.mean((u_down)**2) + torch.mean((u2_down)**2)
        loss3 = torch.mean((u_left-u_bdry)*adv_u_left) + torch.mean((u_right-u_bdry)*adv_u_right) + torch.mean((u2_right)*adv_u2_right)+torch.mean((u2_left)*adv_u2_left)\
              + torch.mean((u_up)*adv_u_up) + torch.mean((u2_up)*adv_u2_up)+ torch.mean((u_down)*adv_u_down) + torch.mean((u2_down)*adv_u2_down)

        loss_bdry = params["beta3"]*loss_bdry

        div_u = du1dxy[:,0]+du2dxy[:,1]

        volume = params["length"]*torch.mean(phi)-params["C"]

        J_alpha = params["length"]*(params["eta0"]*(0.5*torch.mean((grad_u1_2norm_2+grad_u2_2norm_2))+0.5*params["alpha0"]*torch.mean((1-phi)**2*u_2norm_2)))
        J_epsilon =  params["length"]*params["eta"]*(0.5*params["epsilon"]*torch.mean(grad_phi_2norm_2) + 1/(4*params["epsilon"])*torch.mean(phi**2*(1-phi)**2))
        zzz = torch.mean(div_u**2)
        Lag_p = -torch.mean(div_u*p) + params["beta1"]/2*zzz

        Lag_volume = params["beta2"]/2*volume**2

        loss = J_alpha +J_epsilon + Lag_p + Lag_volume + loss_bdry - loss3

        loss1 = -Lag_p
        loss2 = -Lag_volume

        if step%params["writeStep"] == params["writeStep"]-1:
          print("Error: %s , J_alpha: %s, J_epsilon: %s, Lag_p: %s, Lag_volume: %s, sum(div_u^2): %s, mean(abs(div_u)): %s, max(abs(div_u)): %s volume: %s at Step %s."\
                %(loss.cpu().detach().numpy(),J_alpha.cpu().detach().numpy(),J_epsilon.cpu().detach().numpy(),Lag_p.cpu().detach().numpy(),Lag_volume.cpu().detach().numpy(),\
                  zzz.cpu().detach().numpy(),torch.mean(torch.abs(div_u)).cpu().detach().numpy(),torch.max(torch.abs(div_u)).cpu().detach().numpy(),volume.cpu().detach().numpy(),step+1))

          file = open("loss.txt","a")
          file.write(str(loss)+"\n")
          file = open("mass.txt","a")
          file.write(str(volume.cpu().detach().numpy())+"\n")

        if step%params["plotStep"] == params["plotStep"]-1:
          Train_u = model(TestData,params,1002001,device)
          Train_u = Train_u.cpu().detach().numpy()

          Train_u1 = Train_u[:,0].reshape(Pix+1,Pix+1)
          Train_u2 = Train_u[:,1].reshape(Pix+1,Pix+1)
          Train_phi = Train_u[:,2].reshape(Pix+1,Pix+1)

          plt.figure()
          plt.imshow(Train_u1.T,cmap = 'viridis',origin='lower',aspect=2/3)
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u2.T,cmap = 'viridis',origin='lower',aspect=2/3)
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_phi.T,cmap = 'viridis',origin='lower',aspect=2/3)
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          phase_fluid = (Train_phi>0.5)+0
          plt.figure()
          plt.imshow(phase_fluid.T,cmap = 'viridis',origin='lower',aspect=2/3)
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow((Train_u2**2+Train_u1**2).T**0.5,cmap = 'viridis',origin='lower',aspect=2/3)
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          Train_u = model(TestData_2,params,441,device)
          Train_u = Train_u.cpu().detach().numpy()
          Train_u1 = Train_u[:,0].reshape(Pix_2+1,Pix_2+1)
          Train_u2 = Train_u[:,1].reshape(Pix_2+1,Pix_2+1)

          plt.figure()
          plt.quiver(XXXX,YYYY,Train_u1,Train_u2,width = 0.001,scale = 3)
          plt.show()

          plt.figure()
          plt.quiver(XXXX,YYYY,Train_u1,Train_u2,width = 0.001,scale = 6)
          plt.show()

          plt.figure()
          plt.quiver(XXXX,YYYY,Train_u1,Train_u2,width = 0.001,scale = 10)
          plt.show()

        if step%params["sampleStep"] == params["sampleStep"]-1:

          data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
          data[:,0] = params["length"]*torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
          data.requires_grad = True

          bdrydata_left = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
          bdrydata_left[:,0] = torch.from_numpy(np.zeros(params["bdryBatch"]))
          bdrydata_left[:,1] = torch.from_numpy(np.random.rand(params["bdryBatch"]))

          bdrydata_right = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
          bdrydata_right[:,0] = bdrydata_left[:,0] + params["length"]
          bdrydata_right[:,1] = bdrydata_left[:,1]

          bdrydata_up = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
          bdrydata_up[:,0] = params["length"]*torch.from_numpy(np.random.rand(params["bdryBatch"]))
          bdrydata_up[:,1] = torch.from_numpy(np.zeros(params["bdryBatch"])) + 1

          bdrydata_down = torch.zeros((params["bdryBatch"],params["width"])).float().to(device)
          bdrydata_down[:,0] = bdrydata_up[:,0]
          bdrydata_down[:,1] = torch.from_numpy(np.zeros(params["bdryBatch"]))

        if step%10 > 8:
          loss2.backward()
          optimizer_2.step()
          scheduler_2.step()

        elif step%10 > 7:
          loss1.backward()

          optimizer_1.step()
          scheduler_1.step()
        elif step%10 > 6:
          loss3.backward()
          optimizer_3.step()
          scheduler_3.step()
        else:
          loss.backward()

          optimizer.step()
          scheduler.step()

        params["beta1"] = params["beta1"]*params["alpha"]
        params["beta2"] = params["beta2"]*params["alpha"]
        params["beta3"] = params["beta3"]*params["alpha"]
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Parameters
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.01
    params["d"] = 2
    params["dd"] = 3
    params["bodyBatch"] = 10000 
    params["bdryBatch"] = 100 
    params["lr"] = 0.006
    params["lr2"] = 0.001
    params["width"] = 120 
    params["depth"] = 10 
    params["trainStep"] = 30000
    params["alpha"] = 1.0003
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 10
    params["milestone"] = [50,500,1500,2200,3000,3700,4500,6000,8000,10000,12000,14000,17000,19000,21000,23000,25000,27000,29000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001
    params["g_bar"] = 1
    params["length"] = 1.5
    params["alpha0"] = 100000
    params["eta0"] = 10
    params["eta"] = 100
    params["beta1"] = 1000
    params["beta2"] = 1000
    params["beta3"] = 10000
    params["C"] = params["length"]/3
    params["width_1"] = 30
    params["depth_1"] = 2
    params["dd_1"] = 1
    params["width_3"] = 30
    params["depth_3"] = 2
    params["dd_3"] = 2

    startTime = time.time()
    model = ResNet(params).to(device)
    model_1 = ResNet_1(params).to(device)
    model_2 = drrnn(1, 1, 1).to(device)
    model_3 = ResNet_3(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_1 = torch.optim.Adam(model_1.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler_1 = MultiStepLR(optimizer_1,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_2 = torch.optim.Adam(model_2.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler_2 = MultiStepLR(optimizer_2,milestones=params["milestone"],gamma=params["gamma"])
    optimizer_3 = torch.optim.Adam(model_3.parameters(),lr=params["lr2"],weight_decay=params["decay"])
    scheduler_3 = MultiStepLR(optimizer_3,milestones=params["milestone"],gamma=params["gamma"])

    print("Generating network costs %s seconds."%(time.time()-startTime))
    startTime = time.time()
    trainnew(model,device,params,optimizer,scheduler,model_1,optimizer_1,scheduler_1,model_2,optimizer_2,scheduler_2,model_3,optimizer_3,scheduler_3)  #trainnew model
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")



if __name__=="__main__":
    main()