import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import mpl_toolkits.mplot3d as p3d


class ResNet(torch.nn.Module):
    def __init__(self, params, device):
        super(ResNet, self).__init__()
        self.params = params
        self.device = device
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        xxx = x[:,0]

        yyy = x[:,1]

        a = x.size()[0]
        x = torch.zeros((a,self.params["width"])).float().to(self.device)


        x[:,0] = xxx
        x[:,1] = yyy


        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            # x_temp = (F.tanh(layer(x)))
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
    BB = open("mass_u4.txt","w")
    YY = open("mass_u5.txt","w")
    CC = open("mass_u6.txt","w")
    SS = open("mass_u7.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")

    for step in range(params["trainStep"]):

        uxt = model(data)
        u1 = uxt[:,0]
        u2 = uxt[:,1]
        u3 = uxt[:,2]
        u4 = uxt[:,3]
        u5 = uxt[:,4]
        u6 = uxt[:,5]
        u7 = uxt[:,6]

        yyy_2 = model_2(xxx_2).reshape(-1,1)
        model.zero_grad()

        du1dxy = torch.autograd.grad(u1,data,grad_outputs=torch.ones_like(u1),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du2dxy = torch.autograd.grad(u2,data,grad_outputs=torch.ones_like(u2),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du3dxy = torch.autograd.grad(u3,data,grad_outputs=torch.ones_like(u3),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du4dxy = torch.autograd.grad(u4,data,grad_outputs=torch.ones_like(u4),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du5dxy = torch.autograd.grad(u5,data,grad_outputs=torch.ones_like(u5),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du6dxy = torch.autograd.grad(u6,data,grad_outputs=torch.ones_like(u6),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        du7dxy = torch.autograd.grad(u7,data,grad_outputs=torch.ones_like(u7),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]


        grad_u1_2norm_2 = du1dxy[:,0]**2+du1dxy[:,1]**2
        grad_u1_2norm_2 = grad_u1_2norm_2.reshape(-1,1)
        grad_u2_2norm_2 = du2dxy[:,0]**2+du2dxy[:,1]**2
        grad_u2_2norm_2 = grad_u2_2norm_2.reshape(-1,1)
        grad_u3_2norm_2 = du3dxy[:,0]**2+du3dxy[:,1]**2
        grad_u3_2norm_2 = grad_u3_2norm_2.reshape(-1,1)
        grad_u4_2norm_2 = du4dxy[:,0]**2+du4dxy[:,1]**2
        grad_u4_2norm_2 = grad_u4_2norm_2.reshape(-1,1)
        grad_u5_2norm_2 = du5dxy[:,0]**2+du5dxy[:,1]**2
        grad_u5_2norm_2 = grad_u5_2norm_2.reshape(-1,1)
        grad_u6_2norm_2 = du6dxy[:,0]**2+du6dxy[:,1]**2
        grad_u6_2norm_2 = grad_u6_2norm_2.reshape(-1,1)
        grad_u7_2norm_2 = du7dxy[:,0]**2+du7dxy[:,1]**2
        grad_u7_2norm_2 = grad_u7_2norm_2.reshape(-1,1)

        mass_u1 = (torch.mean(u1**2)-params["volumn"])
        mass_u2 = (torch.mean(u2**2)-params["volumn"])
        mass_u3 = (torch.mean(u3**2)-params["volumn"])
        mass_u4 = (torch.mean(u4**2)-params["volumn"])
        mass_u5 = (torch.mean(u5**2)-params["volumn"])
        mass_u6 = (torch.mean(u6**2)-params["volumn"])
        mass_u7 = (torch.mean(u7**2)-params["volumn"])
        mass = torch.abs(mass_u1) + torch.abs(mass_u2) + torch.abs(mass_u3) + torch.abs(mass_u4) + torch.abs(mass_u5) + torch.abs(mass_u6) + torch.abs(mass_u7)

        loss = params["penalty2"]*(1/2*(torch.mean(grad_u1_2norm_2)+torch.mean(grad_u2_2norm_2)+torch.mean(grad_u3_2norm_2)+torch.mean(grad_u4_2norm_2)+torch.mean(grad_u5_2norm_2)+torch.mean(grad_u6_2norm_2)+torch.mean(grad_u7_2norm_2))\
                                   + (2/(params["epsilon"]**2))*torch.mean((u1*u2)**2+(u1*u3)**2+(u1*u4)**2+(u1*u5)**2+(u1*u6)**2+(u1*u7)**2\
                                   +(u2*u3)**2+(u2*u4)**2+(u2*u5)**2+(u2*u6)**2+(u2*u7)**2+(u3*u4)**2+(u3*u5)**2+(u3*u6)**2+(u3*u7)**2\
                                   +(u4*u5)**2+(u4*u6)**2+(u4*u7)**2+(u5*u6)**2+(u5*u7)**2+(u6*u7)**2))\
                                   - (yyy_2[0,:]*mass_u1+yyy_2[1,:]*mass_u2+yyy_2[2,:]*mass_u3+yyy_2[3,:]*mass_u4+yyy_2[4,:]*mass_u5+yyy_2[5,:]*mass_u6+yyy_2[6,:]*mass_u7)\
                                   +  params["penalty"]*(mass_u1**2+mass_u2**2+mass_u3**2+mass_u4**2+mass_u5**2+mass_u6**2+mass_u7**2)

        loss_2 =  (yyy_2[0,:]*mass_u1+yyy_2[1,:]*mass_u2+yyy_2[2,:]*mass_u3+yyy_2[3,:]*mass_u4+yyy_2[4,:]*mass_u5+yyy_2[5,:]*mass_u6+yyy_2[6,:]*mass_u7)

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
          file = open("mass_u4.txt","a")
          file.write(str(mass_u4.cpu().detach().numpy())+"\n")
          file = open("mass_u5.txt","a")
          file.write(str(mass_u5.cpu().detach().numpy())+"\n")
          file = open("mass_u6.txt","a")
          file.write(str(mass_u6.cpu().detach().numpy())+"\n")
          file = open("mass_u7.txt","a")
          file.write(str(mass_u7.cpu().detach().numpy())+"\n")

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
          Train_u4 = Train_u[:,3].reshape(Pix+1,Pix+1)
          Train_u5 = Train_u[:,4].reshape(Pix+1,Pix+1)
          Train_u6 = Train_u[:,5].reshape(Pix+1,Pix+1)
          Train_u7 = Train_u[:,6].reshape(Pix+1,Pix+1)

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

          plt.imshow(Train_u[:,3].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u[:,4].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u[:,5].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()

          plt.imshow(Train_u[:,6].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()



          plt.figure()
          plt.imshow(Train_u[:,1].reshape(Pix+1,Pix+1)+Train_u[:,0].reshape(Pix+1,Pix+1)+Train_u[:,2].reshape(Pix+1,Pix+1)\
                     +Train_u[:,3].reshape(Pix+1,Pix+1)+Train_u[:,4].reshape(Pix+1,Pix+1)+Train_u[:,5].reshape(Pix+1,Pix+1)\
                     +Train_u[:,6].reshape(Pix+1,Pix+1),cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()



          plt.figure()
          a12 = (Train_u1>Train_u2)+0
          a13 = (Train_u1>Train_u3)+0
          a14 = (Train_u1>Train_u4)+0
          a15 = (Train_u1>Train_u5)+0
          a16 = (Train_u1>Train_u6)+0
          a17 = (Train_u1>Train_u7)+0


          b21 = (Train_u2>Train_u1)+0
          b23 = (Train_u2>Train_u3)+0
          b24 = (Train_u2>Train_u4)+0
          b25 = (Train_u2>Train_u5)+0
          b26 = (Train_u2>Train_u6)+0
          b27 = (Train_u2>Train_u7)+0



          c31 = (Train_u3>Train_u1)+0
          c32 = (Train_u3>Train_u2)+0
          c34 = (Train_u3>Train_u4)+0
          c35 = (Train_u3>Train_u5)+0
          c36 = (Train_u3>Train_u6)+0
          c37 = (Train_u3>Train_u7)+0



          d41 = (Train_u4>Train_u1)+0
          d42 = (Train_u4>Train_u2)+0
          d43 = (Train_u4>Train_u3)+0
          d45 = (Train_u4>Train_u5)+0
          d46 = (Train_u4>Train_u6)+0
          d47 = (Train_u4>Train_u7)+0




          e51 = (Train_u5>Train_u1)+0
          e52 = (Train_u5>Train_u2)+0
          e53 = (Train_u5>Train_u3)+0
          e54 = (Train_u5>Train_u4)+0
          e56 = (Train_u5>Train_u6)+0
          e57 = (Train_u5>Train_u7)+0


          f61 = (Train_u6>Train_u1)+0
          f62 = (Train_u6>Train_u2)+0
          f63 = (Train_u6>Train_u3)+0
          f64 = (Train_u6>Train_u4)+0
          f65 = (Train_u6>Train_u5)+0
          f67 = (Train_u6>Train_u7)+0


          g71 = (Train_u7>Train_u1)+0
          g72 = (Train_u7>Train_u2)+0
          g73 = (Train_u7>Train_u3)+0
          g74 = (Train_u7>Train_u4)+0
          g75 = (Train_u7>Train_u5)+0
          g76 = (Train_u7>Train_u6)+0

          a12 = torch.from_numpy(a12)
          a13 = torch.from_numpy(a13)
          a14 = torch.from_numpy(a14)
          a15 = torch.from_numpy(a15)
          a16 = torch.from_numpy(a16)
          a17 = torch.from_numpy(a17)


          b21 = torch.from_numpy(b21)
          b23 = torch.from_numpy(b23)
          b24 = torch.from_numpy(b24)
          b25 = torch.from_numpy(b25)
          b26 = torch.from_numpy(b26)
          b27 = torch.from_numpy(b27)


          c31 = torch.from_numpy(c31)
          c32 = torch.from_numpy(c32)
          c34 = torch.from_numpy(c34)
          c35 = torch.from_numpy(c35)
          c36 = torch.from_numpy(c36)
          c37 = torch.from_numpy(c37)


          d41 = torch.from_numpy(d41)
          d42 = torch.from_numpy(d42)
          d43 = torch.from_numpy(d43)
          d45 = torch.from_numpy(d45)
          d46 = torch.from_numpy(d46)
          d47 = torch.from_numpy(d47)


          e51 = torch.from_numpy(e51)
          e52 = torch.from_numpy(e52)
          e53 = torch.from_numpy(e53)
          e54 = torch.from_numpy(e54)
          e56 = torch.from_numpy(e56)
          e57 = torch.from_numpy(e57)


          f61 = torch.from_numpy(f61)
          f62 = torch.from_numpy(f62)
          f63 = torch.from_numpy(f63)
          f64 = torch.from_numpy(f64)
          f65 = torch.from_numpy(f65)
          f67 = torch.from_numpy(f67)



          g71 = torch.from_numpy(g71)
          g72 = torch.from_numpy(g72)
          g73 = torch.from_numpy(g73)
          g74 = torch.from_numpy(g74)
          g75 = torch.from_numpy(g75)
          g76 = torch.from_numpy(g76)


          phase_u1 = 1*F.relu(a12+a13+a14+a15+a16+a17-5)
          phase_u2 = 2*F.relu(b21+b23+b24+b25+b26+b27-5)
          phase_u3 = 3*F.relu(c31+c32+c34+c35+c36+c37-5)
          phase_u4 = 4*F.relu(d41+d42+d43+d45+d46+d47-5)
          phase_u5 = 5*F.relu(e51+e52+e53+e54+e56+e57-5)
          phase_u6 = 6*F.relu(f61+f62+f63+f64+f65+f67-5)
          phase_u7 = 7*F.relu(g71+g72+g73+g74+g75+g76-5)



          phase = phase_u1+phase_u2+phase_u3+phase_u4+phase_u5+phase_u6+phase_u7
          phase = phase.cpu().detach().numpy()
          plt.imshow(phase,cmap = 'viridis',origin='lower')
          cb = plt.colorbar(shrink=0.7)
          plt.show()
          torch.save(model.state_dict(),"last_model.pt")


          file_name = f"last_model_{step}.pt"

          torch.save(model.state_dict(),file_name)

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


        params["penalty"] = params["penalty"]*params["alpha"]



def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 



def main():
    # Parameters
    torch.manual_seed(110)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.05
    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 7
    params["bodyBatch"] = 100000 
    params["lr"] = 0.016 
    params["width"] = 120 
    params["depth"] = 8
    params["trainStep"] = 40000
    params["penalty"] = 10000000 
    params["alpha"] = 1.0003
    params["volumn"] = 1
    params["penalty2"] = 5000
    params["writeStep"] = 50
    params["plotStep"] = 2000
    params["sampleStep"] = 40
    params["milestone"] = [1000,3000,5000,6000,8000,10000,14000,17000,19500,22500,24000,26000,28000,29000,33000,35000,37000,39000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = ResNet(params,device).to(device)
    model_2 = drrnn(1, 1, 7).to(device)
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