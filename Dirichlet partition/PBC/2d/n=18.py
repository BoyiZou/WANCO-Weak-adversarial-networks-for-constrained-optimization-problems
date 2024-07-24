import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import mpl_toolkits.mplot3d as p3d
!pip install deepxde
import deepxde.geometry.sampler as XDE_sampler



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

        X_1 = x[:,0]
        X_2 = x[:,1]
        a = x.size()[0]
        x = torch.zeros((a,self.params["width"])).float().to(self.device)

        x[:,0] = torch.sin(torch.pi*2*X_1)
        x[:,1] = torch.cos(torch.pi*2*X_1)
        x[:,2] = torch.sin(torch.pi*2*X_2)
        x[:,3] = torch.cos(torch.pi*2*X_2)

        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)

        x = F.relu(x)

        return x



class Block(nn.Module):

    def __init__(self, in_N, width, out_N):
        super(Block, self).__init__()
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x))**3))**3 + x



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
    model_2.train()

    num_samples = params["bodyBatch"]  
    num_dimensions = 2  


    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0:2] = torch.from_numpy(XDE_sampler.sample(num_samples, num_dimensions, sampler= 'Hammersley'))

    data.requires_grad = True

    FF = open("loss.txt","w")
    EE = open("lr.txt","w")
    GG = open("lambda.txt","w")
    KK = open("beta.txt","w")
    II = open("penalty2.txt","w")

    for i in range(params["dd"]):
      filename = f"mass_u{i}.txt"
      MM = open(filename, 'w')

    for step in range(params["trainStep"]):


      uxt = model(data)
      u_dict = {}
      for i in range(params["dd"]):
        u_dict[f"u{i}"] = uxt[:,i].reshape(-1,1)


      lambda_value = model_2(xxx_2).reshape(-1,1)
      lambda_dict = {}
      for i in range(params["dd"]):
        lambda_dict[f"lambda{i}"] = lambda_value[i,0].reshape(-1,1)


      model.zero_grad()
      model_2.zero_grad()

      gradient_u_dict = {}
      gradient_norm_2_dict = {}
      mass_dict = {}
      mass = 0
      loss_1 = 0
      loss_4 = 0

      for i in range(params["dd"]):
        gradient_u_dict[f"du{i}dxy"] = torch.autograd.grad(u_dict[f"u{i}"],data,grad_outputs=torch.ones_like(u_dict[f"u{i}"]),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        gradient_norm_2_dict[f"gradient_u{i}_norm2"] = gradient_u_dict[f"du{i}dxy"][:,0]**2 + gradient_u_dict[f"du{i}dxy"][:,1]**2
        gradient_norm_2_dict[f"gradient_u{i}_norm2"] = gradient_norm_2_dict[f"gradient_u{i}_norm2"].reshape(-1,1)
        mass_dict[f"mass_u{i}"] = (torch.mean(u_dict[f"u{i}"]**2)-params["volumn"])
        mass += torch.abs(mass_dict[f"mass_u{i}"])

        loss_1 += torch.mean(gradient_norm_2_dict[f"gradient_u{i}_norm2"])
        loss_4 += mass_dict[f"mass_u{i}"]**2

      loss_2 = 0
      loss_3 = 0
      sum_result = 0
      for i in range(params["dd"]):
        for j in range(i+1,params["dd"]):
          sum_result += (u_dict[f"u{i}"]*u_dict[f"u{j}"])**2
        loss_2 += lambda_dict[f"lambda{i}"]*mass_dict[f"mass_u{i}"]

      loss_3 = torch.mean(sum_result)

      loss = params["penalty2"]*(0.5*loss_1 + 2/(params["epsilon"]**2) * loss_3) - loss_2 + params["penalty"]*loss_4



      if step%params["writeStep"] == params["writeStep"]-1:
        print("Error at Step %s is %s."%(step+1,loss))
        file = open("loss.txt","a")
        file.write(str(loss.cpu().detach().numpy())+"\n")
        file = open("lr.txt","a")
        file.write(str(scheduler.get_last_lr())+"\n")

        file = open("lambda.txt","a")
        file.write(str(lambda_dict)+"\n")

        file = open("beta.txt","a")
        file.write(str(params["penalty"])+"\n")
        file = open("penalty2.txt","a")
        file.write(str(params["penalty2"])+"\n")
        for i in range(params["dd"]):


          file = open(f"mass_u{i}.txt","a")
          file.write(str(mass_dict[f"mass_u{i}"].cpu().detach().numpy())+"\n")

      if step%params["plotStep"] == params["plotStep"]-1:
        Test_u = model(TestData)
        Test_u = Test_u.cpu().detach().numpy()
        Test_u_dict = {}

        Test_u_total = 0
        for i in range(params["dd"]):
          Test_u_dict[f"u{i}"] = Test_u[:,i]
          Test_u_total +=  Test_u_dict[f"u{i}"]


        if (params["dd"]) <= 3:
          for i in range(params["dd"]):
            plt.figure()

            plt.imshow(Test_u_dict[f"u{i}"].reshape(Pix+1,Pix+1).T,cmap = 'viridis',origin='lower')
            cb = plt.colorbar(shrink=0.7)
            plt.show()

        plt.figure()
        plt.imshow(Test_u_total.reshape(Pix+1,Pix+1).T,cmap = 'viridis',origin='lower')
        cb = plt.colorbar(shrink=0.7)
        plt.show()


        test_u_ind = 0
        indication_function = modify_function_values(Test_u)
        for i in range(params["dd"]):
          test_u_ind += i*indication_function[:,i]


        plt.imshow(test_u_ind.reshape(Pix+1,Pix+1).T,cmap = 'viridis',origin='lower')
        cb = plt.colorbar(shrink=0.7)
        plt.show()

        file_name = f"last_model_{step}.pt"

        torch.save(model.state_dict(),file_name)

      if step%4 > 2:
        loss_2.backward()

        optimizer_2.step()
        scheduler_2.step()
      else:
        loss.backward()

        optimizer.step()
        scheduler.step()

      if step < 15000:
        params["penalty"] = params["penalty"]*params["alpha"]

def modify_function_values(f):

    m, n = f.shape

    modified_f = np.zeros_like(f)

    max_indices = np.argmax(f,axis=1)

    modified_f[np.arange(m),max_indices] = 1

    return modified_f

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 



def main():
    # Parameters
    torch.manual_seed(5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.04
    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 18
    params["bodyBatch"] = 20000 
    params["lr"] = 0.006 
    params["lr2"] = 0.032 
    params["width"] = 50 
    params["depth"] = 3 
    params["trainStep"] = 20000
    params["penalty"] = 700000 
    params["alpha"] = 1.0003
    params["volumn"] = 1
    params["penalty2"] = 100
    params["writeStep"] = 50
    params["plotStep"] = 2000
    params["sampleStep"] = 40
    params["milestone"] = [10000,20000,30000,40000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = ResNet(params,device).to(device)
    model_2 = drrnn(1, 1, params["dd"]).to(device)
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