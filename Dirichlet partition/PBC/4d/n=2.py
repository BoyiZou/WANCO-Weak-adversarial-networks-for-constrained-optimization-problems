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
        X_3 = x[:,2]
        X_4 = x[:,3]

        a = x.size()[0]
        x = torch.zeros((a,self.params["width"])).float().to(self.device)

        x[:,0] = torch.sin(torch.pi*2*X_1)
        x[:,1] = torch.cos(torch.pi*2*X_1)
        x[:,2] = torch.sin(torch.pi*2*X_2)
        x[:,3] = torch.cos(torch.pi*2*X_2)
        x[:,4] = torch.sin(torch.pi*2*X_3)
        x[:,5] = torch.cos(torch.pi*2*X_3)
        x[:,6] = torch.sin(torch.pi*2*X_4)
        x[:,7] = torch.cos(torch.pi*2*X_4)

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

    Pix = 50
    n_points = Pix
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    z = np.linspace(0, 1, n_points)

    XX,YY,ZZ = torch.from_numpy(np.array(np.meshgrid(x, y, z)))

    TestData = torch.zeros([(n_points)**3,params["d"]]).to(device)
    TestData[:,0:1] = XX.reshape(-1,1)
    TestData[:,1:2] = YY.reshape(-1,1)
    TestData[:,2:3] = ZZ.reshape(-1,1)

    TestData_2 = TestData.clone()
    TestData_3 = TestData.clone()
    TestData_4 = TestData.clone()

    TestData_2[:,3] = 0.25
    TestData_3[:,3] = 0.5
    TestData_4[:,3] = 0.75

    TestData.requires_grad = True
    TestData_2.requires_grad = True
    TestData_3.requires_grad = True
    TestData_4.requires_grad = True

    X = TestData[:,0].cpu().detach().numpy()
    Y = TestData[:,1].cpu().detach().numpy()
    Z_flatten = TestData[:,2].flatten().cpu().detach().numpy()

    xxx_2 = torch.zeros((1,1)).float().to(device)

    model.train()
    model_2.train()

    num_samples = params["bodyBatch"] 
    num_dimensions = 4  


    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0:4] = torch.from_numpy(XDE_sampler.sample(num_samples, num_dimensions, sampler= 'Hammersley'))

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
        gradient_norm_2_dict[f"gradient_u{i}_norm2"] = gradient_u_dict[f"du{i}dxy"][:,0]**2 + gradient_u_dict[f"du{i}dxy"][:,1]**2 + gradient_u_dict[f"du{i}dxy"][:,2]**2 + gradient_u_dict[f"du{i}dxy"][:,3]**2
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
        for jjj in [0,0.25,0.5,0.75]:

          if jjj == 0:
            TestData_here = TestData
          elif jjj == 0.25:
            TestData_here = TestData_2
          elif jjj == 0.5:
            TestData_here = TestData_3
          elif jjj == 0.75:
            TestData_here = TestData_4



          Test_u = model(TestData_here)
          Test_u = Test_u.cpu().detach().numpy()
          Test_u_dict = {}

          Test_u_total = 0
          for i in range(params["dd"]):
            Test_u_dict[f"u{i}"] = Test_u[:,i]
            Test_u_total +=  Test_u_dict[f"u{i}"]

          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')

          ax.scatter(X, Y, Z_flatten, c=Test_u_total, cmap='jet')

          ax.set_xlabel('X')
          ax.set_ylabel('Y')
          ax.set_zlabel('Z')

          plt.title('u')

          plt.show()



          test_u_ind = 0
          indication_function = modify_function_values(Test_u)
          for i in range(params["dd"]):
            test_u_ind += i*indication_function[:,i]


          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')

          ax.scatter(X, Y, Z_flatten, c=test_u_ind.reshape(Pix,Pix,Pix), cmap='jet')

          ax.set_xlabel('X')
          ax.set_ylabel('Y')
          ax.set_zlabel('Z')

          plt.title('u')

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

      if step < 10000:
        params["penalty"] = params["penalty"]*params["alpha"]

def modify_function_values(f):

    m,n = f.shape

    modified_f = np.zeros_like(f)

    max_indices = np.argmax(f,axis=1)

    modified_f[np.arange(m),max_indices] = 1

    return modified_f

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 


def main():
    # Parameters
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["epsilon"] = 0.04
    params["radius"] = 1
    params["d"] = 4
    params["dd"] = 2
    params["bodyBatch"] = 60000 
    params["lr"] = 0.006 
    params["lr2"] = 0.01 
    params["width"] = 200 
    params["depth"] = 3 
    params["trainStep"] = 40000
    params["penalty"] = 15000  
    params["alpha"] = 1.0003
    params["volumn"] = 1
    params["penalty2"] = 10
    params["writeStep"] = 50
    params["plotStep"] = 2000
    params["sampleStep"] = 50
    params["milestone"] = [500,1500,3000,4000,5000,6000,7000,8000,9000,11000,13000,15000,17000,19000,25000,35000,45000,55000,60500,75000,85000,95000]
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