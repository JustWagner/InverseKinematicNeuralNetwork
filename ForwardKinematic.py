import numpy as np
from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt


# Forward kinematic for a 2 segment continuum robot
# TendonAngles is an array of 2 angles, one denotes the direction in which the base segment bends, the other in which the second segment bend compared to the base segment
# TendonCurv is an array of 2 curvature which indicate how much each tendon is bend.
# Curvature is defined as 1/r of the circle. This means that if a segment has the length of 5, a curvature of pi/10 would be a quarter circle segment.
# Tendon length is an array of two lengths of each segment.
# For this example we assume that the bending of one segment does not impact the other segment.a
def ForwardKin2Segments(TendonAngles, TendonCurv, TendonLength):
    x = 0
    y = 0
    z = 0
    YRotationDegree = 0
    ZRotationDegree = 0

    if TendonCurv[1] == 0:
        z = z + TendonLength[1]
    else:
        x = x + (np.cos(TendonLength[1]) / TendonCurv[1]) * (1 - np.cos(TendonCurv[1] * TendonLength[1]))
        y = y + (np.sin(TendonAngles[1]) / TendonCurv[1]) * (1 - np.cos(TendonCurv[1] * TendonLength[1]))
        z = z + np.sin(TendonCurv[1] * TendonLength[1]) / TendonCurv[1]



    YRotationDegree = TendonLength[0] * TendonCurv[0]
    x2 = x * np.cos(YRotationDegree) + z * np.sin(YRotationDegree)
    z = x * -np.sin(YRotationDegree) + z * np.cos(YRotationDegree)

    ZRotationDegree = TendonAngles[0]
    x = x2 * np.cos(ZRotationDegree) + y * -np.sin(ZRotationDegree)
    y = x2 * np.sin(ZRotationDegree) + y * np.cos(ZRotationDegree)
    
    
    if (TendonCurv[0] == 0):
        z = z + TendonLength[0]
    else:
        x = x + (np.cos(TendonAngles[0]) / TendonCurv[0]) * (1 - np.cos(TendonCurv[0] * TendonLength[0]))
        y = y + (np.sin(TendonAngles[0]) / TendonCurv[0]) * (1 - np.cos(TendonCurv[0] * TendonLength[0]))
        z = z + np.sin(TendonCurv[0] * TendonLength[0]) / TendonCurv[0]

  
    return x,y,z





#Initialize Input for the forward kinematic. length remains constant
TendonLength = [5,5]
SampleNumber = 100000
Input = []
Output = []
#Sampling random Curvatures and Angles. The values are then used to generate matching coordinates using the forward kinematic
for i in range(SampleNumber):
    curv = np.random.rand(2) * np.pi / 5
    angle = (np.random.rand(2) * np.pi) - np.pi
    x,y,z = ForwardKin2Segments(angle,curv,TendonLength)
    Input.append([curv[0],curv[1],angle[0],angle[1]])
    Output.append([x,y,z])
#print(Input)
#print(" ")
#print(Output)


#The network architecture of the Inverse kinematic
class InverseKinematic(nn.Module):
    def __init__(self):
        super(InverseKinematic, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(3, 1024),
                                       nn.LeakyReLU(0.5),
                                       nn.Linear(1024, 1024),
                                       nn.LeakyReLU(0.5),
                                       nn.Linear(1024, 4))
    def forward(self, x):
        output = self.regressor(x)
        return output
    

#Learn Rate Epochs and Batch
LR = 1e-6
MAX_EPOCH = 60
BATCH_SIZE = 512
#Selecting Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Splitting the data into Training and Testing. 
#The output of the forward kinematic is used as input for the model and the Curvatures and Angles are the prediction targets.
In_train, In_val, Out_train, Out_val = map(torch.tensor, train_test_split(Output, Input, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(In_train.unsqueeze(1), Out_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(In_val.unsqueeze(1), Out_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model = InverseKinematic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="mean")

print(In_train)
print("")
print(Out_train)


# training loop
train_loss_list = list()
val_loss_list = list()
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch+1, MAX_EPOCH))
    model.train()
    temp_loss_list = list()
    for In_train, Out_train in train_dataloader:
        In_train = In_train.type(torch.float32).to(device)
        Out_train = Out_train.type(torch.float32).to(device)
        
        optimizer.zero_grad()
       
        score = model(In_train)
     
        loss = criterion(input=score, target=Out_train)
        
       
        #loss.requires_grad = True
      
        loss.backward()
        
        optimizer.step()

        temp_loss_list.append(loss.detach().cpu().numpy())
    
    temp_loss_list = list()
    for In_train, Out_train in train_dataloader:
        In_train = In_train.type(torch.float32).to(device)
        Out_train = Out_train.type(torch.float32).to(device)

        score = model(In_train)
        loss = criterion(input=score, target=Out_train)

        temp_loss_list.append(loss.detach().cpu().numpy())
    
    train_loss_list.append(np.average(temp_loss_list))

    # validation
    model.eval()
    
    temp_loss_list = list()
    for In_val, Out_val in val_dataloader:
        In_val = In_val.type(torch.float32).to(device)
        Out_val = Out_val.type(torch.float32).to(device)

        score = model(In_val)
        loss = criterion(input=score, target=Out_val)

        temp_loss_list.append(loss.detach().cpu().numpy())
    
    val_loss_list.append(np.average(temp_loss_list))

    print("\ttrain loss: %.5f" % train_loss_list[-1])
    print("\tval loss: %.5f" % val_loss_list[-1])


#Plotting results
fig, ax = plt.subplots()
ax.plot(val_loss_list)

ax.set(xlabel='Loss', ylabel='Epoch',
       title='ValidationLoss')
ax.grid()

# fig.savefig("test.png")
plt.show()
#Simply save the model like this
#torch.save(model.state_dict())