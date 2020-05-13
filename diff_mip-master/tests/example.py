from utils import cplex_utils, experiment_utils
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sympy
from mip_solvers import MIPFunction
import cplex

# Hyper-parameters
input_size = 1
output_size = 4

learning_rate = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        # from IPython import embed; import sys; embed(); sys.exit(1)
        # softmax = torch.nn.Softmax(dim=1)
        # return softmax(self.lin(x) + self.net(x))
        return self.lin(x)

def item2name(item):
    return f"x{item}"

net = Net()

#from IPython import embed
#embed()

num_epochs = 10

# Toy dataset
x_train = np.array([[1], [2], [3], [4], [5], [6], 
                    [7], [8], [9], [10]], dtype=np.float32)

#y_train = np.array([[11,1.3,0,0],[22,2.1,0,0],[31,2.8,0,0],[38,4.2,0,0],[53,4.9,0,0],[62,6.1,0,0],[79,7.1,0,0],[81,8.1,0,0],[92,8.9,0,0],[99,10,0,0]], dtype=np.float32)
y_train = np.array([[1.3,11,0.6,0.5],[2.1,22,0.7,0.4],[2.8,31,0.8,0.2],[4.2,38,0.1,0],[4.9,53,0.5,0.3],[6.1,62,0.5,0.3],[7.1,79,0.4,0.2],[8.1,81,0.8,0.3],[8.9,92,1.1,0.1],[10,99,0.4,0.1]], dtype=np.float32)
#y_train = np.array([[1.3,11,0.5,0.6],[2.1,22,0.7,0.4],[31,2.8,0.8,0.2],[4.2,38,0.1,0.0],[4.9,53,0.3,0.5],[6.1,62,0.3,0.5],[7.1,79,0.2,0.4],[8.1,81,0.3,0.8],[8.9,92,0.1,1.1],[10,99,0.1,0.4]], dtype=np.float32)


# Loss and optimizer
criterion = nn.MSELoss()

##optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

cpx = cplex.Cplex()
num_items = 4


cpx.variables.add(types=["I" for _ in range(num_items)],
                  names=[item2name(i) for i in range(num_items)],
                  lb=[0.0 for _ in range(num_items)],
                  ub=[1.0 for _ in range(num_items)])

# add budget constraint
cpx.linear_constraints.add(names=["in"],
                               senses=["L"],
                               rhs=[1])

# make variable coefficients in budget constraint 1 (unit weighted)
cpx.linear_constraints.set_linear_components("in",
                                                 [[item2name(i) for i in range(num_items)],
                                                  [1.0, 1.0, 0.0,0.0]])

# add budget constraint
cpx.linear_constraints.add(names=['e1'],
                               senses=["E"],
                               rhs=[1])

# make variable coefficients in budget constraint 1 (unit weighted)
cpx.linear_constraints.set_linear_components('e1',
                                                 [[item2name(i) for i in range(num_items)],
                                                   [0.0, 0.0, 1.0, 1.0]])

#cpx.objective.set_sense(cpx.objective.sense.maximize)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# get problem specification
c, G, h, A, b, var_type = cplex_utils.cplex_to_matrices(cpx)


# preprocess A to remove linearly independent rows (Bryan code)
_, inds = sympy.Matrix(A).T.rref()


A = A[np.array(inds)]
b = b[np.array(inds)]
G = torch.from_numpy(G)
h = torch.from_numpy(h)
A = torch.from_numpy(A)
b = torch.from_numpy(b)

Q = 1e-6 * torch.eye(A.shape[1])
Q = Q.type_as(G)



# Train the model
for epoch in range(num_epochs):

    optimizer.zero_grad()
    for ii in range(0, 10):
        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(x_train[ii])
        c_true = torch.from_numpy(y_train[ii][:])
        
        # Forward pass
        pred_coefs = net(inputs)

  
        mip_function = MIPFunction(var_type, G, h, A, b, verbose=0)
        x = mip_function(Q, pred_coefs.flatten(), G, h, A, b)


        loss = x @ -c_true.double()
    
        true_loss = -np.array([0,1,1,0]).dot(np.array(y_train[ii][:]))

        print ('Epoch [{}/{}], Loss: {:.4f}, True Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(),true_loss))

        loss.backward()
        mip_function.release()
    
    optimizer.step()
        #loss.backward()

     


# Plot the graph
predicted = net(torch.from_numpy(x_train)).detach().numpy()

plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(net.state_dict(), 'model.ckpt')
