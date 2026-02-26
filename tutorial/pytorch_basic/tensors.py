import torch
import numpy as np

#############
# Basic setup
#############

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Tensor from np array: \n {x_np} \n")

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple, it represent 2 rows and 3 columns
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}") # by default will use cpu 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

tensor = torch.rand(3,4).to(device) # you can explicitly move tensor to device (in this case cuda)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

##########################
# Operations
##########################

# This is another way to move it to the GPU
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    
# Standard numpy-like indexing and slicing:
tensor = torch.rand(4, 4)
print(f"The random tensor: {tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1) # dim 1 concat columns dim 0 would be rows
print(f"Joining tensors (random tensor joined 3 times): {t1}")

# example 
# tensor = torch.tensor([
#  [1, 2],
#  [3, 4]
#])  # shape: (2, 2)

# looks like this
# [
#  [1, 2],
#  [3, 4]
# ]
# 
#  t1 = torch.cat([tensor, tensor, tensor], dim=1) # concat colunms
#  
# [
#  [1, 2, 1, 2, 1, 2],
#  [3, 4, 3, 4, 3, 4]
# ]
# 
#  t1 = torch.cat([tensor, tensor, tensor], dim=0) # concat rows
# [
#  [1, 2],
#  [3, 4],
#  [1, 2],
#  [3, 4],
#  [1, 2],
#  [3, 4]
# ]
# 

#######################
# Arithmetic operations
#######################

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z2)
print(z3)

# Single-element tensors If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():
#
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to tensor

n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")