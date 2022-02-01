import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
import dataparser

from torch import optim
import transforms
import distributions
import flows
from torch.utils import data
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)
np.set_printoptions(suppress=True)

# Reproductivity;
np.random.seed(1137)
torch.manual_seed(114514)

# Setup device;
assert torch.cuda.is_available()
device = torch.device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)

# Setup datasets and batch size;
dataset_name = 'power'
train_batch_size = 256
val_batch_size = 256


# Setup working directory, please check the working directory is the one containing main.py before use;
# workpath = 'D:\/Coding\/SpectralNFWorkSpace'
workpath = 'D:\/Study_Files\/UCSB\Projects\/SNFWorkSpace'

os.environ['DATAROOT'] = workpath + '\datasets'
summary_path = workpath + '\/summary_{}.txt'.format(dataset_name)

# Construct data;
data_train = dataparser.load_dataset(dataset_name, split='train')
train_loader = data.DataLoader(data_train, batch_size=train_batch_size, shuffle=True, drop_last=True)
train_generator = dataparser.batch_generator(train_loader)

data_val = dataparser.load_dataset(dataset_name, split='val')
val_loader = data.DataLoader(data_val, batch_size=val_batch_size, shuffle=True, drop_last=True)

data_test = dataparser.load_dataset(dataset_name, split='test')
test_loader = data.DataLoader(data_test, batch_size=val_batch_size, shuffle=True, drop_last=True)

# Extract the dimension of the dataset as feature;
feature = data_train.dim
print('Dimension of dataset:', feature)

# Flow model parameters and training setting, including training iterations and validation intervals;
batch_size = 64
num_transformation = 4
num_breakpoints = 20
num_iter = 5000
val_interval = 250
lr = 0.001

# Base distribution;
base_dist = distributions.StandardNormal(shape=[feature])

# Constructing transformation;
transform = []
for i in range(num_transformation):
    transform.append(
        transforms.LULinear(features=feature)
    )
    transform.append(
        transforms.PWLTransformation(feature=feature, num_breakpoints=num_breakpoints)
    )


# Composition of the final transformation and flow model;
Transform = transforms.CompositeTransform(transform)
flow = flows.Flow(Transform, base_dist).to(device)

# Setup optimizer;
optimizer = optim.Adam(flow.parameters(), lr=lr)

# Main training part;
tbar = tqdm(range(num_iter))
train_loss = np.zeros(shape=(num_iter))
val_score = np.zeros(shape=(int(num_iter / val_interval)))

start = time.time()

count_val = 0

for i in tbar:
    # Training iterations;
    flow.train()
    batch = next(train_generator).to(device)
    optimizer.zero_grad()
    loss_batch = -flow.log_prob(inputs=batch)
    loss = loss_batch.mean()
    train_loss[i] = loss.detach().numpy()
    o, lj = Transform(batch)
    if torch.any(torch.isnan(loss_batch)) or torch.any(torch.isinf(loss_batch)):
        ab_index = (torch.logical_or(torch.isnan(loss_batch), torch.isinf(loss_batch)) == torch.tensor(True)).nonzero(
            as_tuple=True)[0]
        raise ValueError('Invalid loss detected')
    loss.backward()
    optimizer.step()

    # o, lj = Transform(batch)
    # if (torch.any(torch.isnan(o)) or torch.any(torch.isnan(lj))):
    #     raise ValueError('NAN detected')
    # print('Current transformation output is:\n', o)

    # Validation;
    if (i + 0) % val_interval == 0:
        print('Current loss:', train_loss[i])
        flow.eval()
        avg_val_log_likelyhood = 0
        for val_batch in val_loader:
            avg_val_log_likelyhood += -flow.log_prob(inputs=val_batch.to(device)).mean()
        avg_val_log_likelyhood = avg_val_log_likelyhood / len(val_loader)
        val_score[count_val] = avg_val_log_likelyhood.cpu().detach().numpy()
        # print('Current val score:\n', val_score[count_val])
        count_val += 1

        # print(transform[1].mono_pwl_function.get_slopes())

        # # Monitor the log det of each linear layer;
        # logdetlayers = torch.zeros(num_transformation)
        # output = val_batch
        # for i in range(num_hidden_layer + 2):
        #     output, logdetlayer = transform[i].forward(output)
        #     logdetlayers[i] = logdetlayer.mean()
        # print('logdet of layers:\n', logdetlayers)
        #
        # o, lj = Transform(val_batch)
        # print('Current transformation output is:\n',  lj.mean())

end = time.time()
elapsed_time = end - start

print('Total time:', elapsed_time)

# np.savetxt('C:\/Users\Yu\Desktop\/test\/train_loss.txt', train_loss, fmt='%f')
np.savetxt(summary_path, val_score, fmt='%f')

with open(summary_path, 'a') as fp:
    fp.write('Total_time: {}\n'.format(elapsed_time))

# Plot training loss;
plt.plot(train_loss)
plt.show()
