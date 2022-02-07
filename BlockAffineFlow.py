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
torch.set_printoptions(precision=4, sci_mode=False)

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
workpath = 'D:\/Coding\/SpectralNFWorkSpace'
# workpath = 'D:\/Study_Files\/UCSB\Projects\/SNFWorkSpace'

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

hidden_feature = 256
block_feature = 2

# Flow model parameters and training setting, including training iterations and validation intervals;
batch_size = 256
num_transformation = 1
num_iter = 20000
val_interval = 250
lr = 0.005

# Base distribution;
base_dist = distributions.StandardNormal(shape=[feature])

# Constructing transformation;
transform = []
for i in range(num_transformation):
    transform.append(
        transforms.BlockAffineTransformation(
            feature=feature,
            block_feature=block_feature,
            hidden_feature=hidden_feature,
            sigma_max=10,
            sigma_min=1
        )
    )
    transform.append(
        transforms.LULinear(features=feature)
    )


# Composition of the final transformation and flow model;
Transform = transforms.CompositeTransform(transform)
flow = flows.Flow(Transform, base_dist).to(device)

# Setup optimizer;
optimizer = optim.Adam(flow.parameters(), lr=lr)

# Main training part;
tbar = tqdm(range(num_iter))
train_loss = np.zeros(shape=(num_iter))
train_loss_kl = np.zeros(shape=(num_iter))
train_loss_reg = np.zeros(shape=(num_iter))
val_score = np.zeros(shape=(int(num_iter / val_interval)))

start = time.time()

count_val = 0

for i in tbar:
    # Training iterations;
    flow.train()
    batch = next(train_generator).to(device)
    optimizer.zero_grad()
    loss_kl = -flow.log_prob(inputs=batch).mean()
    loss_reg = transform[0].reg_error
    loss = loss_kl + 5 * loss_reg
    train_loss[i] = loss.detach().numpy()
    train_loss_kl[i] = loss_kl.detach().numpy()
    train_loss_reg[i] = loss_reg.detach().numpy()
    # o, lj = Transform(batch)
    # if torch.any(torch.isnan(loss_batch)) or torch.any(torch.isinf(loss_batch)):
    #     ab_index = (torch.logical_or(torch.isnan(loss_batch), torch.isinf(loss_batch)) == torch.tensor(True)).nonzero(
    #         as_tuple=True)[0]
    #     out_1 = transform[0](batch[ab_index])[0]
    #     slope_at_x = transform[1].mono_pwl_function.slope_at(out_1)
    #     slope_all = transform[1].mono_pwl_function.get_slopes()
    #     # slope_at_x_new = transform[1].mono_pwl_function.slope_at_new(out_1)
    #     plt.show()
    #     raise ValueError('Invalid loss detected')
    loss.backward()
    optimizer.step()

    # o, lj = Transform(batch)
    # if (torch.any(torch.isnan(o)) or torch.any(torch.isnan(lj))):
    #     raise ValueError('NAN detected')
    # print('Current transformation output is:\n', o)

    # Validation;
    if (i + 0) % val_interval == 0:
        print('Current total loss: {:.3f}'.format(train_loss[i]))
        print('Current KL loss: {:.3f}'.format(train_loss_kl[i]))
        print('Current reg loss: {:.3f}'.format(train_loss_reg[i]))
        flow.eval()
        avg_val_log_likelyhood = 0
        for val_batch in val_loader:
            avg_val_log_likelyhood += -flow.log_prob(inputs=val_batch.to(device)).mean()
        avg_val_log_likelyhood = avg_val_log_likelyhood / len(val_loader)
        val_score[count_val] = avg_val_log_likelyhood.cpu().detach().numpy()
        # print('Current val score:\n', val_score[count_val])
        count_val += 1

        # out_1 = transform[0](batch)[0]
        # slope_at_x = transform[1].mono_pwl_function.slope_at(out_1)
        # out2, slope = transform[1].mono_pwl_function(out_1)
        # print('1\n', slope[0])
        # print('2\n', slope_at_x[0])

        # print(transform[1].mono_pwl_function.get_slopes())

        # x = torch.zeros(2000, feature)
        # for z in range(feature):
        #     x[:, z] = torch.arange(-10, 10, 0.01)
        #
        # y = transform[1](x)[0]
        #
        # plt.plot(x[:, 0], y.detach().numpy()[:, 0], label='{}th act'.format(i))
        # plt.legend()

end = time.time()
elapsed_time = end - start

print('Total time:', elapsed_time)

# np.savetxt('C:\/Users\Yu\Desktop\/test\/train_loss.txt', train_loss, fmt='%f')
np.savetxt(summary_path, val_score, fmt='%f')

with open(summary_path, 'a') as fp:
    fp.write('Total_time: {}\n'.format(elapsed_time))

# Plot training loss;
plt.plot(train_loss)
plt.plot(train_loss_kl)
plt.plot(train_loss_reg)
plt.show()
