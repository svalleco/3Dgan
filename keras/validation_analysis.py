#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py

num_events=1000
with h5py.File('/eos/user/g/gkhattak/FixedAngleData/EleEscan_1_1.h5', 'r') as h5f:
    data = h5f['ECAL'][:num_events]
    target = h5f['target'][:num_events, 1]  # E_p

with h5py.File('gen_imgs.h5', 'r') as h5f:
    gen_imgs = h5f['gen_imgs'][()]

gen_imgs = np.squeeze(gen_imgs)
ecal = np.sum(data, axis=(1,2,3))
ecal_gen = np.sum(gen_imgs, axis=(1,2,3))

real_std_vec = []
fake_std_vec = []
real_mean_vec = []
fake_mean_vec = []

for i in range(gen_imgs.shape[0]):
    fake = gen_imgs[i]
    fake_std_vec.append(np.std(fake))
    fake_mean_vec.append(np.mean(fake))
    
for i in range(data.shape[0]):
    real = data[i]
    real_std_vec.append(np.std(real))
    real_mean_vec.append(np.mean(real))
    
plt.figure(0)
plt.hist(np.array(real_std_vec), bins=100, label='std(real)')
plt.hist(np.array(fake_std_vec), bins=100, label='std(fake)')
plt.xlabel('Standard deviation energy [GeV]')
plt.ylabel('Count')
plt.legend()
plt.savefig('std_real_fake.pdf')

plt.figure(1)
plt.hist(np.array(real_mean_vec), bins=100, label='mean(real)')
plt.hist(np.array(fake_mean_vec), bins=100, label='mean(fake)')
plt.xlabel('Mean energy [GeV]')
plt.ylabel('Count')
plt.legend()
plt.savefig('mean_real_fake.pdf')

plt.figure(3)
plt.hist(ecal, bins='auto', label='ecal sum (real)')
plt.hist(ecal_gen, bins='auto', label='ecal sum (fake)')
plt.xlabel('Ecal sum [GeV]')
plt.ylabel('Count')
plt.legend()
plt.savefig('ecal_real_fake.pdf')

