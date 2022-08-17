# %%

import neuron
from neuron import h, rxd, gui2

print(neuron.__version__)

from neuron import h, rxd, gui
from neuron.units import ms, mV
import textwrap

# %% md

# Step 2: Create a cell

# %% md

##Aside 1: NEURON's h.topology function
NEURON
's h.topology() function displays the topological structure of the entire model, indicating which sections are connected to which sections, where they are connected, and how many segments each section is divided into.

# %% md

load
library

# %%

import numpy as np
import matplotlib.pyplot as plt

image = np.load('data_from_19513019.npy')

# %%

print(image.shape)
x = image[:, 0]
y = image[:, 1]
plt.plot(x, y)
plt.show()

# %% md

# Our model

# %%

h.load_file('stdrun.hoc')

# %%

soma = h.Section(name='soma')
soma.L = 20
soma.diam = 20
soma.insert('hh')
dendrite = h.Section(name='dendrite')
dendrite.L = 500
dendrite.diam = 0.5
dendrite.nseg = 10
dendrite.insert('pas')
dendrite_1 = h.Section(name='dendrite_1')
dendrite_1.L = 100
dendrite_1.diam = 0.5
dendrite_1.nseg = 10
dendrite_1.insert('pas')
dendrite_2 = h.Section(name='dendrite_2')
dendrite_2.L = 100
dendrite_2.diam = 0.5
dendrite_2.nseg = 10
dendrite_2.insert('pas');

# %% md

Connection

# %%

dendrite.connect(soma, 1, 0)
dendrite_1.connect(dendrite, 1, 1)
dendrite_2.connect(dendrite, 1, 1);

# %%

h.topology()

# %% md

Insert
an
alpha
synaspe

# %% md

Alpha
synaspe

# %% md

h.load_file('stdrun.hoc')

s = h.NetStim()

# %%

asyn = h.Exp2Syn(dendrite(0.5))
asyn.tau1 = 2
asyn.tau2 = 2
asyn.e = 30
asyn.i = 100
s.interval = 50
s.number = 5
s.noise = 0.1
s.start = 0

stim = h.NetCon(s, asyn)
stim.delay = 100

stim.weight[0] = 0.3

h.finitialize(-65 * mV)

v_vec = h.Vector()  # Membrane potential vector
t_vec = h.Vector()  # Time stamp vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

h.tstop = 500
h.run()

plt.figure(figsize=(8, 4))  # Default figsize is (8,6)
plt.plot(t_vec, v_vec)

plt.xlabel('time (ms)')
plt.xlim(0, 500)
plt.ylabel('mV')
plt.show()
plt.plot(x, y, color='g')
plt.show()
print(y)

# %%

x1 = image[1000:, 0]
y1 = image[1000:, 1]
print(x1)
print(y1)

# %%

v = h.Vector(2)

print(v)


def efun(v):
    return (v.x1[0] + v.x1[1] - 5) ** 2 + 5 * (v.x1[0] - v.x1[1] - 15) ** 2(v[0] + v[1]) ** 2 + (v[0] - v[1]) ** 2


h.attr_praxis(1e-5, 0.5, 0)
e = h.fit_praxis(efun, v)
print("e=%g x=%g y=%g\n" % (e, v[0], v[1]))

# %%


# %%

import numpy as np

c = np.linspace(0.1, 1, 10)

for a in c:
    for i in range(10):
        asyn = h.Exp2Syn(a, sec=dendrite)
        asyn.tau1 = 10
        asyn.tau2 = 10
        s.interval = 80
        s.number = 5
        s.noise = 0
        s.start = 100
        stim = h.NetCon(s, asyn)
        h.finitialize(-70 * mV)
        stim.weight[0] = 0.1
        v_vec = h.Vector()  # Membrane potential vector
        t_vec = h.Vector()  # Time stamp vector
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)

        h.tstop = 500
        h.run()

        ax = plt.subplot(5, 2, i + 1)
        ax.plot(t_vec, v_vec)

ax.set_xlabel('time (ms)')
ax.set_ylabel('mV')
plt.savefig('test.jpg')
plt.show()

# %%


# %%

v_vec = h.Vector()  # Membrane potential vector
t_vec = h.Vector()  # Time stamp vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)

# %%


# %%

h.tstop = 500
h.run()

# %%

for i in range(0, len(c)):
    ax = plt.subplot(5, 2, i + 1)
    ax.plot(t_vec, v_vec)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('mV')

plt.show()

# %%


plt.figure(figsize=(8, 4))  # Default figsize is (8,6)
plt.plot(t_vec, v_vec)
plt.xlabel('time (ms)')
plt.xlim(0, 500)
plt.ylabel('mV')
plt.show()

# %%

x1 = image[1000:, 0]
y1 = image[1000:, 1]

# %%

v = h.Vector([0, 0])


def efun(v):
    return (v[0] + v[1]) ** 2 + (v[0] - v[1]) ** 2


h.attr_praxis(1e-5, 0.5, 0)
e = h.fit_praxis(efun, v)
print("e=%g x=%g y=%g\n" % (e, v[0], v[1]))

# %%


# %%



