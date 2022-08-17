from brian2 import *
import time
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import numpy as np
import plot_utils as pu
import numpy.fft

t        = time.time()


duration = 2000

# Parameters neurons
C        = 1       * uF
gLe      = 0.05    * msiemens
gLi      = 0.1     * msiemens
taume    = C / gLe
taumi    = C / gLi
EL       = -70.6   * mV
VT       = -50.4   * mV
DeltaT   = 2       * mV
Vcut     = VT + 5  * DeltaT

## === ## Ctx
probK    = 0.1
NN       = 5000
NE       = round(NN * 0.8)
NI       = NN - NE


K        = NN * probK

taue     = 5       * ms
taui     = 2.5     * ms
Ee       = 0       * mV
Ei       = -80     * mV


###  CTX  ###
## Fast spiking    === Inhibitory
Vr       = -70.6 * mV
## Regular spiking ===  Excitatory
tauw     = 60 * ms
a        = 24 * nS
b        = 0.01 * nA

VrP      = -70.6 * mV
# Regular spiking  ===  Excitatory
tauwP    = 60 * ms
aP       = 24 * nS
bP       = 0.01 * nA


## Strength connections
scale    = 0.009
# Balanced network CTXs
Je0      = scale * 0.95  * msiemens ## 0.95 #1.1
Ji0      = scale * 1.26   * msiemens

Jei      = scale * 60     * msiemens
Jii      = scale * 90     * msiemens
Jee      = scale * 0.15   * msiemens
Jie      = scale * 0.45   * msiemens



############ Factors ###############

W_is     = 5   # 70 SUP -> INF
W_si     = 10  # 40  INF -> SUP

W_ff     = 5
W_fb     = 1.5
W_pulIn  = 1
W_fbe    = 0  ###  Ctx2 -> Ctx1 deep to deep (sature le signal)


W_pC1    = 5   ### Ctx1 -> Pul
W_C1p    = 2.5   ### Pul  -> Ctx1

W_pC2    = 5 ### Pul  -> Ctx2
W_C2p    = 2.5  ### Ctx2 -> Pul

#### Balanced network pul
Jei_P      = Jei
Jii_P      = Jii
Jee_P      = Jee * W_pulIn
Jie_P      = Jie


####### Ctxs between slices

Jepes    = Je0 * W_is #### Excitatory superior --> Excitatory deep
Jisep    = Je0 * W_si #### Excitatory superior --> Inhibitory deep
Jipes    = Ji0 * W_is
Jesep    = Je0 * W_si


# Feedforward Long range connexion Ctx 1 superficial -> Ctx2 superficial
JeeC1C2  = W_ff * Je0
JieC1C2  = W_ff * Ji0


# Feedback Long range connexion Ctx 2 deep - Ctx1 superficial
JesepC2C1  = W_fb * Je0
JisepC2C1  = W_fb * Ji0

JepepC2C1  = W_fbe * Je0
JipepC2C1  = W_fbe * Ji0

## FF Pul --> Ctx1 superficiel
JeseC1P    = W_C1p * Je0
JiseC1P    = W_C1p * Ji0

## Fb Ctx 1 superficiel --> pul
JesePC1    = W_pC1 * Je0
JisePC1    = W_pC1 * Ji0


# FF Long range connexion Ctx 2 superficielle IV - Pul
JeesC2P = W_pC2 * Je0
JiesC2P = W_pC2 * Ji0

# FB Long range connexion Pul - Ctx2
JesePC2 = W_C2p * Je0
JisePC2 = W_C2p * Ji0



print('Jepes='+str(Jepes))
print('Jisep='+str(Jisep))

#Prob Ctxs
pConE0CC = 0.05   ##0.05
pConI0CC = 0.05   ## 0.05
pConEICC = 0.2    ## 0.2
pConIICC = 0.2    ## 0.1
pConEECC = 0.05   ## 0.05
pConIECC = 0.05   ##  0.05

## Feedforward long range connexion Ctx 1 - Ctx 2
pConEEC1C2 = pConE0CC
pConIEC1C2 = pConI0CC

## Feedback long range connexion Ctx 1 <- Ctx 2
pConEEC2C1 = pConE0CC
pConIEC2C1 = pConI0CC

# Feedforward Long range connexion Ctx 1 - Pul
pConEEC1P = pConE0CC
pConIEC1P = pConI0CC

# Feedback Long range connexion Pul - Ctx 1
pConEEPC1 = pConE0CC
pConIEPC1 = pConI0CC

# Feedforward Long range connexion Ctx 2 - Pul
pConEEC2P = pConE0CC
pConIEC2P = pConI0CC

# Feedback Long range connexion Pul - Ctx2
pConEEPC2 = pConE0CC
pConIEPC2 = pConI0CC


#### NETWORK ########
## Input
inptRate = 10 * Hz
n_inpt   = 8000

## Equations
eqsI     = """
dvm/dt   = (gLi*(EL - vm) + gLi*DeltaT*exp((vm - VT)/DeltaT) + ge*(Ee-vm) + gi*(Ei-vm) )/C : volt
dge/dt   = -ge*(1./taue) : siemens
dgi/dt   = -gi*(1./taui) : siemens
"""
eqsE     = """
dvm/dt   = (gLe*(EL - vm) + gLe*DeltaT*exp((vm - VT)/DeltaT)+ ge*(Ee-vm) + gi*(Ei-vm)  - w)/C : volt
dw/dt    = (a*(vm - EL) - w)/tauw  : amp
dge/dt   = -ge*(1./taue)           : siemens
dgi/dt   = -gi*(1./taui)           : siemens
"""


R0        = 0  * Hz
R1        = 25 * Hz

d = 50

inputLGN = PoissonGroup(n_inpt, inptRate)
inputContraste  = R0 + R1 * np.log10(1+d)
inputLGN.rates  = inputContraste
#
# inputLGN.rates = 50 * Hz


#Ctx V1
#Couche superficielle
CtxI1_s    = NeuronGroup(NI, model = eqsI, threshold='vm>Vcut',reset="vm=Vr", method='euler')
CtxE1_s    = NeuronGroup(NE, model = eqsE, threshold='vm>Vcut',reset="vm=Vr;w+=b", method='euler')

#Couche profonde
CtxI1_p    = NeuronGroup(NI, model = eqsI, threshold='vm>Vcut',reset="vm=Vr", method='euler')
CtxE1_p    = NeuronGroup(NE, model = eqsE, threshold='vm>Vcut',reset="vm=Vr;w+=b", method='euler')

#CtxV2
#Couche superficielle
CtxI2_s    = NeuronGroup(NI, model = eqsI, threshold='vm>Vcut',reset="vm=Vr", method='euler')
CtxE2_s    = NeuronGroup(NE, model = eqsE, threshold='vm>Vcut',reset="vm=Vr;w+=b", method='euler')

#Couche profonde
CtxI2_p    = NeuronGroup(NI, model = eqsI, threshold='vm>Vcut',reset="vm=Vr", method='euler')
CtxE2_p    = NeuronGroup(NE, model = eqsE, threshold='vm>Vcut',reset="vm=Vr;w+=b", method='euler')


#Pul
PulI     = NeuronGroup(NI,model = eqsI, threshold='vm>Vcut', reset= "vm = Vr", method='euler')
PulE     = NeuronGroup(NE, model = eqsE, threshold='vm>Vcut', reset="vm=Vr;w+=b", method = 'euler')


CtxI1_s.vm = 'EL * rand()'
CtxE1_s.vm = 'EL * rand()'
CtxE1_s.w  = 'rand() * nA'

CtxI1_p.vm = 'EL * rand()'
CtxE1_p.vm = 'EL * rand()'
CtxE1_p.w  = 'rand() * nA'

CtxI2_s.vm = 'EL * rand()'
CtxE2_s.vm = 'EL * rand()'
CtxE2_s.w  = 'rand() * nA'

CtxI2_p.vm = 'EL * rand()'
CtxE2_p.vm = 'EL * rand()'
CtxE2_p.w  = 'rand() * nA'


PulI.vm = 'EL * rand()'
PulE.vm = 'EL * rand()'
PulE.w  = 'rand() * nA'


##### Create Synapses #####
#Ctx 1
# Long range connexion LGN -> Ctx1_s = Feedforward
C1_e0    = Synapses(inputLGN, CtxE1_s, model = 'wwe0_1 : siemens', on_pre = 'ge+=wwe0_1')
C1_i0    = Synapses(inputLGN, CtxI1_s, model = 'wwi0_1 : siemens', on_pre = 'ge+=wwi0_1')


#Balanced network
#####  Ctx1

####### Superficial
C1C1_ei_s  = Synapses(CtxI1_s, CtxE1_s, model = 'wwei_1 : siemens', on_pre = 'gi+=wwei_1')
C1C1_ii_s  = Synapses(CtxI1_s, CtxI1_s, model = 'wwii_1 : siemens', on_pre = 'gi+=wwii_1')
C1C1_ie_s  = Synapses(CtxE1_s, CtxI1_s, model = 'wwie_1 : siemens', on_pre = 'ge+=wwie_1')
C1C1_ee_s  = Synapses(CtxE1_s, CtxE1_s, model = 'wwee_1 : siemens', on_pre = 'ge+=wwee_1')

######## Deep
C1C1_ei_p  = Synapses(CtxI1_p, CtxE1_p, model = 'wwei_1 : siemens', on_pre = 'gi+=wwei_1')
C1C1_ii_p  = Synapses(CtxI1_p, CtxI1_p, model = 'wwii_1 : siemens', on_pre = 'gi+=wwii_1')
C1C1_ie_p  = Synapses(CtxE1_p, CtxI1_p, model = 'wwie_1 : siemens', on_pre = 'ge+=wwie_1')
C1C1_ee_p  = Synapses(CtxE1_p, CtxE1_p, model = 'wwee_1 : siemens', on_pre = 'ge+=wwee_1')

########### Between the 2 slices

C1C1_esep = Synapses(CtxE1_s,CtxE1_p, model = 'wwee_1 : siemens', on_pre = 'ge+=wwee_1')
C1C1_ipes = Synapses(CtxE1_p,CtxI1_s, model = 'wwie_1 : siemens', on_pre = 'gi+=wwie_1')


#####Ctx2

####### Superficial
C2C2_ei_s  = Synapses(CtxI2_s, CtxE2_s, model = 'wwei_2 : siemens', on_pre = 'gi+=wwei_2')
C2C2_ii_s  = Synapses(CtxI2_s, CtxI2_s, model = 'wwii_2 : siemens', on_pre = 'gi+=wwii_2')
C2C2_ie_s  = Synapses(CtxE2_s, CtxI2_s, model = 'wwie_2 : siemens', on_pre = 'ge+=wwie_2')
C2C2_ee_s  = Synapses(CtxE2_s, CtxE2_s, model = 'wwee_2 : siemens', on_pre = 'ge+=wwee_2')

######## Deep
C2C2_ei_p  = Synapses(CtxI2_p, CtxE2_p, model = 'wwei_2 : siemens', on_pre = 'gi+=wwei_2')
C2C2_ii_p  = Synapses(CtxI2_p, CtxI2_p, model = 'wwii_2 : siemens', on_pre = 'gi+=wwii_2')
C2C2_ie_p  = Synapses(CtxE2_p, CtxI2_p, model = 'wwie_2 : siemens', on_pre = 'ge+=wwie_2')
C2C2_ee_p  = Synapses(CtxE2_p, CtxE2_p, model = 'wwee_2 : siemens', on_pre = 'ge+=wwee_2')

########### Between the 2 slices

C2C2_esep  = Synapses(CtxE2_s,CtxE2_p, model = 'wwee_2 : siemens', on_pre = 'ge+=wwee_2')
C2C2_ipes  = Synapses(CtxE2_p,CtxI2_s, model = 'wwie_2 : siemens', on_pre = 'gi+=wwie_2')

#########Pul
PP_ei      = Synapses(PulI, PulE, model = 'wwei_P : siemens', on_pre = 'gi+=wwei_P')
PP_ii      = Synapses(PulI, PulI, model = 'wwii_P : siemens', on_pre = 'gi+=wwii_P')
PP_ie      = Synapses(PulE, PulI, model = 'wwie_P : siemens', on_pre = 'ge+=wwie_P')
PP_ee      = Synapses(PulE, PulE, model = 'wwee_P : siemens', on_pre = 'ge+=wwee_P')



##### Ctx1 --> Ctx2
C1C2_ee    = Synapses(CtxE1_s, CtxE2_s, model = 'wwee_1 : siemens', on_pre= 'ge+=wwee_1')
C1C2_ie    = Synapses(CtxE1_s, CtxI2_s, model = 'wwie_1 : siemens', on_pre = 'ge+=wwie_1')

#### Ctx2 ---> Ctx1
C2C1_esep  = Synapses(CtxE2_p, CtxE1_s, model = 'wwee_2 : siemens', on_pre= 'ge+=wwee_2')
C2C1_isep  = Synapses(CtxE2_p, CtxI1_s, model = 'wwie_2 : siemens', on_pre = 'ge+=wwie_2')


C2C1_epep  = Synapses(CtxE2_p, CtxE1_p, model = 'wwee_2 : siemens', on_pre= 'ge+=wwee_2')
C2C1_ipep  = Synapses(CtxE2_p, CtxI1_p, model = 'wwie_2 : siemens', on_pre= 'ge+=wwie_2')

# Long range connexion Ctx1 - Pul Feedforward
C1P_ese    = Synapses(CtxE1_s,PulE,model = 'wwee_P : siemens', on_pre = 'ge+=wwee_P')
C1P_ise    = Synapses(CtxE1_s,PulI,model = 'wwie_P : siemens', on_pre = 'ge+=wwie_P')


# Long range connexion Pul - Ctx 1 Feedback
PC1_ese   = Synapses(PulE,CtxE1_s,model = 'wwee_P : siemens', on_pre= 'ge+=wwee_P')
PC1_ise   = Synapses(PulE,CtxI1_s,model = 'wwie_P : siemens', on_pre = 'ge+=wwie_P')

# Long range connexion Ctx2 - Pul Feedforward
C2P_ees   = Synapses(CtxE2_s,PulE,model = 'wwee_P : siemens', on_pre = 'ge+=wwee_P')
C2P_ies   = Synapses(CtxE2_s,PulI,model = 'wwie_P : siemens', on_pre = 'ge+=wwie_P')

# Long range connexion Pul - Ctx2 Feedback
PC2_ese    = Synapses(PulE,CtxE2_s,model = 'wwee_P : siemens', on_pre= 'ge+=wwee_P')
PC2_ise    = Synapses(PulE,CtxI2_s,model = 'wwie_P : siemens', on_pre = 'ge+=wwie_P')


#make the connexions

#Ctx1
#Long range connexion LGN - Ctx 1
C1_e0.connect(p=pConE0CC)
C1_i0.connect(p=pConE0CC)

#Balanced network superficiel
C1C1_ei_s.connect(p=pConEICC)
C1C1_ii_s.connect(p=pConIICC)
C1C1_ie_s.connect(p=pConIECC)
C1C1_ee_s.connect(p=pConEECC)

#Balaneced network profond
C1C1_ei_p.connect(p=pConEICC)
C1C1_ii_p.connect(p=pConIICC)
C1C1_ie_p.connect(p=pConIECC)
C1C1_ee_p.connect(p=pConEECC)

#entre les couches
C1C1_esep.connect(p=pConE0CC) 
C1C1_ipes.connect(p=pConE0CC) 

#CtxV2

#Balanced network superficiel
C2C2_ei_s.connect(p=pConEICC)
C2C2_ii_s.connect(p=pConIICC)
C2C2_ie_s.connect(p=pConIECC)
C2C2_ee_s.connect(p=pConEECC)

#Balaneced network profond
C2C2_ei_p.connect(p=pConEICC)
C2C2_ii_p.connect(p=pConIICC)
C2C2_ie_p.connect(p=pConIECC)
C2C2_ee_p.connect(p=pConEECC)

#entre les couches
C2C2_esep.connect(p=pConE0CC)
C2C2_ipes.connect(p=pConE0CC)

#Pul BN
PP_ie.connect(p=pConIECC)
PP_ii.connect(p=pConIICC)
PP_ei.connect(p=pConEICC)
PP_ee.connect(p=pConEECC)

#Ctx1 --> Ctx2
C1C2_ie.connect(p=pConE0CC)
C1C2_ee.connect(p=pConE0CC)

#Ctx2 --> Ctx1
C2C1_epep.connect(p=pConE0CC)
C2C1_esep.connect(p=pConE0CC)
C2C1_ipep.connect(p=pConE0CC)
C2C1_isep.connect(p=pConE0CC)


#Ctx1 --> Pul
C1P_ese.connect(p=pConEEC1P)
C1P_ise.connect(p=pConIEC1P)

#Pul --> Ctx1
PC1_ese.connect(p=pConEEPC1)
PC1_ise.connect(p=pConIEPC1)

#Ctx2 --> Pul
C2P_ees.connect(p=pConEEC2P)
C2P_ies.connect(p=pConIEC2P)

#Pul --> Ctx2
PC2_ese.connect(p=pConEEPC2)
PC2_ise.connect(p=pConIEPC2)


# Normalization by number of synapses K

# Feedforward Long range connexion LGN -> Ctx1
C1_e0.wwe0_1    = Je0 * (1./sqrt(K))
C1_i0.wwi0_1    = Ji0 * (1./sqrt(K))

#### Balanced network
##Ctx 1 superficiel
C1C1_ei_s.wwei_1  = Jei * (1./sqrt(K))
C1C1_ii_s.wwii_1  = Jii * (1./sqrt(K))
C1C1_ee_s.wwee_1  = Jee * (1./sqrt(K))
C1C1_ie_s.wwie_1  = Jie * (1./sqrt(K))

##Ctx 1 profonde
C1C1_ei_p.wwei_1  = Jei * (1./sqrt(K))
C1C1_ii_p.wwii_1  = Jii * (1./sqrt(K))
C1C1_ee_p.wwee_1  = Jee * (1./sqrt(K))
C1C1_ie_p.wwie_1  = Jie * (1./sqrt(K))

C1C1_esep.wwee_1  = Jepes * (1./sqrt(K))
C1C1_ipes.wwie_1  = Jisep * (1./sqrt(K))


##Ctx 2 superficiel
C2C2_ei_s.wwei_2  = Jei * (1./sqrt(K))
C2C2_ii_s.wwii_2  = Jii * (1./sqrt(K))
C2C2_ee_s.wwee_2  = Jee * (1./sqrt(K))
C2C2_ie_s.wwie_2  = Jie * (1./sqrt(K))

##Ctx 2 profonde
C2C2_ei_p.wwei_2  = Jei * (1./sqrt(K))
C2C2_ii_p.wwii_2  = Jii * (1./sqrt(K))
C2C2_ee_p.wwee_2  = Jee * (1./sqrt(K))
C2C2_ie_p.wwie_2  = Jie * (1./sqrt(K))

C2C2_esep.wwee_2  = Jepes * (1./sqrt(K))
C2C2_ipes.wwie_2  = Jisep * (1./sqrt(K))

#Pul

PP_ee.wwee_P      = Jee_P * (1./sqrt(K))
PP_ei.wwei_P      = Jei_P * (1./sqrt(K))
PP_ie.wwie_P      = Jie_P * (1./sqrt(K))
PP_ii.wwii_P      = Jii_P * (1./sqrt(K))


# Ctx1 --> Ctx2
C1C2_ee.wwee_1    = JeeC1C2 * (1./sqrt(K))
C1C2_ie.wwie_1    = JieC1C2 * (1./sqrt(K))

#Ctx2 --> Ctx1
C2C1_isep.wwie_2  = JisepC2C1 * (1./sqrt(K))
C2C1_ipep.wwie_2  = JipepC2C1 * (1./sqrt(K))
C2C1_esep.wwee_2  = JesepC2C1 * (1./sqrt(K))
C2C1_epep.wwee_2  = JepepC2C1 * (1./sqrt(K))

#Ctx1 -> Pul
C1P_ese.wwee_P    = JeseC1P  * (1./sqrt(K))
C1P_ise.wwie_P    = JiseC1P  * (1./sqrt(K))

#Pul -> Ctx1

PC1_ese.wwee_P    = JesePC1  * (1./sqrt(K))
PC1_ise.wwie_P    = JisePC1  * (1./sqrt(K))

#Ctx2 -> Pul
C2P_ees.wwee_P    = JeesC2P  * (1./sqrt(K))
C2P_ies.wwie_P    = JiesC2P  * (1./sqrt(K))

#Pul -> Ctx2
PC2_ese.wwee_P    = JesePC2  * (1./sqrt(K))
PC2_ise.wwie_P    = JisePC2  * (1./sqrt(K))

# Spike monitor
SpE1_s       = SpikeMonitor(CtxE1_s)
SpE1_p       = SpikeMonitor(CtxE1_p)
SpE2_s       = SpikeMonitor(CtxE2_s)
SpE2_p       = SpikeMonitor(CtxE2_p)
SpPE         = SpikeMonitor(PulE)


# SpI1_s       = SpikeMonitor(CtxI1_s)
# SpI1_p       = SpikeMonitor(CtxI1_p)
# SpI2_s       = SpikeMonitor(CtxI2_s)
# SpI2_p       = SpikeMonitor(CtxI2_p)
# SpPI         = SpikeMonitor(PulI)


# Pop rate monitor

#Ctx 1
PopE1_s = PopulationRateMonitor(CtxE1_s)
PopE1_p = PopulationRateMonitor(CtxE1_p)
PopE2_s = PopulationRateMonitor(CtxE2_s)
PopE2_p = PopulationRateMonitor(CtxE2_p)
PopPE   = PopulationRateMonitor(PulE)

# PopI1_s = PopulationRateMonitor(CtxI1_s)
# PopI1_p = PopulationRateMonitor(CtxI1_p)
# PopPI   = PopulationRateMonitor(PulI)

run(duration * ms)

E1S =PopE1_s.smooth_rate(window='flat', width=2.1*ms)/Hz
E1P =PopE1_p.smooth_rate(window='flat', width=2.1*ms)/Hz
E2S =PopE2_s.smooth_rate(window='flat', width=2.1*ms)/Hz
E2P =PopE2_p.smooth_rate(window='flat', width=2.1*ms)/Hz



############### plot #########################
 ##### Raster plot population

plot(SpE1_s.t/ms, SpE1_s.i, ',r')
plot(SpE1_p.t/ms, SpE1_p.i+ NE, ',b')
plot(SpE2_s.t/ms,SpE2_s.i+NE+NE,',m')
plot(SpE2_p.t/ms,SpE2_p.i+NE+NE + NE,',c')
plt.ylabel('Neuron index')
plt.xlabel('Time (ms)')
plt.xlim(duration-100,duration)

red_patch = mpatches.Patch(color='red', label='C1E_s')
blue_patch = mpatches.Patch(color='blue', label='C1E_p')
maganta_patch = mpatches.Patch(color='magenta', label='C2E_s')
cyan_patch = mpatches.Patch(color='cyan', label='C2E_p')
#legend(handles=[red_patch,blue_patch,maganta_patch,cyan_patch],loc='upper right')
legend(handles=[red_patch,blue_patch],loc='upper right')



#show()

####### PSTH ###### EXCITATION

#fig1, ax = plt.subplots(4, sharex=False,
#                        gridspec_kw={'height_ratios': [1,1,1,1],
#                                     'left': 0.18, 'bottom': 0.18, 'top': 0.95,
#                                     'hspace': 0.1},
#                        figsize=(3.07, 3.07))

# fig1, ax = plt.subplots(2, sharex=False,
#                         gridspec_kw={'height_ratios': [1,1],
#                                      'left': 0.18, 'bottom': 0.18, 'top': 0.95,
#                                      'hspace': 0.1},
#                         figsize=(3.07, 3.07))



### PSTH Ctx1 PopE-S


# ax[0].plot(PopE1_s.t/ms, PopE1_s.smooth_rate(window='flat', width=2.1*ms)/Hz)
# ax[0].set(xlim=(duration-1000,duration),
#           ylabel = "Cortex E1S")


# ### PSTH Ctx1 PopE-P

# ax[1].plot(PopE1_p.t/ms, PopE1_p.smooth_rate(window='flat', width=2.1*ms)/Hz)
# ax[1].set(xlim=(duration-1000,duration),
#           ylabel = "Cortex E1p")


# plt.xlabel('time (ms)')


############### FOURIER #######################
scN = 1000 #### Fourier in ms, normalize factor

### E1S

Fs  = scN/defaultclock.dt
Ts  = scN/Fs
t   = np.arange(0,1,Ts)
n   = len(E1S)
k   = np.arange(n)
T   = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))]/scN # one side frequency range
hfP = int((len(E1S))/2)
y   = E1S[hfP:]
Y   = np.fft.fft(y)/n # fft computing and normalization
Y   = Y[range(int(n/2))]
fig, ax = plt.subplots(2, 1)
suptitle('E1S')
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('E1S Amplitude')
ax[1].plot(frq[1:],abs(Y[1:]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('E1S |Y(freq)|')
ax[1].set(xlim=(0,100))


###E1P

Fs  = scN/defaultclock.dt
Ts  = scN/Fs
t   = np.arange(0,1,Ts)
n   = len(E1P)
k   = np.arange(n)
T   = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))]/scN # one side frequency range
hfP = int((len(E1P))/2)
y   = E1P[hfP:]
Y   = np.fft.fft(y)/n # fft computing and normalization
Y   = Y[range(int(n/2))]
fig, ax = plt.subplots(2, 1)
suptitle('E1P')

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('E1P Amplitude')
ax[1].plot(frq[1:],abs(Y[1:]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('E1P |Y(freq)|')
ax[1].set(xlim=(0,100))

#####E2S

Fs  = scN/defaultclock.dt
Ts  = scN/Fs
t   = np.arange(0,1,Ts)
n   = len(E2S)
k   = np.arange(n)
T   = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))]/scN # one side frequency range
hfP = int((len(E2S))/2)
y   = E2S[hfP:]
Y   = np.fft.fft(y)/n # fft computing and normalization
Y   = Y[range(int(n/2))]
fig, ax = plt.subplots(2, 1)
suptitle('E2S')

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('E2S Amplitude')
ax[1].plot(frq[1:],abs(Y[1:]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('E2S |Y(freq)|')
ax[1].set(xlim=(0,100))


###E2P

Fs  = scN/defaultclock.dt
Ts  = scN/Fs
t   = np.arange(0,1,Ts)
n   = len(E2P)
k   = np.arange(n)
T   = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))]/scN # one side frequency range
hfP = int((len(E2P))/2)
y   = E2P[hfP:]
Y   = np.fft.fft(y)/n # fft computing and normalization
Y   = Y[range(int(n/2))]
fig, ax = plt.subplots(2, 1)
suptitle('E2P')

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('E2P Amplitude')
ax[1].plot(frq[1:],abs(Y[1:]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('E2P |Y(freq)|')
ax[1].set(xlim=(0,100))


show()
plot(SpPE.t/ms, SpPE.i,',r')
show()