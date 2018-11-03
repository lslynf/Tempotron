import numpy as np
import random
import matplotlib.pyplot as plt

class Tempotron:
    def __init__(self,numbers,synapseWeight):
        #定义常数
        self.Vth = 1
        self.Vrest = 0
        self.V0 = 2.12
        self.tau = 15
        self.taus = self.tau/4
        self.T = 500
        self.tstep = 0.1
        self.synapseWeight = synapseWeight
        self.n = numbers

    def kFunction(self,t,ti):
        v = 0
        #当时间比发出spike的时间小时,对突触后膜的膜电位没有影响
        if t<ti:
           v = 0
        else:
           v = self.V0*(np.exp(-(t-ti)/self.tau)-np.exp(-(t-ti)/self.taus))
        return v

    def postsynapicPotentials(self,spikeTimes,t):
        #某一时刻的电位
        vMembrane=np.zeros(self.n)

        for afferent in range(self.n):
            for Ti in spikeTimes[afferent]:
                vMembrane[afferent] += self.kFunction(t,Ti)

        sumOfV = 0
        for i in range(self.n):
            sumOfV += self.synapseWeight[i]*vMembrane[i]

        sumOfV += self.Vrest

        return sumOfV

    def changeOfMenbrane(self,spikeTimes):
        #计算每个时刻突触后膜的电位变化
        t = np.arange(0,self.T+self.tstep,self.tstep)
        Vm = np.zeros(len(t))

        for i,time in enumerate(t):
            Vm[i] = self.postsynapicPotentials(spikeTimes,time)

        return Vm

    def updateWeight(self,learningRate,spikeTimes,tmax):
        #计算每个传入神经的权重变化
        dw = np.zeros(self.n)

        for afferent in range(self.n):
            for Ti in spikeTimes[afferent]:
                dw[afferent] += self.kFunction(tmax,Ti)

        dw = dw*learningRate

        return dw

    def train(self,spikeTimes,output,learningRate):

        '''
           对于每次训练,计算整个时间范围的膜电位,计算Vmax和阈值进行比较
           若与output所期望的输出不一致则更改权重
        '''
        trainTimes = 200
        t = np.arange(0,500+0.1,0.1)

        for i in range(trainTimes):
            #计算每一次的Vmax和tmax
            vMembrane = self.changeOfMenbrane(spikeTimes)

            Vmax = 0
            tmax = 0

            for key,value in enumerate(vMembrane):
                if value > Vmax:
                    Vmax = value
                    tmax = t[key]

            #若为正的模式
            if output == 1:
                if Vmax < self.Vth:
                    dw = self.updateWeight(learningRate,spikeTimes,tmax)
                    for i in range(self.n):
                        self.synapseWeight[i] += dw[i]

            #若为负的模式
            else:
                if Vmax > self.Vth:
                    dw = self.updateWeight(learningRate,spikeTimes,tmax)
                    for i in range(self.n):
                        self.synapseWeight[i] -= dw[i]

            random.shuffle(spikeTimes)


if __name__=='__main__':


    spikeTimeOfPros = [[240,360],[280,355],[150],[320],[],[80,300],[],[45,185],[400],[]]
    spikeTimeOfCons = [[120],[315],[390],[270,290,370],[255],[145],[],[420,440],[],[80,220,412]]

    synapseWeight = np.random.rand(1,10)[0]

    a=Tempotron(10,synapseWeight)


    v0=a.changeOfMenbrane(spikeTimeOfCons)
    a.train(spikeTimeOfCons,0,0.01)
    v1=a.changeOfMenbrane(spikeTimeOfCons)

    v2=a.changeOfMenbrane(spikeTimeOfPros)
    a.train(spikeTimeOfPros,1,0.01)
    v3=a.changeOfMenbrane(spikeTimeOfPros)

    t = np.arange(0, 500 + 0.1, 0.1)
    y=np.ones(5001)


    plt.figure("result")
    plt.subplot(221)
    plt.ylim(0,1.7)
    plt.plot(t,y,'--')
    plt.plot(t,v0,'r--')
    plt.annotate('negetiveBefore', xy=(200, 1.5), xytext=(100, 1.5))

    plt.subplot(222)
    plt.ylim(0,1.7)
    plt.plot(t, y, '--')
    plt.plot(t,v1,'r--')
    plt.annotate('negetiveAfter', xy=(200, 1.5), xytext=(100, 1.5))

    plt.subplot(223)
    plt.ylim(0,1.7)
    plt.plot(t,y,'--')
    plt.plot(t,v2,'b--')
    plt.annotate('positiveBefore', xy=(200, 1.5), xytext=(100, 1.5))

    plt.subplot(224)
    plt.ylim(0,1.7)
    plt.plot(t,y,'--')
    plt.plot(t,v3,'b--')
    plt.annotate('positiveAfter', xy=(200, 1.5), xytext=(100, 1.5))

    plt.show()
