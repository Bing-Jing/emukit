from ppo import PPO
from ppo2 import PPO2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from emukit.core.acquisition import Acquisition
from emukit.experimental_design.model_free.random_design import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement,ProbabilityOfImprovement
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

from funcEnv import funcEnv

model_path = "model/modelMin.ckpt"
Exploration_parameter = 0.01



def plotFIG(bomodel,poltNum,fun,titleName = ""):
            
            x_plot = np.linspace(0, 1, 1000)[:, None]
            y_plot = fun(x_plot)
            mu_plot, var_plot = bomodel.model.predict(x_plot)
            plt.subplot(poltNum)
            l1, = plt.plot(bomodel.loop_state.X[num_data_points:], bomodel.loop_state.Y[num_data_points:], "ro", markersize=10, label="Observations")
            l0, = plt.plot(bomodel.loop_state.X[:num_data_points], bomodel.loop_state.Y[:num_data_points], "bo", markersize=10, label="init_sample")
            l2, = plt.plot(x_plot, y_plot, "k", label="Objective Function")
            l3, = plt.plot(x_plot, mu_plot, "C0", label="Model")
            plt.fill_between(x_plot[:, 0],
                            mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                            mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)

            plt.fill_between(x_plot[:, 0],
                            mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                            mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)

            plt.fill_between(x_plot[:, 0],
                            mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                            mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$f(x)$")
            plt.grid(True)
            plt.xlim(0, 1)
            plt.legend(handles=[l0,l1,l2,l3,],loc="best")
            plt.title(titleName)


if __name__ == "__main__":
    
    env = funcEnv()
    parameter_space = ParameterSpace([ContinuousParameter('x1', 0, 1)])
    num_data_points = 5
    fun = env.reset(upper_bound=1,lower_bound=0)

    ppo = PPO(model = None)


###### testing
    
    saver = tf.train.Saver()
    saver.restore(ppo.sess, model_path)
    for ep in range(50):
        
        fun = env.reset(upper_bound=1,lower_bound=0)
        ppo.ppoMax = 0
        ppoMin = float('inf')
        ppo.ep_r = 0
        ppo.funCurMax = env.maxVal
        ppo.curFun = env.getCurFun()

        design = RandomDesign(parameter_space) # Collect random points
        X = design.get_samples(num_data_points)
        Y = fun(X)
        model_gpy = GPRegression(X,Y) # Train and wrap the model in Emukit
        model_emukit = GPyModelWrapper(model_gpy)
        ppo.model = model_emukit
        bo = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = ppo,
                                         batch_size = 1)
        mu = np.array([np.mean(bo.loop_state.X)])[np.newaxis]
        var = np.array([np.var(bo.loop_state.X)])[np.newaxis]
        s = np.concatenate((mu,var),axis=1)
        boPPOep_r = []

        
        model_gpyEI = GPRegression(X,Y) # Train and wrap the model in Emukit
        model_emukitEI = GPyModelWrapper(model_gpyEI)
        boEI = BayesianOptimizationLoop(model = model_emukitEI,
                                         space = parameter_space,
                                         acquisition = ExpectedImprovement(model = model_emukit,jitter = Exploration_parameter),
                                         batch_size = 1)
        boEIep_r = []
        EIcurMax = 0
        EIcurMin = float('inf')


        model_gpyPI = GPRegression(X,Y) # Train and wrap the model in Emukit
        model_emukitPI = GPyModelWrapper(model_gpyPI)
        boPI = BayesianOptimizationLoop(model = model_emukitPI,
                                         space = parameter_space,
                                         acquisition = ProbabilityOfImprovement(model = model_emukit,jitter = Exploration_parameter),
                                         batch_size = 1)
        boPIep_r = []
        PIcurMax = 0
        PIcurMin = float('inf')


        for t in range(50):    # in one episode
            bo.run_loop(fun, 1)
            a = bo.loop_state.X[-1]
            ppo.ppoMax = max(ppo.ppoMax,fun(a))
            ppoMin = min(ppoMin,ppo.curFun(a))
            # r = -(env.maxVal-ppo.ppoMax)
            r = (env.minVal-ppoMin)
            r = float(r)
            boPPOep_r.append(r)

            ### EI
            boEI.run_loop(fun, 1)
            aEI = boEI.loop_state.X[-1]
            EIcurMax = max(EIcurMax,fun(aEI))
            EIcurMin = min(EIcurMin,fun(aEI))

            # rEI = -(env.maxVal-EIcurMax)
            rEI = (env.minVal-EIcurMin)
            boEIep_r.append(rEI)

            ### PI
            boPI.run_loop(fun,1)
            aPI = boPI.loop_state.X[-1]
            PIcurMax = max(PIcurMax,fun(aPI))
            PIcurMin = min(PIcurMin,fun(aPI))

            # rPI = -(env.maxVal-PIcurMax)
            rPI = (env.minVal-PIcurMin)
            boPIep_r.append(rPI)


        plt.figure(figsize=(12, 8))

        plotFIG(bo,234,fun,"ppo")
        plotFIG(boEI,235,fun,"EI")
        plotFIG(boPI,236,fun,"PI")

        plt.subplot(211)
        plt.ylabel(r"$reward$")
        plt.xlabel(r"$episode$")
        plotep = np.linspace(0, len(boPPOep_r), len(boPPOep_r))
        plt.plot(plotep, boPPOep_r,"r")
        plt.plot(plotep, boEIep_r,"b")
        plt.plot(plotep, boPIep_r,"g")

        p1, = plt.plot(plotep, boPPOep_r, "ro", markersize=5,label="ppo")
        p2, = plt.plot(plotep, boEIep_r, "bo", markersize=7,label="EI")
        p3, = plt.plot(plotep, boPIep_r, "go", markersize=5,label="PI")
        plt.legend(handles=[p1,p2,p3,],loc="best")


        fig1 = plt.gcf()
        # plt.show()
        plt.draw()
        fig1.savefig('compareMin2/ep_{}.png'.format(ep), dpi=100)