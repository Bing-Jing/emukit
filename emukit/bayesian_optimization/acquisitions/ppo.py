
"""
code modify from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition
from emukit.experimental_design.model_free.random_design import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import interp1d

from funcEnv import funcEnv


###########################################################################################################################


model_path = "model/model4.ckpt"


EP_MAX = 100
EP_LEN = 64
GAMMA = 0.98
LR = 0.0001
BATCH = 20
UPDATE_STEPS = 10
A_DIM = 1
CPI_epsilon = 0.15                 # Clipped surrogate objective, find this is better


class PPO(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable] = None,pre_train=None):
        self.model = model
        self.sess = tf.Session()
        self.S_DIM = 2
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.funCurMax = 0
        self.curFun = None
        self.ppoMax = 0
        self.ep_r = 0
        self.buffer_ep = []
        self.s_ = None
        # critic
        
        with tf.variable_scope('critic'):
            l1 = slim.fully_connected(self.tfs, 200,activation_fn=tf.nn.tanh)
            l1 = slim.fully_connected(l1, 200,activation_fn=tf.nn.tanh)
            l1 = slim.fully_connected(l1, 200,activation_fn=tf.nn.tanh)

            # l1 = tf.layers.dense(self.tfs, 200, activation =tf.nn.tanh)
            # l1 = tf.layers.dense(l1, 200, activation =tf.nn.tanh)
            # l1 = tf.layers.dense(l1, 200, activation =tf.nn.tanh)
            self.v = slim.fully_connected(l1,1)
            # self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-CPI_epsilon, 1.+CPI_epsilon)*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        ########
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEPS)]
        
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = slim.fully_connected(self.tfs, 200,activation_fn=tf.nn.tanh,trainable=trainable)
            l1 = slim.fully_connected(l1, 200,activation_fn=tf.nn.tanh,trainable=trainable)
            l1 = slim.fully_connected(l1, 200,activation_fn=tf.nn.tanh,trainable=trainable)

            # l1 = tf.layers.dense(self.tfs, 200, tf.nn.tanh, trainable=trainable)
            # l1 = tf.layers.dense(l1, 200, tf.nn.tanh, trainable=trainable)
            # l1 = tf.layers.dense(l1, 200, tf.nn.tanh, trainable=trainable)
            mu = 2 * slim.fully_connected(l1, A_DIM,activation_fn=tf.nn.tanh,trainable=trainable)
            # mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = 2 * slim.fully_connected(l1, A_DIM,activation_fn=tf.nn.softplus,trainable=trainable)
            # sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):     
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        a = np.clip(a, 0, 1)

        self.buffer_a.append(a)
        self.buffer_s.append(self.s_)
        # self.ppoMax = max(self.ppoMax,self.curFun(a))
        r = -(self.funCurMax-self.ppoMax)
        r = float(r)/1000
        self.ep_r += r
        self.buffer_r.append(r) 
        self.s_ = s
        return a
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    def evaluate(self, x) -> np.ndarray:
        mean, variance = self.model.predict(x)
        evarr = np.empty(x.shape[0])
        for i in range(len(x)):
            s = np.concatenate((mean[i],variance[i]),axis=0).reshape(-1,2)
            evarr[i] = self.choose_action(s)
        ## state size can not dynamically change
        # s = np.concatenate((mean,variance),axis=1).reshape(-1,2)
        # evarr = self.choose_action(s)  
        return evarr[np.newaxis, :]
    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False

if __name__ == "__main__":
    
    env = funcEnv()
    parameter_space = ParameterSpace([ContinuousParameter('x1', 0, 1)])
    num_data_points = 5
    fun = env.reset(upper_bound=1,lower_bound=0)

    ppo = PPO(model = None)
    ##### training
    ppo.buffer_ep = []
    for ep in range(EP_MAX):
        
        fun = env.reset(upper_bound=1,lower_bound=0)
        ppo.ppoMax = 0
        ppo.ep_r = 0
        boPPOep_r = []
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
        mu_, var_ = bo.model.predict(bo.loop_state.X[-1].reshape(-1,1))
        ppo.s_ = np.concatenate((mu_,var_),axis=1)                                
        for t in range(EP_LEN):    # in one episode
        
            bo.run_loop(fun, 1)
            ppo.ppoMax = max(ppo.ppoMax,env.curFun(bo.loop_state.X[-1]))
            
            
            boPPOep_r.append(ppo.ppoMax-env.maxVal)

            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                print("updating.......")
                v_s_ = ppo.get_v(ppo.s_)
                discounted_r = []
                for r in ppo.buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(ppo.buffer_s), np.vstack(ppo.buffer_a), np.array(discounted_r)[:, np.newaxis]
                ppo.buffer_s, ppo.buffer_a, ppo.buffer_r = [], [], []
                ppo.update(bs, ba, br)
        print("current ep = {} reward = {}".format(ep,ppo.ep_r))
        ppo.buffer_ep.append(ppo.ep_r)
        ### plot function
        x_plot = np.linspace(0, 1, 200)[:, None]
        y_plot = fun(x_plot)
        mu_plot, var_plot = bo.model.predict(x_plot)

        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        
        l1, = plt.plot(bo.loop_state.X[num_data_points:], bo.loop_state.Y[num_data_points:], "ro", markersize=10, label="Observations")
        l0, = plt.plot(bo.loop_state.X[:num_data_points], bo.loop_state.Y[:num_data_points], "bo", markersize=10, label="init_sample")
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

        plt.subplot(212)
        plt.ylabel(r"$reward$")
        plt.xlabel(r"$episode$")
        
        plotep = np.linspace(0, len(boPPOep_r), len(boPPOep_r))
        plt.plot(plotep, boPPOep_r,"r")
        plt.plot(plotep, boPPOep_r, "ro", markersize=5,label="ppo")
        plt.annotate("final reward = {}".format(boPPOep_r[-1]),xy=(plotep[-1],boPPOep_r[-1]), xycoords="data")

        fig1 = plt.gcf()
        # plt.show()
        plt.draw()
        fig1.savefig('image4/ep_{}.png'.format(ep), dpi=100)
    saver = tf.train.Saver()
    save_path = saver.save(ppo.sess, model_path)
    saver.restore(ppo.sess, model_path)
    ## continue training

    plt.figure()
    fig1 = plt.gcf()
    plt.ylabel(r"$reward$")
    plt.xlabel(r"$episode$")
    plotep = np.linspace(0, len(ppo.buffer_ep), len(ppo.buffer_ep))
    plt.plot(plotep, ppo.buffer_ep)
    plt.plot(plotep, ppo.buffer_ep, "ro", markersize=5)
    # plt.show()
    plt.draw()
    fig1.savefig('traingin_reward4.png', dpi=100)


