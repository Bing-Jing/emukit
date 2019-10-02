
"""
code modify from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""
import tensorflow as tf
import numpy as np
import gym
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



class funcEnv():
    def __init__(self):
        self.curFun = None
        self.maxVal = 0
    def reset(self,sample_point = 1000,upper_bound = 1, lower_bound = 0):
        X = np.linspace(lower_bound, upper_bound, num=sample_point)[:, None]
        # 2. Specify the GP kernel (the smoothness of functions)
        # Smaller lengthscale => less smoothness
        kernel_var = 1.0
        kernel_lengthscale = 0.1
        kernel = kernel_var * RBF(kernel_lengthscale)
        # 3. Sample true function values for all inputs in X
        trueF = self.sample_true_u_functions(X, kernel)
        Y = trueF[0]
        self.curFun = interp1d(X.reshape(-1), Y, kind='cubic')
        self.maxVal = max(self.curFun(X))
        return self.curFun
    def getCurFun(self):
        return self.curFun
    # functions sampled from GP
    def sample_true_u_functions(self,X, kernel):
        u_task = np.empty(X.shape[0])
        mu = np.zeros(len(X)) # vector of the means
        C = kernel.__call__(X,X) # covariance matrix
        u_task[:,None] = np.random.multivariate_normal(mu,C).reshape(len(X),1)
        
        return [u_task]


###########################################################################################################################


model_path = "model/model.ckpt"


EP_MAX = 50
EP_LEN = 64
GAMMA = 0.98
LR = 0.0001
BATCH = 32
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
        # critic
        
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, activation =tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
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
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEPS)]
        
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):     
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        a = np.clip(a, 0, 1)

        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.ppoMax = max(self.ppoMax,self.curFun(a))
        r = -(self.funCurMax-self.ppoMax)
        r = float(r)
        self.ep_r += r
        self.buffer_r.append(r) 
        
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
        # s = np.array([[np.mean(mean),np.mean(variance)]])
        # a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        # a = np.clip(a, 0, 1)
        
        # self.buffer_s.append(s)
        # self.buffer_a.append(a)
        # self.ppoMax = max(self.ppoMax,self.curFun(a))
        # r = -(self.funCurMax-self.ppoMax)
        # r = float(r)
        # self.ep_r += r
        # self.buffer_r.append(r)       
        return evarr[np.newaxis, :]
    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False

if __name__ == "__main__":
    
    env = funcEnv()
    parameter_space = ParameterSpace([ContinuousParameter('x1', 0, 1)])
    all_ep_r = []
    num_data_points = 5
    fun = env.reset(upper_bound=1,lower_bound=0)
    design = RandomDesign(parameter_space)
    X = design.get_samples(num_data_points)
    Y = fun(X)
    model_gpy = GPRegression(X,Y) # Train and wrap the model in Emukit
    model_emukit = GPyModelWrapper(model_gpy)
    ppo = PPO(model = model_emukit)
    
    for ep in range(EP_MAX):
        
        fun = env.reset(upper_bound=1,lower_bound=0)
        ppo.ppoMax = 0
        ppo.ep_r = 0
        ppo.funCurMax = env.maxVal
        ppo.curFun = env.getCurFun()

        design = RandomDesign(parameter_space) # Collect random points
        num_data_points = 3
        X = design.get_samples(num_data_points)
        Y = fun(X)
        model_gpy = GPRegression(X,Y) # Train and wrap the model in Emukit
        model_emukit = GPyModelWrapper(model_gpy)

        bo = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = ppo,
                                         batch_size = 1)
        for t in range(EP_LEN):    # in one episode
        
            bo.run_loop(fun, 1)
            mu_plot, var_plot = bo.model.predict(np.array(ppo.buffer_a[-1]).reshape(-1,1))
            s_ = np.concatenate((mu_plot,var_plot),axis=1)

            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                print("updating.......")
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in ppo.buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(ppo.buffer_s), np.vstack(ppo.buffer_a), np.array(discounted_r)[:, np.newaxis]
                ppo.buffer_s, ppo.buffer_a, ppo.buffer_r = [], [], []
                ppo.update(bs, ba, br)
        print("current ep = {} reward = {}".format(ep,ppo.ep_r))

        ### plot function
        x_plot = np.linspace(0, 1, 200)[:, None]
        y_plot = fun(x_plot)
        mu_plot, var_plot = bo.model.predict(x_plot)

        plt.figure(figsize=(12, 8))
        plt.plot(bo.loop_state.X, bo.loop_state.Y, "ro", markersize=10, label="Observations")
        plt.plot(x_plot, y_plot, "k", label="Objective Function")
        plt.plot(x_plot, mu_plot, "C0", label="Model")
        plt.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)

        plt.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)

        plt.fill_between(x_plot[:, 0],
                        mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                        mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
        plt.legend(loc=2, prop={'size': 15})
        plt.xlabel(r"$x$")
        plt.ylabel(r"$f(x)$")
        plt.grid(True)
        plt.xlim(0, 1)

        fig1 = plt.gcf()
        # plt.show()
        plt.draw()
        fig1.savefig('image/ep_{}.png'.format(ep), dpi=100)
    saver = tf.train.Saver()
    save_path = saver.save(ppo.sess, model_path)
    saver.restore(ppo.sess, model_path)
# env = gym.make('Pendulum-v0').unwrapped
# ppo = PPO()
# all_ep_r = []

# for ep in range(EP_MAX):
#     s = env.reset()
#     buffer_s, buffer_a, buffer_r = [], [], []
#     ep_r = 0
#     for t in range(EP_LEN):    # in one episode
#         env.render()
#         a = ppo.choose_action(s)
#         s_, r, done, _ = env.step(a)
#         buffer_s.append(s)
#         buffer_a.append(a)
#         buffer_r.append((r+8)/8)    # normalize reward, find to be useful
#         s = s_
#         ep_r += r

#         # update ppo
#         if (t+1) % BATCH == 0 or t == EP_LEN-1:
#             v_s_ = ppo.get_v(s_)
#             discounted_r = []
#             for r in buffer_r[::-1]:
#                 v_s_ = r + GAMMA * v_s_
#                 discounted_r.append(v_s_)
#             discounted_r.reverse()

#             bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
#             buffer_s, buffer_a, buffer_r = [], [], []
#             ppo.update(bs, ba, br)
#     if ep == 0: all_ep_r.append(ep_r)
#     else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
#     print(
#         'Ep: %i' % ep,
#         "|Ep_r: %i" % ep_r,
#         ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
#     )
