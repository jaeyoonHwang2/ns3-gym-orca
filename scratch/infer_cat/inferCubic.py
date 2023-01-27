'''
MIT License
Copyright (c) Chen-Yu Yen - Soheil Abbasloo 2020
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from telnetlib import DM
import threading
import logging
import tensorflow as tf
import sys
from agent import Agent
import os

from tcpCubic import TcpCubic
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import gym
import numpy as np
import time
import random
import datetime
import signal
import pickle
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper
import json
from ns3gym import ns3env
import math

tf.get_logger().setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
parser.add_argument('--eval', action='store_true', default=True, help='default is  %(default)s')
parser.add_argument('--tb_interval', type=int, default=1)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--mem_r', type=int, default = 123456)
parser.add_argument('--mem_w', type=int, default = 12345)
base_path = '../../../Desktop/inferLearner'
job_name = 'actor'  


# parameters from parser
global config
global params
config = parser.parse_args()

# parameters from .json file
params = Params(os.path.join(base_path,'params.json'))
seed = 12 
debug = False  

# parameters for config
task = 0
startSim = 0.00001
port = 6025+task
stepTime = 0.000001
simTime = 20
simArgs = {"--duration": simTime, }

# monitoring variables
numSenders = 2
mtp = 20000
iteration = 0

alpha = [1 for i in range (numSenders)]
ack_count = [0 for i in range (numSenders)] 
loss_count = [0 for i in range (numSenders)]
rtt_sum = [0 for i in range (numSenders)] 
min_rtt = [0 for i in range (numSenders)]
srtt = [0 for i in range (numSenders)]
timestamp = [0 for i in range (numSenders)] 
interval_cnt = [0 for i in range (numSenders)]
throughput = [0 for i in range (numSenders)]
loss_rate = [0 for i in range (numSenders)]
rtt = [0 for i in range (numSenders)]
epoch = [0 for i in range (numSenders)]
epoch_ = [0 for i in range (numSenders)]
r = [0 for i in range (numSenders)]
ret = [0 for i in range (numSenders)]

if params.dict['single_actor_eval']:
    local_job_device = ''
    shared_job_device = ''
    def is_actor_fn(i): return True
    global_variable_device = '/gpu'
    is_learner = False
    server = tf.train.Server.create_local_server()
    filters = []
    
else:

    local_job_device = '/job:%s/task:%d' % (job_name, task)
    shared_job_device = '/job:learner/task:0'

    is_learner = job_name == 'learner'
    global_variable_device = shared_job_device + '/gpu'

    def is_actor_fn(i): return job_name == 'actor' and i == task

    if params.dict['remote']:
        cluster = tf.train.ClusterSpec({
            'actor': params.dict['actor_ip'][:params.dict['num_actors']],
            'learner': [params.dict['learner_ip']]
        })
    else:
        cluster = tf.train.ClusterSpec({
                'actor': ['localhost:%d' % (8001 + i) for i in range(params.dict['num_actors'])],
                'learner': ['localhost:8000']
            })

    server = tf.train.Server(cluster, job_name=job_name,
                            task_index=task)
    filters = [shared_job_device, local_job_device]

if params.dict['use_TCP']:
    env_str = "TCP"
    env_peek = TCP_Env_Wrapper(env_str, params,use_normalizer=params.dict['use_normalizer'])

else:
    env_str = 'YourEnvironment'
    env_peek =  Env_Wrapper(env_str)

s_dim, a_dim = env_peek.get_dims_info()
action_scale, action_range = env_peek.get_action_info()

if not params.dict['use_TCP']:
    params.dict['state_dim'] = s_dim
if params.dict['recurrent']:
    s_dim = s_dim * params.dict['rec_dim']

if params.dict['use_hard_target'] == True:
    params.dict['tau'] = 1.0

with tf.Graph().as_default(),\
    tf.device(local_job_device + '/cpu'):

    tf.set_random_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    actor_op = []
    check_op = []
    now = datetime.datetime.now()
    tfeventdir = os.path.join( base_path, params.dict['logdir'], job_name+str(task) )
    params.dict['train_dir'] = tfeventdir

    if not os.path.exists(tfeventdir):
        os.makedirs(tfeventdir)
    summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

    with tf.device(shared_job_device):

        agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        cubicAgent = [TcpCubic() for i in range(numSenders)]
        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
        queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

    if is_learner:
        with tf.device(params.dict['device']):
            agent.build_learn()

            agent.create_tf_summary()

        if config.load is True and config.eval==False:
            if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                    replay_memory = pickle.load(fp)

    for i in range(params.dict['num_actors']):
        if is_actor_fn(i):
            env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, use_normalizer=params.dict['use_normalizer']) 
            envNs3 = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

            a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0') 
            a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
            a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward') 
            a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1') 
            a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal') 
            a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

            with tf.device(shared_job_device):
                actor_op.append(queue.enqueue(a_buf))

    if is_learner:
        Dequeue_Length = params.dict['dequeue_length']
        dequeue = queue.dequeue_many(Dequeue_Length)

    queuesize_op = queue.size()

    if params.dict['ckptdir'] is not None:
        params.dict['ckptdir'] = os.path.join( base_path, params.dict['ckptdir'])
        print("## checkpoint dir:", params.dict['ckptdir'])
        isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
        print("## checkpoint exists?:", isckpt)
        if isckpt== False:
            print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
    else:
        params.dict['ckptdir'] = tfeventdir

    tfconfig = tf.ConfigProto(allow_soft_placement=True)

    if params.dict['single_actor_eval']:
        mon_sess = tf.train.SingularMonitoredSession(
            checkpoint_dir=params.dict['ckptdir'])
    else:
        mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                save_checkpoint_secs=15,
                save_summaries_secs=None,
                save_summaries_steps=None,
                is_chief=is_learner,
                checkpoint_dir=params.dict['ckptdir'],
                config=tfconfig,
                hooks=None)

    agent.assign_sess(mon_sess)
    

    obs = envNs3.reset()
    
    Uuid = obs[0] - 1

    done = False
    info = None

    # init. variables

    alpha = [1 for i in range (numSenders)]
    a = [0 for i in range (numSenders)]
    ack_count = [0 for i in range (numSenders)] 
    loss_count = [0 for i in range (numSenders)]
    rtt_sum = [0 for i in range (numSenders)] 
    min_rtt = [0 for i in range (numSenders)]
    srtt = [0 for i in range (numSenders)]
    timestamp = [0 for i in range (numSenders)] 
    interval_cnt = [0 for i in range (numSenders)]
    throughput = [0 for i in range (numSenders)]
    loss_rate = [0 for i in range (numSenders)]
    rtt = [0 for i in range (numSenders)]
    epoch = [0 for i in range (numSenders)]
    epoch_ = [0 for i in range (numSenders)]
    r = [0 for i in range (numSenders)]
    ret = [0 for i in range (numSenders)]
    action = [[0, 0] for i in range (numSenders)]
    orca_init = [True for i in range (numSenders)]
    slow_start = [True for i in range (numSenders)]
    terminal = [False for i in range (numSenders)]
    error_code = [True for i in range (numSenders)]

    s0 = [np.zeros([s_dim]) for i in range (numSenders)]
    s1 = [np.zeros([s_dim]) for i in range (numSenders)]
    s0_rec_buffer = [np.zeros([s_dim]) for i in range (numSenders)]
    s1_rec_buffer = [np.zeros([s_dim]) for i in range (numSenders)]
    if params.dict['recurrent']: a[Uuid] = agent.get_action(s0_rec_buffer[Uuid], not config.eval)
    else: a[Uuid] = agent.get_action(s0[Uuid], not config.eval)
    a[Uuid] = a[0][0]
    
    while True: # one iteration
    
        Uuid = obs[0] - 1
        
        # increase counts
        ack_count[Uuid] += 1
        if not (obs[11]): loss_count[Uuid] += 1
        rtt_sum[Uuid] += obs[9] / 1000

        if not (min_rtt[Uuid]): min_rtt[Uuid] = obs[9] / 1000 # min_rtt_ms
        else: min_rtt[Uuid] = min(min_rtt[Uuid], obs[9] / 1000)

        if (srtt[Uuid]): srtt[Uuid] = 7/8 * srtt[Uuid] + 1/8 * obs[9] / 1000 # srtt_ms 
        else: srtt[Uuid] = obs[9] / 1000

        # mtp has passed from the last report 
        # -> Orca will take into action
        if (int(obs[2] / mtp) != timestamp[Uuid]):
            
            if not (timestamp[Uuid]): timestamp[Uuid] = int(obs[2] / mtp) - 1
            
            interval_cnt[Uuid] = int(obs[2] / mtp) - timestamp[Uuid]

            # throughput / loss rate / rtt is calculated
            throughput[Uuid] = ack_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
            if (throughput[Uuid] > 100): throughput[Uuid] = 100
            loss_rate[Uuid] = loss_count[Uuid] * 1500 * 8 / (interval_cnt[Uuid] * (mtp / (1000 * 1000))) / (1024 * 1024) # Mbps
            rtt[Uuid] = rtt_sum[Uuid] / ack_count[Uuid] # ms

            # print throughput / loss rate / rtt
            for i in range(interval_cnt[Uuid]):
                print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "throughput", throughput[Uuid])
                print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "loss_rate", loss_rate[Uuid])
                print (Uuid, (timestamp[Uuid] + 1 + i) * 0.02, "srtt", srtt[Uuid])
            
                ack_count[Uuid] = 0
                loss_count[Uuid] = 0
                rtt_sum[Uuid] = 0

                timestamp[Uuid] = int(obs[2] / mtp)

        
        action[Uuid], slow_start[Uuid] = cubicAgent[Uuid].get_action(obs, srtt[Uuid], min_rtt[Uuid], done, info)
        print (Uuid, (obs[2] / 1000000), "cubic_cwnd", action[Uuid][1])

        obs, reward, done, info = envNs3.step(action[Uuid])


        if done: 

            done = False
            info = None
            iteration += 1 
            print ("An episode is over")
            break
