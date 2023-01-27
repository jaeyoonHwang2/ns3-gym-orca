#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. COPA with various velocity
2. No slow start
"""
import numpy as np
from collections import deque
from tcp_base import TcpEventBased
import random

class Sender_COPA(TcpEventBased):

    def __init__(self, mode = 'ss'):

        super(Sender_COPA, self).__init__()
        
        # hwang var
        self.velocity_hwang = 1 # initial value
        self.directionUp = True 
        self.saveLastCwnd = 0 
        self.saveLastDirection = True
        self.numRtt = 0 
    
    #def __init__(self, segment_size=1, delta = 0.5, flow_ID = 0, initial_cwnd = 1 , mode = 'ss', v = 1, eq = 1, time = 1):
    #    super(Sender_COPA, self).__init__()
    #    # For data monitoring
    #    self.g = 0.1
    #    self.RTT_list = deque([],maxlen = 2000)
    #    self.min_RTT = 100 # minimum RTT so far
    #    self.srtt = 0 # standing RTT
    #    self.max_thp = 0 # maximum throughput so far
    #    self.last_time = 0 # last time ack has arrived
    #    self.delta = delta
    #    self.d_q = 0
    #    # For velocity update
    #    self.prev = initial_cwnd  # cwnd
    #    self.v_update = 0
    #    #For convenience
    #    self.previous_rate = 0
    #    self.current_cwnd_packets = initial_cwnd
    #    self.RTT_standing = 0
    #    self.target = 0
    #    self.current_rate = 0
    #    self.mode = mode # ss for slow start, n for normal
    #    self.last_update = 0
    #    self.counter = 0
    #    # number
    #    self.num_ack_rmn = 1
    #    self.pkt_num = 0 # # of packet sent
    #    self.ack_num = 0 # # of ack received
    #    # number of packets actually sent  = new_cwnd - num_ack_rmn
    #    self.is_eq = 1
    #    self.eq = eq
    #    self.constant_time = time
    #    # hwang var
    #    self.velocity_hwang = 1 # initial value
    #    self.directionUp = True 
    #    self.saveLastCwnd = 0 
    #    self.saveLastDirection = True
    #    self.numRtt = 0 
        

    def reset(self, time):
        # For data monitoring
        self.g = 0.1
        self.RTT_list = deque([], maxlen=2000)
        self.min_RTT = 100  # minimum RTT so far
        self.srtt = 0  # standing RTT
        self.max_thp = 0  # maximum throughput so far
        self.last_time = 0  # last time ack has arrived
        self.d_q = 0
        # For velocity update
        self.num_RTT = 3  # number of RTT for velocity update
        self.prev = 1  # cwnd
        self.v_update = 0
        # For convenience
        self.previous_rate = 0
        self.current_cwnd_packets = 1
        self.RTT_standing = 0.002
        self.target = 0
        self.current_rate = 0
        self.last_update = time
        self.counter = 0
        # number
        self.num_ack_rmn = 1
        self.pkt_num = 0  # # of packet sent
        self.ack_num = 0  # # of ack received
        # number of packets actually sent  = new_cwnd - num_ack_rmn
        self.is_eq = 1
        self.constant_time = 1
        # hwang var
        self.velocity_hwang = 1 # initial value
        self.directionUp = True 
        self.saveLastCwnd = 0 
        self.saveLastDirection = True
        self.numRtt = 0 
        


    def record_RTT(self, obs):
        now = obs[2] * 0.000001
        last_RTT = obs[9] * 0.000001 
        if not(self.counter):
            self.srtt = last_RTT
        self.RTT_list.append([now, last_RTT])
        #print(last_RTT,self.min_RTT)

    def update_min_RTT(self, obs):
        min_RTT = obs[10]*0.000001 
        if self.min_RTT>min_RTT:
            self.min_RTT = min_RTT

    def update_cwnd_ss(self, target, obs, RTT_standing):
        current_rate = (obs[5] / obs[6]) / RTT_standing
        # obs[5]/obs[6] : number of packets => current rate : number of packets per unit time

        self.current_rate = current_rate
        if current_rate > target:
            self.mode = 'n' # convert to the normal mode
            new_cwnd_packets = self.current_cwnd_packets
        else:
            if obs[2]*0.000001 > self.last_update + RTT_standing + self.min_RTT:
                self.last_update = obs[2] * 0.000001
                new_cwnd_packets = 2*self.current_cwnd_packets
                self.current_cwnd_packets = new_cwnd_packets
            else:
                new_cwnd_packets = self.current_cwnd_packets
        return new_cwnd_packets

    def get_action(self, obs):
        current_time = obs[2]*0.000001
        self.v_update = 0
        # RTT record
        self.record_RTT(obs)
        self.update_min_RTT(obs)
        RTT_standing = self.estimate_RTT_standing(obs)
        self.RTT_standing = RTT_standing
        # Update queuing delay
        self.d_q = self.RTT_standing - self.min_RTT
        # STEP1-1
        # and srtt using the standard TCP exponentially weighted moving average estimator
        last_RTT = obs[9] * 0.000001
        self.get_srtt(last_RTT)  # update smoothed RTT
        # STEP1-2
        # set target according to
        target = self.calculate_target(RTT_standing)
        # STEP2
        self.target = target
        # update cwnd / velocity
        if self.mode == 'ss': #slow start
            new_cwnd_packets = self.update_cwnd_ss(target, obs, RTT_standing)
        else: # normal mode
            new_cwnd_packets = self.update_cwnd(target, obs, RTT_standing)
        new_cwnd_bytes = int(max(2, new_cwnd_packets)) 
        # STEP3
        new_pacing_rate = int(2 * (new_cwnd_bytes / self.RTT_standing) * 8)
        action = [new_pacing_rate, new_cwnd_bytes]

        self.counter += 1
        return action


    def update_velocity_hwang(self, obs):
        
        #####
        if (self.current_cwnd_packets > self.saveLastCwnd): self.directionUp = True
        elif (self.current_cwnd_packets < self.saveLastCwnd): self.directionUp = False
        else: self.directionUp = ~(self.directionUp)

        #print ("CURRENT CWND", self.current_cwnd_packets, "LAST CWND", self.saveLastCwnd)

        if (self.directionUp == self.saveLastDirection): #
            if (self.numRtt < 3): self.numRtt += 1
            else: self.velocity_hwang *= 2

        else: # direction changed 
            self.numRtt = 0 
            self.velocity_hwang = 1
            
        self.saveLastCwnd = self.current_cwnd_packets
        self.saveLastDirection = self.directionUp
        
        print (obs[0]-1, 'velocity', obs[2] / 1000000, self.velocity_hwang)


    def update_cwnd(self, target, obs, RTT_standing):
        
        current_rate = (obs[5] / obs[6]) / RTT_standing
        self.current_rate = current_rate
        #new_cwnd_packets = self.current_cwnd_packets + self.velocity_hwang/(self.delta*self.current_cwnd_packets) \
        #    if current_rate < target else max(self.current_cwnd_packets - self.velocity_hwang/(self.delta * self.current_cwnd_packets), 2)
        
        if current_rate < target: new_cwnd_packets = self.current_cwnd_packets + self.velocity_hwang/(self.delta*self.current_cwnd_packets)
        else:            
            #new_cwnd_packets = max(int(self.current_cwnd_packets - self.velocity_hwang/(self.delta * self.current_cwnd_packets)), 2)
            new_cwnd_packets = max(self.current_cwnd_packets - self.velocity_hwang/(self.delta * self.current_cwnd_packets), 2)
        self.current_cwnd_packets = new_cwnd_packets
        
        return new_cwnd_packets


    def get_srtt(self, last_RTT):
        ##### the first ten samples should be given as mean 
        self.srtt = (1-self.g)*self.srtt + self.g*last_RTT        


    def estimate_RTT_standing(self, obs):
        now = obs[2] * 0.000001
        tau = self.srtt / 2
        time = now - tau
        '''
        len_ = int(len(self.RTT_list))  # length of the memory
        idx = len_ - 1
        for i in range(len_):  # search from the most recent data
            if self.RTT_list[len_ - (1 + i)][0] <= time:  # [time, RTT]
                idx = i - 1  # i+1 elements
                break
        RTT = [list(self.RTT_list)[-(idx + 1):][i][1] for i in range(idx + 1)]
        '''
        len_ = int(len(self.RTT_list))
        idx = 0
        for i in range(len_):
            if self.RTT_list[i][0] > time:
                idx = i
                break

        standingRTT = self.RTT_list[idx][1]
        
        for i in range (len_-idx):
           
            if (self.RTT_list[idx+i][1] < standingRTT): standingRTT = self.RTT_list[idx+i][1]

        return standingRTT


    def calculate_target(self, RTT_standing):
        #target = 1/(self.delta*(RTT_standing-self.min_RTT)) if RTT_standing > self.min_RTT else 10000
        if RTT_standing > self.min_RTT: target = 1/(self.delta*(RTT_standing-self.min_RTT)) 
        else: target = 10000000000
        #print ("STANDING RTT ", RTT_standing, "MIN RTT", self.min_RTT)
        return target
