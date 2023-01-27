from random import randrange
from tcp_base import TcpEventBased

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


class TcpCubic(TcpEventBased):
    """docstring for TcpCubic"""
    def __init__(self):
        super(TcpCubic, self).__init__()
        self.C = 0.4 # scaling constant
        self.beta = 717 / 1024 # multiplicative decrease factor
        self.cnt = 0
        self.max_cnt = 0
        self.cWnd_cnt = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.K = 0
        self.T = 0
        self.max_cWnd = 1000000000 # window size where loss has occurred
        self.target = 0
        self.ack_cnt = 0
        self.tcp_cWnd = 0
        self.beforeFirstLoss = True
        self.save_new_cWnd = 90
        self.first = True
        self.dupCnt = True 

    def reset(self):
        self.C = 0.4 # scaling constant
        self.beta = 717 / 1024 # multiplicative decrease factor
        self.cnt = 0
        self.max_cnt = 0
        self.cWnd_cnt = 0
        self.epoch_start = 0
        self.origin_point = 0
        self.K = 0
        self.T = 0
        self.max_cWnd = 1000000000 # window size where loss has occurred
        self.target = 0
        self.ack_cnt = 0
        self.tcp_cWnd = 0
        self.beforeFirstLoss = True
        self.save_new_cWnd = 90
        self.first = True
        self.dupCnt = True 

    def get_action(self, obs, srtt, dMin, done, info):
        # unique socket ID
        socketUuid = obs[0]
        # TCP env type: event-based = 0 / time-based = 1
        envType = obs[1]
        # sim time in us
        simTime_us = obs[2]
        # unique node ID
        nodeId = obs[3]
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[7]
        # estimated bytes in flight
        bytesInFlight  = obs[8]
        # last estimation of RTT
        lastRtt_us  = obs[9]
        # min value of RTT
        minRtt_us  = obs[10]
        # function from Congestion Algorithm (CA) interface:
        #  GET_SS_THRESH = 0 (packet loss),
        #  INCREASE_WINDOW (packet acked),
        #  PKTS_ACKED (unused),
        #  CONGESTION_STATE_SET (unused),
        #  CWND_EVENT (unused),
        calledFunc = obs[11]
        # Congetsion Algorithm (CA) state:
        # 0 CA_OPEN = 0,
        # 1 CA_DISORDER,
        # 2 CA_CWR,
        # 3 CA_RECOVERY,
        # 4 CA_LOSS,
        # 5 CA_LAST_STATE
        caState = obs[12]
        # Congetsion Algorithm (CA) event:
        # 1 CA_EVENT_TX_START = 0,
        # 2 CA_EVENT_CWND_RESTART,
        # 3 CA_EVENT_COMPLETE_CWR,
        # 4 CA_EVENT_LOSS,
        # 5 CA_EVENT_ECN_NO_CE,
        # 6 CA_EVENT_ECN_IS_CE,
        # 7 CA_EVENT_DELAYED_ACK,
        # 8 CA_EVENT_NON_DELAYED_ACK,
        caEvent = obs[13]

        new_cWnd = segmentSize
        new_pacingRate = segmentSize
        

        #print (srtt)
        #print (obs[11], obs[12], obs[13])
        if (segmentsAcked):
            if (self.beforeFirstLoss): # IF NO LOSS HAS EVER HAPPENED, slowStart
                
                if (calledFunc == 0): # packetLoss
                    self.epoch_start = 0
                    self.max_cWnd = self.save_new_cWnd
                    # self.max_cWnd = cWnd
                    new_cWnd = int(max(self.save_new_cWnd * self.beta, 2 * segmentSize))
                    self.epoch_start = 0   
                    self.beforeFirstLoss = False   

                else: # slowStart
                    new_cWnd = int(cWnd + segmentSize)
                    #new_pacingRate = ssThresh #whyWillTheSimulationNotProceedIfnew_pacingRateIsTheAbove?

            else: # AFTER LOSS HAS HAPPENED, slowStart SHOULD NOT APPEAR AND Orca CWND REDUCTION SHOULD BE COUNTED
                # if loss: reduce cWnd
                # if cwnd <= save_cwnd (orca reduction) OR loss: epoch_start = 0 
                #### what if orca increment?            
                if (calledFunc == 0): # packetLoss 

                    self.max_cWnd = self.save_new_cWnd
                    new_cWnd = int(max(self.save_new_cWnd * self.beta, 2 * segmentSize))
                    self.epoch_start = 0                

                
                else: # congestionAvoidance
                    
                    # if new cWnd is smaller than last cubic cWnd decision, 
                    #if (cWnd_ < self.save_new_cWnd): self.epoch_start = simTime_us

                    #if ((self.epoch_start <= 0) | (cWnd != self.save_new_cWnd)): 
                    if (self.epoch_start <= 0): # Orca just change cwnd
                    # packetLoss (before) or OrcaCwndChange (now)

                        #if (cWnd != self.save_new_cWnd):
                        #    self.max_cWnd = self.save_new_cWnd

                        self.epoch_start = simTime_us
                        # ackCnt = 1
                        # tcpCwnd = cWnd

                        if (cWnd < self.max_cWnd): # cWnd is reduced
                            self.K = ((self.max_cWnd - cWnd) / (segmentSize * self.C))**(1/3)  
                            self.origin_point = self.max_cWnd / segmentSize

                        else: # cWnd is not reduced
                            self.K = 0
                            self.origin_point = cWnd / segmentSize
                    
                    t = simTime_us + dMin - self.epoch_start            
                    #new_cWnd = segmentSize * (self.origin_point + self.C * (((t/1000000)-self.K)**3))
                    self.target = self.origin_point + self.C * (((t/1000000)-self.K)**3)

                    if (self.target > (cWnd / segmentSize)): self.cnt = (cWnd / segmentSize) / (self.target-(cWnd / segmentSize))
                    else: self.cnt = 100 * cWnd / segmentSize
                    
                    if (self.cWnd_cnt > self.cnt): 
                        
                        new_cWnd = cWnd + segmentSize
                        self.cWnd_cnt = 0

                    else: 

                        new_cWnd = cWnd
                        self.cWnd_cnt += 1
        
        # clamp the maximum cwnd to not allow too much packet drops 
        if (new_cWnd > 100000): new_cWnd = 100000
        if (new_cWnd < 180): new_cWnd = 180
        if (self.beforeFirstLoss & calledFunc): new_pacingRate = int(2 * new_cWnd * 8 / (srtt / 1000) * (obs[6] + 60) / obs[6]) # pacingRate: mss -> mtu
        else: new_pacingRate = int(1.2 * max(new_cWnd, obs[8]) * 8 / (srtt / 1000) * (obs[6] + 60) / obs[6]) # pacingRate: mss -> mtu

        self.save_new_cWnd = new_cWnd
        self.save_new_pacingRate = new_pacingRate
        
        actions = [new_pacingRate, new_cWnd]      
        
        # Question 1. any loss result in zero cWnd, why the fuck?
        # Qeustion 2. activating slowStart result in         
        return actions, self.beforeFirstLoss
