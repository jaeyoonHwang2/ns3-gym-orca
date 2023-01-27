/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on script: ./examples/tcp/tcp-variants-comparison.cc
 *
 * Topology:
 *
 *   Right Leafs (Clients)                      Left Leafs (Sinks)
 *           |            \                    /        |
 *           |             \    bottleneck    /         |
 *           |              R0--------------R1          |
 *           |             /                  \         |
 *           |   access   /                    \ access |
 *           N -----------                      --------N
 */

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/applications-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"

#include "ns3/opengym-module.h"
#include "tcp-rl.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpVariantsComparison");

static std::vector<uint32_t> rxPkts;

static void
CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address & srcAddr)
{
  rxPkts[sinkId]++;
}


static void
PrintRxCount()
{
  uint32_t size = rxPkts.size();
  NS_LOG_UNCOND("RxPkts:");
  for (uint32_t i=0; i<size; i++){
    NS_LOG_UNCOND("---SinkId: "<< i << " RxPkts: " << rxPkts.at(i));
  }
}

void ThroughputMonitor (FlowMonitorHelper *fmhelper, Ptr<FlowMonitor> flowMon)
	{
		static int flag = 0;
		//for single flow
		//double thr = 0;
		static int last_num_packet = 0;
		flag ++;
		std::ofstream outfile;
		outfile.open(".//scratch//Equilibrium_distance//result//20211206//throughput_100M.txt", std::ios_base::app);
		std::map<FlowId, FlowMonitor::FlowStats> flowStats = flowMon->GetFlowStats();
		//Ptr<Ipv4FlowClassifier> classing = DynamicCast<Ipv4FlowClassifier> (fmhelper->GetClassifier());
		//outfile << "================================================================================="<<std::endl;
		for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator stats = flowStats.begin (); stats != flowStats.end (); ++stats)
		{
			//Ipv4FlowClassifier::FiveTuple fiveTuple = classing->FindFlow (stats->first);
			//outfile << "Flow ID                       :" << stats->first <<" ;  "<<fiveTuple.sourceAddress<<" -----> "<<fiveTuple.destinationAddress<<std::endl;
			//std::cout << "Flow ID                       :" << stats->first <<" ;  "<<fiveTuple.sourceAddress<<" -----> "<<fiveTuple.destinationAddress<<std::endl;
                	//outfile << "Duration              :" << stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds() <<std::endl;
            //outfile <<"Throughput: " << stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024  << " Mbps"<<std::endl;
            //outfile << stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024  <<std::endl;
            //thr += stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024;
            //for single flow

            if (stats->first == 1 and flag%2 ==0)
            {
                //outfile << "Flow ID                       :" << stats->first <<" ;  "<<fiveTuple.sourceAddress<<" -----> "<<fiveTuple.destinationAddress<<std::endl;
                //outfile <<"rxBytes "<<stats->second.rxBytes<< std::endl;
                //outfile <<"Last "<<stats->second.timeLastRxPacket.GetSeconds() <<"Sec"<< std::endl;
                //outfile <<"First "<<stats->second.timeFirstTxPacket.GetSeconds() <<"Sec"<< std::endl;
                //outfile <<"Throughput : "<< stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024 << std::endl;
                //outfile <<"Throughput : "<< (stats->second.rxBytes-last_num_packet) * 8.0 / (0.02)/1024/1024 << std::endl;
                outfile << (stats->second.rxBytes-last_num_packet) * 8.0 / (0.01)/1024/1024 << std::endl;
                //outfile << Simulator::Now()	<< std::endl;
                //outfile << Simulator::Now()	<< std::endl;
                //outfile << "Flag: "<<flag<< std::endl;
                //outfile << "Last num packet: "<<last_num_packet<< std::endl;
                last_num_packet = stats->second.rxBytes;
            }
            }
		    //outfile << thr<< std::endl;
		    //outfile << "Flag: "<<flag<< std::endl;
		    outfile.close();
			Simulator::Schedule(Seconds(0.04),&ThroughputMonitor, fmhelper, flowMon);
	}
/////
static void
ChangeBW(NetDeviceContainer *dev) {
    static int flag = 0;
    std::ofstream outfile1;
    outfile1.open(".//scratch//New_topology//result//20211215//ChangeBW.txt", std::ios_base::app);
    if (flag==0){
    dev->Get(0)-> SetAttribute ("DataRate", StringValue ("20Mbps"));
    dev->Get(1)-> SetAttribute ("DataRate", StringValue ("20Mbps"));
    outfile1 << "BW changed to 20 " <<flag<< "  Time: "<<Simulator::Now()/1000000000 <<std::endl;
    Simulator::Schedule(Seconds(10), &ChangeBW, dev);
    }
    if (flag==1){
    dev->Get(0)-> SetAttribute ("DataRate", StringValue ("20Mbps"));
    dev->Get(1)-> SetAttribute ("DataRate", StringValue ("20Mbps"));
    outfile1 << "BW changed to 20 " <<flag<< "  Time: "<<Simulator::Now()/1000000000 <<std::endl;
    Simulator::Schedule(Seconds(10), &ChangeBW, dev);
    }
    outfile1.close();
    flag ++;
}


int main (int argc, char *argv[])
{
  uint32_t openGymPort = 5555;
  double tcpEnvTimeStep = 0.1;

  uint32_t nLeaf = 1;
  std::string transport_prot = "TcpRl";
  double error_p = 0.0;
  std::string bottleneck_bandwidth = "10Mbps";//"130Mbps";//"2Mbps";//
  std::string bottleneck_delay = "0.001ms";
  std::string access_bandwidth = "20000Mbps";//"10Mbps";//
  std::string access_delay = "5ms";
  std::string prefix_file_name = "TcpVariantsComparison";
  uint64_t data_mbytes = 0;
  uint32_t mtu_bytes = 150;//400;
  double duration = 20.0;
  uint32_t run = 0;
  bool flow_monitor = false;
  bool sack = true;
  std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
  std::string recovery = "ns3::TcpClassicRecovery";

  bool isPacingEnabled = true;
  //std::string pacingRate = "50Mbps";//"60Mbps";//"0.9Mbps";//

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", run);
  cmd.AddValue ("envTimeStep", "Time step interval for time-based TCP env [s]. Default: 0.1s", tcpEnvTimeStep);
  // other parameters
  cmd.AddValue ("nLeaf",     "Number of left and right side leaf nodes", nLeaf);
  cmd.AddValue ("transport_prot", "Transport protocol to use: TcpNewReno, "
                "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                "TcpBic, TcpYeah, TcpIllinois, TcpWestwood, TcpWestwoodPlus, TcpLedbat, "
		            "TcpLp, TcpRl, TcpRlTimeBased", transport_prot);
  cmd.AddValue ("error_p", "Packet error rate", error_p);
  cmd.AddValue ("bottleneck_bandwidth", "Bottleneck bandwidth", bottleneck_bandwidth);
  cmd.AddValue ("bottleneck_delay", "Bottleneck delay", bottleneck_delay);
  cmd.AddValue ("access_bandwidth", "Access link bandwidth", access_bandwidth);
  cmd.AddValue ("access_delay", "Access link delay", access_delay);
  cmd.AddValue ("prefix_name", "Prefix of output trace file", prefix_file_name);
  cmd.AddValue ("data", "Number of Megabytes of data to transmit", data_mbytes);
  cmd.AddValue ("mtu", "Size of IP packets to send in bytes", mtu_bytes);
  cmd.AddValue ("duration", "Time to allow flows to run in seconds", duration);
  cmd.AddValue ("run", "Run index (for setting repeatable seeds)", run);
  cmd.AddValue ("flow_monitor", "Enable flow monitor", flow_monitor);
  cmd.AddValue ("queue_disc_type", "Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)", queue_disc_type);
  cmd.AddValue ("sack", "Enable or disable SACK option", sack);
  cmd.AddValue ("recovery", "Recovery algorithm type to use (e.g., ns3::TcpPrrRecovery", recovery);
  cmd.Parse (argc, argv);

  transport_prot = std::string ("ns3::") + transport_prot;

  SeedManager::SetSeed (1);
  SeedManager::SetRun (run);

  NS_LOG_UNCOND("Ns3Env parameters:");
  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  } else {
    NS_LOG_UNCOND("--openGymPort: No OpenGym");
  }

  NS_LOG_UNCOND("--seed: " << run);
  NS_LOG_UNCOND("--Tcp version: " << transport_prot);


  // OpenGym Env --- has to be created before any other thing
  Ptr<OpenGymInterface> openGymInterface;
  if (transport_prot.compare ("ns3::TcpRl") == 0)
  {
    openGymInterface = OpenGymInterface::Get(openGymPort);
    Config::SetDefault ("ns3::TcpRl::Reward", DoubleValue (2.0)); // Reward when increasing congestion window
    Config::SetDefault ("ns3::TcpRl::Penalty", DoubleValue (-30.0)); // Penalty when decreasing congestion window
  }

  if (transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    openGymInterface = OpenGymInterface::Get(openGymPort);
    Config::SetDefault ("ns3::TcpRlTimeBased::StepTime", TimeValue (Seconds(tcpEnvTimeStep))); // Time step of TCP env
  }

  // Calculate the ADU size
  Header* temp_header = new Ipv4Header ();
  uint32_t ip_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("IP Header size is: " << ip_header);
  delete temp_header;
  temp_header = new TcpHeader ();
  uint32_t tcp_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("TCP Header size is: " << tcp_header);
  delete temp_header;
  uint32_t tcp_adu_size = mtu_bytes - 20 - (ip_header + tcp_header);
  NS_LOG_LOGIC ("TCP ADU size is: " << tcp_adu_size);

  // Set the simulation start and stop time
  double start_time = 0.001;
  double stop_time = start_time + duration;

  // 4 MB of TCP buffer
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (sack));
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (1));
  Config::SetDefault ("ns3::TcpSocketState::EnablePacing", BooleanValue (isPacingEnabled));
  //Config::SetDefault ("ns3::TcpSocketState::MaxPacingRate", StringValue (pacingRate));


  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType",
                      TypeIdValue (TypeId::LookupByName (recovery)));
  // Select TCP variant
  if (transport_prot.compare ("ns3::TcpWestwoodPlus") == 0)
    {
      // TcpWestwoodPlus is not an actual TypeId name; we need TcpWestwood here
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TcpWestwood::GetTypeId ()));
      // the default protocol type in ns3::TcpWestwood is WESTWOOD
      Config::SetDefault ("ns3::TcpWestwood::ProtocolType", EnumValue (TcpWestwood::WESTWOODPLUS));
    }
  else
    {
      TypeId tcpTid;
      NS_ABORT_MSG_UNLESS (TypeId::LookupByNameFailSafe (transport_prot, &tcpTid), "TypeId " << transport_prot << " not found");
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TypeId::LookupByName (transport_prot)));
    }

  // Configure the error model
  // Here we use RateErrorModel with packet error rate
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  uv->SetStream (50);
  RateErrorModel error_model;
  error_model.SetRandomVariable (uv);
  error_model.SetUnit (RateErrorModel::ERROR_UNIT_PACKET);
  error_model.SetRate (error_p);
  ///////////////////////
  /*
  NodeContainer nodes1;
  nodes1.Create (2);
  NodeContainer nodes2;
  nodes2.Create (2);
  */
  //////////////////
  // Create the point-to-point link helpers
  PointToPointHelper bottleNeckLink;
  bottleNeckLink.SetDeviceAttribute  ("DataRate", StringValue (bottleneck_bandwidth));
  bottleNeckLink.SetChannelAttribute ("Delay", StringValue (bottleneck_delay));
  //bottleNeckLink.SetDeviceAttribute  ("ReceiveErrorModel", PointerValue (&error_model));

  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute  ("DataRate", StringValue (access_bandwidth));
  pointToPointLeaf.SetChannelAttribute ("Delay", StringValue (access_delay));

  PointToPointDumbbellHelper d (nLeaf, pointToPointLeaf,
                                nLeaf, pointToPointLeaf,
                                bottleNeckLink);
  //////////////
  /*
  NetDeviceContainer devices1;
  devices1 = bottleNeckLink.Install (nodes1);
  NetDeviceContainer devices2;
  devices2 = pointToPointLeaf.Install (nodes2);
  */
  ///////////////
  //NetDeviceContainer devices1;
  //devices1 = d.GetLeft()->GetDevice(0); // 0 is the router
  //NetDeviceContainer devices2;
  //devices2 = d.GetRibottleneckRouterght()->GetDevice(0);  // 0 is the router

  //////
  NetDeviceContainer *bottleneckRouter;
  bottleneckRouter = &d.m_routerDevices;
  bottleneckRouter->Get(0)->SetAttribute  ("DataRate", StringValue ("10Mbps"));
  bottleneckRouter->Get(1)->SetAttribute  ("DataRate", StringValue ("10Mbps"));


  // Install IP stack
  InternetStackHelper stack;
  stack.InstallAll ();

  // Traffic Control
  TrafficControlHelper tchPfifo;
  tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");

  TrafficControlHelper tchCoDel;
  tchCoDel.SetRootQueueDisc ("ns3::CoDelQueueDisc");

  DataRate access_b (access_bandwidth);
  DataRate bottle_b (bottleneck_bandwidth);
  Time access_d (access_delay);
  Time bottle_d (bottleneck_delay);

  uint32_t size = static_cast<uint32_t>((std::min (access_b, bottle_b).GetBitRate () / 8) *
    ((access_d + bottle_d + access_d) * 2).GetSeconds () * 5); // switch queue size = 5 * BDP

  Config::SetDefault ("ns3::PfifoFastQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, size / mtu_bytes)));

  
  //std::ofstream outfile;
  //outfile.open(".//scratch//Reno_vs_COPA//queue_size.txt", std::ios_base::app);

  Config::SetDefault ("ns3::CoDelQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, size)));

  std::cout << "Size " << size << std::endl;
  
  if (queue_disc_type.compare ("ns3::PfifoFastQueueDisc") == 0)
  {
    tchPfifo.Install (d.GetLeft()->GetDevice(1));
    tchPfifo.Install (d.GetRight()->GetDevice(1));
  }
  else if (queue_disc_type.compare ("ns3::CoDelQueueDisc") == 0)
  {
    tchCoDel.Install (d.GetLeft()->GetDevice(1));
    tchCoDel.Install (d.GetRight()->GetDevice(1));
  }
  else
  {
    NS_FATAL_ERROR ("Queue not recognized. Allowed values are ns3::CoDelQueueDisc or ns3::PfifoFastQueueDisc");
  }

  // Assign IP Addresses
  d.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.2.1.0", "255.255.255.0"),
                         Ipv4AddressHelper ("10.3.1.0", "255.255.255.0"));


  NS_LOG_INFO ("Initialize Global Routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Install apps in left and right nodes
  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);
  ApplicationContainer sinkApps;
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
    sinkApps.Add (sinkHelper.Install (d.GetRight (i)));
  }
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop  (Seconds (stop_time));
  /////double start_t[nLeaf]=[0.001, 10, 15];
  /////double stop_t[nLeaf]=[20, 25, 35-0.1];

  double start_t[nLeaf];
  double stop_t[nLeaf];

  for (uint32_t i = 0; i < nLeaf; ++i)
  {
  start_t[i] = start_time + 0.001 + 0.01 * i;
  stop_t[i] = stop_time - 0.1;
  }
  //////
  for (uint32_t i = 0; i < d.LeftCount (); ++i)
  {
    // Create an on/off app sending packets to the left side
    AddressValue remoteAddress (InetSocketAddress (d.GetRightIpv4Address (i), port));
    Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (tcp_adu_size));
    BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
    ftp.SetAttribute ("Remote", remoteAddress);
    ftp.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
    ftp.SetAttribute ("MaxBytes", UintegerValue (data_mbytes * 1000000));

    ApplicationContainer clientApp = ftp.Install (d.GetLeft (i));
    clientApp.Start (Seconds (start_t[i])); // Start after sink
    clientApp.Stop (Seconds (stop_t[i])); // Stop before the sink
  }

  // Flow monitor
  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> allMon = flowHelper.InstallAll();

  // Count RX packets
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    rxPkts.push_back(0);
    Ptr<PacketSink> pktSink = DynamicCast<PacketSink>(sinkApps.Get(i));
    pktSink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&CountRxPkts, i));
  }

  Simulator::Stop (Seconds (stop_time));
  ThroughputMonitor(&flowHelper, allMon);
  /////
  //Simulator::Schedule(Seconds(10), &ChangeBW, bottleneckRouter); //change pointToPoint
  Simulator::Schedule(Seconds(0.04),&ThroughputMonitor,&flowHelper, allMon); 
  
  Simulator::Run ();

  if (flow_monitor)
    {
      flowHelper.SerializeToXmlFile (prefix_file_name + ".flowmonitor", true, true);
    }

  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    openGymInterface->NotifySimulationEnd();
  }

  PrintRxCount();
  Simulator::Destroy ();
  return 0;
}