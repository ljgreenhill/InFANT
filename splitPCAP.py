from scapy.all import *
import csv
import datetime
import random

arrContents = []
#SRC, DST, LENGTH, TIME, PROTOCOL
arrSrc = []
arrDst = []
arrLen = []
arrTim = []
arrPro = []
arrStat = []
counter = 0
status = 0

#number of examples, number of features, possible label names
#number of packets, 5, 0, 1
#'source', 'destination', 'size', 'time', 'protocol', 'roomState'

#returns two halves of array
def splitArr(arr):
    #rounds to whole number
    size = len(arr)//2
    return arr[:size], arr[size:]

#find unique values in array
def getUnique(arr):
    finArr = []
    for a in arr:
        if a not in finArr:
            finArr.append(a)
    return finArr

#replaces array value with number
def replaceItems(arr):
  uArr = getUnique(arr)
  for i in range(len(uArr)):
      for z in range(len(arr)):
          if uArr[i] == arr[z]:
              arr[z] = i
  return arr

#returns our from epoch time
def getHour(time):
    time = int(time)
    t = datetime.datetime.fromtimestamp(time).strftime('%c')
    _,_,_,dec,_ = t.split(' ')
    hour, _, _ = dec.split(':')
    return int(hour, base=10)

#add for random values 
'''def getStatus():
    global counter
    global status
    counter = counter + 1
    if counter < 250000:
        return status
    else:
        status = random.randint(0,1)
        counter = 0
        return status  '''

def getStatus(arrS):
    status = 0
    if arrS == 4:
        status = 1
    elif arrS == 16:
        status = 3
    elif arrS > 4 and arrS < 10:
        status = 2
    return status
    
def getIPData(arrS, arrD):
    arrIP = []
    tempArr = []
    for i in range(len(arrS)):
        arrIP += (arrS[i] + arrD[i])
    arrIP = replaceItems(arrIP)
    return arrIP

#combines arrays of specific info into one 2D array
def buildArrContents(arrS, arrD, arrL, arrT, arrP):
    arrIP = arrS + arrD
    arrIP = replaceItems(arrIP)
    arrS, arrD = splitArr(arrIP)
    arrP = replaceItems(arrP)
    size = len(arrS)//4
    for i in range(len(arrS)):
        print([arrT[i],arrS[i],arrD[i],arrL[i],arrP[i], getStatus(arrS[i])])
        #if ((getHour(arrT[i]) <= 4 and getHour(arrT[i]) >= 0) or getHour(arrT[i]) > 22):
        if i<size:
            writeCSV([arrT[i],arrS[i],arrD[i],arrL[i],arrP[i], getStatus()],"test","default")
        else:
            writeCSV([arrT[i],arrS[i],arrD[i],arrL[i],arrP[i], getStatus(arrS[i])],"train","default")

#writes packet data to csv file (1/4 test, 3/4 training)
def writeCSV(arr, status, kind):
    switcher = {
        "ip": "ip.csv",
        "length": "length.csv",
        "protocol": "protocol.csv",
        "default": ".csv"
    }
    if status is "test":
        with open(("input_test3" + switcher.get(kind)), mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(arr)
    if status is "train":
        with open(("merged" + switcher.get(kind)), mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(arr)

#returns packet protocol
def findProtocol(pkt):
    summary = pkt.command()
    if "Cisco" in summary:
        return "LLDP"
    if "ARP" in summary:
        return "ARP"
    if "proto=0" in summary:
        return "HOPOPT"
    if "proto=1" in summary:
        return "ICMP"
    if "proto=6" in summary:
        return "TCP"
    if "proto=17" in summary:
        return "UDP"

#adds important information about each packet to designated array       
def filterPCAP(pkt):
  print("Filtering PCAP")
  #summary = pkt.command()
  src = pkt.src
  dst = pkt.dst
  length = len(pkt.payload)
  time = pkt.time
  protocol = findProtocol(pkt)
      
  arrSrc.append(src)
  arrDst.append(dst)  
  arrLen.append(length)
  arrTim.append(time)
  arrPro.append(protocol)

   
sniff(offline="merge291205-1305.pcap",prn=filterPCAP,store=0)

buildArrContents(arrSrc, arrDst, arrLen, arrTim, arrPro)
