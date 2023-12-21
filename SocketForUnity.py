from collections.abc import Callable, Iterable, Mapping
import os
from typing import Any
import torch
import numpy as np
import pickle
import socket
from threading import Thread
import json
import time

###################################   Method

def initOption(options) -> dict:
    opt = options

    if options["socket_AddressFamily"]==0:
        opt["socket_AddressFamily"]=socket.AddressFamily.AF_INET
    elif options["socket_AddressFamily"]==1:
        opt["socket_AddressFamily"]=socket.AddressFamily.AF_INET6
    else:
        print("No matach addressfamily")
    
    if options["socket_Type"]==0:
        opt["socket_Type"]=socket.SOCK_STREAM
    elif options["socket_Type"]==1:
        opt["socket_Type"]=socket.SOCK_DGRAM
    elif options["socket_Type"]==2:
        opt["socket_Type"]=socket.SOCK_RAW
    else:
        print("No matach socket type")

    return opt

def save_npz2json(filepath,outpath=".\\results\\json")->None:
    data={}

    dirname,filename = os.path.split(filepath)
    filenames = filename.split(".")
    save_file_path=os.path.join(outpath,dirname)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    save_file_path=os.path.join(outpath,dirname,filenames[0]+".json")
    
    bdata = np.load(filepath,allow_pickle=True)
    keys = list(bdata.keys())
    for key in keys:
        data[key]= bdata[key].tolist()
        print("{0}-{1}-{2}".format( key, type(bdata[key]), bdata[key].shape ))
    
    with open(save_file_path,"w") as f:
        json.dump(data,f)

    print("{0}{1}".format("save to : ",save_file_path))
    return

def save_pkl2json(filepath,outpath=".\\results\\json")->None:
    data={}

    dirname,filename = os.path.split(filepath)
    filenames = filename.split(".")
    save_file_path=os.path.join(outpath,dirname)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    save_file_path=os.path.join(outpath,dirname,filenames[0]+".json")
    
    with open(filepath,'rb') as f:
        bdata = pickle.load(f)
    keys = list(bdata.keys())
    for key in keys:
        #print("{0}----------{1}".format(key,type(bdata[key])))
        if type(bdata[key]) is torch.Tensor :
            data[key]=bdata[key].numpy().tolist()
            #print(type(data[key]))
        elif type(bdata[key]) is np.ndarray :
            data[key]=bdata[key].tolist()
            #print(type(data[key]))
        elif type(bdata[key]) is dict :
            for bodyparm in bdata[key]:
                bdata[key][bodyparm]=bdata[key][bodyparm].numpy().tolist()
            data[key]=bdata[key]
        else:
            data[key]=bdata[key]
            #print(type(data[key]))

    with open(save_file_path,"w") as f:
        json.dump(data,f)

    #print("{0}{1}".format("save to : ",save_file_path))
    return

####################################  Class
class CMD(Thread):
    def __init__(self,group) -> None:
        super().__init__()
        self.group=group

    def start(self) -> None:

        return super().start()
    
    def run(self) -> None:
        while True:
            cmd_list = input("CMD control:").split("-")
            if len(cmd_list)>=2:
                if cmd_list[1] == "quit":
                    break

                elif cmd_list[1] =="server":
                    if cmd_list[2]=="start":
                        self.group[0].start()
                    elif cmd_list[2]=="send":
                        self.group[0].send(cmd_list[3])

                elif cmd_list[1] =="client":
                    if cmd_list[2] =="start":
                        self.group[1].start()
                    elif cmd_list[2] =="send":
                        self.group[1].send(cmd_list[3])
        return super().run()


class ServerSK(Thread):
    def __init__(self):
        super(ServerSK, self).__init__()
        self.options=None
        self.message=None
        self.link_sk = None
        optionPath="options/ConnectUnity.json"
        with open(optionPath,'r') as f:
            self.options=json.load(f)
            self.options=initOption(self.options)

        self.server_sk=socket.socket(self.options["socket_AddressFamily"],
                                     self.options["socket_Type"]) 


    def start(self) -> None:
        #写入自己的代码
        self.server_sk.bind(tuple(self.options["ip_port"]))
        #监听一个端口,这里的数字3是一个常量，表示阻塞3个连接，也就是最大等待数为3
        self.server_sk.listen(3)
        
        return super().start()
    
    def run(self) -> None:
        Thread(target=self.recvqueue,name="Server_Recv_Queue").start()
        Thread(target=self.sendqueue,name="Server_Send_Queue").start()
        print("server running")
    
    def recvqueue(self):
        self.link_sk,address=self.server_sk.accept()
        print(self.link_sk)
        while True:
            data = None
            if getattr(self.link_sk,'_closed')==False:
                data=self.link_sk.recv(1024)        #客户端发送的数据存储在recv里，1024指最大接受数据的量
                print("recive form Client:{0}".format(data.decode('utf-8')))

    def sendqueue(self):
        while True:
            if self.message and (getattr(self.link_sk,'_closed')==False):
                self.link_sk.send(self.message.encode('utf-8'))
                self.message = None


    def shutdown(self):
        self.link_sk.close()

    def send(self,message):
        self.message = message


class ClientSK(Thread):
    def __init__(self):
        super().__init__()
        self.options=None
        self.message=None
        optionPath="options/ConnectUnity.json"
        with open(optionPath,'r') as f:
            self.options=json.load(f)
            self.options=initOption(self.options)

        self.client_sk=socket.socket(self.options["socket_AddressFamily"],
                                     self.options["socket_Type"])

    def start(self) -> None:
        self.connect()
        return super().start()
    
    def run(self) -> None:
        Thread(target=self.recvqueue,name="Client_Recv_Queue").start()
        Thread(target=self.sendqueue,name='Client_Send_Queue').start()
        print("client running")
    
    def recvqueue(self):
        while True:
            data = None
            if getattr(self.client_sk,'_closed')==False:
                data=self.client_sk.recv(1024)
                print("recive form Server:{0}".format(data.decode('utf-8')))
    def sendqueue(self):
        while True:
            if self.message and (getattr(self.client_sk,'_closed')==False):
                self.client_sk.send(self.message.encode('utf-8'))
                self.message = None

    def connect(self):
        self.client_sk.connect(tuple(self.options["ip_port"]))

    def shutdown(self):
        self.client_sk.close()

    def send(self,message):
        self.message = message
   
####################################   Main    
def main():
    # ############################### ---服务器配置流程
    # Server = ServerSK()
    # Client = ClientSK()
    # cmdm = CMD([Server,Client])
    # # 启动CMD线程
    # cmdm.start()

    # #启动ServerSK和ClientSK线程
    # cmdm.group[0].start()
    # cmdm.group[1].start()

    # # 发送消息
    # cmdm.group[0].send("1111111111")
    # time.sleep(1)
    # cmdm.group[1].send("2222222222")

    ################################-----pkl数据文件读写测试
    # testfilepath = os.path.join("data_fps60","CMU","test","1.pkl")
    # save_pkl2json(testfilepath)

    ################################------npz数据文件读写测试
    testfilepath = os.path.join("data_split","CMU","01","01_01_poses.npz")
    save_npz2json(testfilepath)
    return

if __name__ == '__main__':
    main() 





