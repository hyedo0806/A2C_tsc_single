import os
import pickle
import numpy as np
import win32com.client as com

BASE = [470, 490, 350, 230, 260]
MIN_GREEN = [390, 460, 320, 200, 140]
PHASE = ['00001100', '11000000', '00110000', '00100010', '00000011']
#PHASE = ['11113311', '33111111', '11331111', '11311131', '11111133']  ## RED : 1 // GREEN : 3 // AMBER : 4

class YuseongEnv():
    def __init__(self):
        self.Vissim = com.gencache.EnsureDispatch("Vissim.Vissim")

        cwd = os.getcwd()

        Filename = os.path.join(cwd, 'yuseong.inpx')
        flag_read_additionally = False
        self.Vissim.LoadNet(Filename, flag_read_additionally)

        self.Vissim.Simulation.SetAttValue('SimSpeed', 1)

        self.Max_Q = [0 for i in range(5)]
        self._num_states = 10
        self._num_actions = 30

        self.road()

    def veh_input(self):
        with open('./input data/yuseong_traffic.pickle', 'rb') as f:
            df = pickle.load(f)
        df = df.rename(dict(zip(list(df.index), list(range(1, len(df) + 1)))))

        # CCTV id 2 records traffic from West
        CCTV_DIR = {1: 'S', 2: 'W', 3: 'N', 4: 'E'}

        # direction to routing decision id mapping
        DIR_SVRD = {'N': 4, 'S': 3, 'E': 2, 'W': 1}

        # link id to vehicle input id
        DIR_VI = {'E': 1, 'N': 2, 'W': 3, 'S': 4}

        # link id to vehicle composition id
        DIR_VC = {'N': 1, 'E': 2, 'S': 3, 'W': 4}

        # For direction ordering
        dir_tuple = ('RIGHT_TRF_', 'GO_TRF_', 'LEFT_TRF_')

        for i, record in df.iterrows():

            trf_by_cctv = {num: sum([int(record[d + str(num)]) for d in dir_tuple])
                           for num in range(1, 5)}

            for cctv_num, traffic in trf_by_cctv.items():
                dir = CCTV_DIR[cctv_num]
                vi_id = DIR_VI[dir]

                # Set total traffic volume for each time step
                if i > 1:
                    self.Vissim.Net.VehicleInputs.ItemByKey(vi_id).SetAttValue(f'Cont({i})', False)
                self.Vissim.Net.VehicleInputs.ItemByKey(vi_id).SetAttValue(f'Volume({i})', int(traffic))

                # Set Vehicle Routing Decision
                svrd_id = DIR_SVRD[dir]
                cctv_trf = [int(record[d + str(cctv_num)]) for d in dir_tuple]
                total = sum(cctv_trf)
                # 1: right, 2: straight, 3: left
                for svr_id, trf in zip([1, 2, 3], cctv_trf):
                    self.Vissim.Net.VehicleRoutingDecisionsStatic.ItemByKey(svrd_id).VehRoutSta.ItemByKey(
                        svr_id).SetAttValue(
                        f'RelFlow({i})', trf / total)

        # Set Vehicle Composition for each direction
        with open('./input data/yuseong_weight.pickle', 'rb') as f:
            df2 = pickle.load(f)
        # print(df2)
        vc_type = ['CAR', 'BUS', 'BIKE']
        vc_id = [100, 300, 610]
        vc_speed = [50, 40, 40]

        df2 = df2.rename(dict(zip(list(df2.index), list(range(1, len(df2) + 1)))))

        for cctv_num in range(1, 5):
            dir = CCTV_DIR[cctv_num]
            vc_id = DIR_VC[dir]
            Rel_Flows = self.Vissim.Net.VehicleCompositions.ItemByKey(vc_id).VehCompRelFlows.GetAll()
            for i, type in enumerate(vc_type):
                # Rel_Flows[i].SetAttValue('VehType',        vc_id[i]) # Changing the vehicle type -> type subscriptable 오류
                Rel_Flows[i].SetAttValue('DesSpeedDistr', vc_speed[i])  # Changing the desired speed distribution
                Rel_Flows[i].SetAttValue('RelFlow', df2.loc[cctv_num][type])  # Changing the relative flow

    def signal(self):
        self.SC_number = 1  # SC = SignalController
        self.SH = []
        self.SG = []
        ## ====== Signal Controller & Signal Head & Signal Group Setting ======
        # Set a signal controller program:
        self.SignalController = self.Vissim.Net.SignalControllers.ItemByKey(self.SC_number)
        for i in range(16):
            self.SH.append(self.Vissim.Net.SignalHeads.ItemByKey(i + 1).AttValue('SigState'))
        for i in range(8):
            self.SG.append(self.SignalController.SGs.ItemByKey(i + 1))

    def road(self):
        Input = {'9-1': 1, '9-2': 1, '9-3': 2, '9-4': 2, '19-2': 3, '19-3': 3, '10025-2': 3, '10025-3': 3, '20-2': 3,
                 '20-3': 3,
                 '19-4': 4, '19-5': 4, '10024': 4, '10025-4': 4, '20-4': 4, '2-2': 5, '2-3': 5, '1-2': 5, '1-3': 5,
                 '2-4': 6, '2-5': 6, '1-4': 6, '1-5': 6, '13-2': 7, '13-3': 7, '13-4': 7, '10013-2': 7, '10013-3': 7,
                 '10013-4': 7,
                 '13-5': 8, '10030': 8}
        find = [9, 19, 20, 2, 1, 13, 10025, 10024, 10013, 10030]

        self.lane = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': []}

        for link in self.Vissim.Net.Links:
            if int(link.AttValue('No')) in find:
                for lane in link.Lanes.GetAll():
                    temp = str(link.AttValue('No')) + '-' + str(lane.AttValue('Index'))
                    if temp in Input: self.lane[str(Input[temp])].append(lane)
        Detector = self.Vissim.Net.Detectors.GetAll()
        print(Detector)
        self.detector = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': []}
        for item in Detector:
            self.detector[item.AttValue('Name')].append(item)

    def TH_calculate(self):

        alloc = {'1': [1], '2': [1], '3': [2, 3], '4': [2], '5': [0], '6': [0], '7': [4], '8': [3, 4]}

        for dirc in alloc.keys():
            for i in self.detector[dirc]:
                if i.AttValue('VehNo') == None : continue
                if i.AttValue('VehNo') > 0:
                    for num in range(len(alloc[dirc])):
                        self.TH[alloc[dirc][num]] += 1

    def _get_Qtime(self):
        queue = [0 for i in range(5)]
        alloc = {'1':[1],'2':[1],'3':[2,3],'4':[2],'5':[0],'6':[0],'7':[4],'8':[3,4]}

        for dirc in alloc.keys():
            for i in self.lane[dirc]:
                if not i.AttValue('MAX:VEHS\QTIME') == None:
                    for num in range(len(alloc[dirc])):
                        queue[alloc[dirc][num]] += i.AttValue('MAX:VEHS\QTIME')

        for i in range(5):
            self.Max_Q[i] = max(self.Max_Q[i], queue[i])

    def _get_state(self):
        state = []

        for i in self.Max_Q: state.append(i)
        for i in self.phase: state.append(i)

        state = np.reshape(state,(self._num_states))
        #print(type(state), state.shape)
        return state

    def stop(self):
        self.Vissim.Simulation.Stop()

    def reset(self,count):
        self.phase = BASE
        if count >0 :
            for simRun in self.Vissim.Net.SimulationRuns:
                self.Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

        self.signal()
        self.veh_input()  ## 차량 입력 // 추후에는 parameter를 사용하여 입력데이터 다양하게


        for  time_p in range(10):
            self.Vissim.Simulation.RunSingleStep()
            #if time_p % 10 == 9:
            #    self.TH_calculate()
            #    self._get_Qtime()
        for i in range(8):
            self.SG[i].SetAttValue("SigState", "RED")

        state, max_Q, _ = self.step([], True)
        return state, False

    def traffic_volume(self):

        sub = self.TH[0]+self.TH[1]
        main = self.TH[2]+self.TH[3]+self.TH[4]
        temp_s, temp_m, rate = [], [], []
        if sub * main == 0 : rate = [(0.5,0.5),(0.3,0.4,0.3)]
        else :
            for i in range(2): temp_s.append(self.TH[i]/sub)
            for i in range(3): temp_m.append(self.TH[i]/main)
            rate.append(temp_s)
            rate.append(temp_m)

        #rate =
        return rate   # [(a,b),(c,d,e)]

    def step(self, action, if_pre):
        self.TH = [0 for i in range(5)]
        Vol_traffic  = self.traffic_volume()
        if if_pre:
            self.phase = BASE

        else:
            cnt = 0
            self.phase = []
            for i in range(len(action)):
                for j in range(len(Vol_traffic[i])):
                    self.phase.append(round(Vol_traffic[i][j] * action[i]) * 10 + MIN_GREEN[cnt])
                    cnt += 1

        ###################################################################
        ## self.SG[i].SetAttValue("SigState", "...") 부분에서 원인모르게 오류발생하여 잘 실행되다가 중간에 끊김
        ## File "C:\Users\user\AppData\Local\Temp\gen_py\3.7\88F49AEC-253A-4F8B-A06D-9EA631AACA09x0x10x0\ISignalGroup.py", line 43, in SetAttValue, arg1)
        ##    pywintypes.com_error: (-2147352567, '예외가 발생했습니다.', (0, 'VISSIM.Vissim.1000', 'AttValue failed: Object 1 - 3: Signal group
        ##    3: Attribute Signal state is no subject to changes.', None, 0, -2147352567), None)

        ## RED : 1 // GREEN : 3 // AMBER : 4
        for p in range(5):
            for i in range(8):
                #if PHASE[p%5-1][i]^PHASE[p%5][i] == 1  : ## 변경
                #    self.SG[i].SetAttValue("SigState", PHASE[p%5][i])
                #else : ## 그대로

                if PHASE[p][i] == '0':
                    #print(self.SG[i].AttValue("SigState"))
                    self.SG[i].SetAttValue("SigState", 1)
                else:
                    self.SG[i].SetAttValue("SigState", 3)


            for time_p in range(self.phase[p] - 30):
                self.Vissim.Simulation.RunSingleStep()
                if time_p % 10 == 9:
                    self.TH_calculate()
                    self._get_Qtime()

            for i in range(8):
                if PHASE[p][i] == '1':
                    self.SG[i].SetAttValue("SigState", "AMBER")

            for time_p in range(self.phase[p] - 30, self.phase[p]):
                self.Vissim.Simulation.RunSingleStep()
                if time_p % 10 == 9:
                    self.TH_calculate()
                    self._get_Qtime()
        #print(self.phase)
        return self._get_state(), -sum(self.Max_Q), False
