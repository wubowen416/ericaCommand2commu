# -*- coding: utf-8 -*-

from matplotlib.pyplot import table
import numpy as np
import math
import os
import copy

import pandas as pd


# from tottoclass import tottoClass
from ericaclass import ERICAClass
# from skeltonclass import SkeltonClass
from tslearn.preprocessing import TimeSeriesResampler



class CommandManager():
    def __init__(self,robot_name):

        if robot_name == 'totto':
            # self.robot_class = tottoClass()
            self.robot_class = None
        elif robot_name == 'ERICA':
            self.robot_class = ERICAClass()
        elif robot_name == 'Skelton':
            # self.robot_class = SkeltonClass()
            self.robot_class = None
        else:
            raise('robot name is wrong !!')

        self.read_configfile()

    def read_configfile(self):
        """
        ジェスチャーのpreparationやretractionのパラメータやホームポジションの指令値情報を読み取る関数

        """
        f = open('./config/gestureConfig.txt','r',encoding = 'utf-8')
        gf = f.read().split('\n')
        f.close()
        for ind in range(len(gf)):
            if 'fps' in gf[ind]:
                self.motion_fps = int(gf[ind].split('\t')[1])
            elif 'hold' in gf[ind]:
                self.hold_time = float(gf[ind].split('\t')[1])
            elif 'preparation' in gf[ind]:
                self.pre_frame = int(float(gf[ind].split('\t')[1])*self.motion_fps)
            elif 'retraction' in gf[ind]:
                self.ret_frame = int(float(gf[ind].split('\t')[1])*self.motion_fps)
            elif 'pre parameter' in gf[ind]:
                self.pre_p = float(gf[ind].split('\t')[1])
            elif 'ret parameter' in gf[ind]:
                self.ret_p = float(gf[ind].split('\t')[1])

    def read_text_data(self,text_path):
        '''
        motion editorで作成したtxtファイルを読み込む関数
        input
        text_path: motionのtxtファイルのpath情報

        output
        motiontext(new_set):各軸の指令値の時系列配列 [step][axis number]
        text_inf:motion inverval, motion steps, motion axes の配列

        '''
        f = open(text_path,'r')
        data = f.read().split('\n')
        f.close()


        try:
            text_inf = np.array(data[3].split('\t')).astype(int)
        except ValueError:
            text_inf = np.array(data[3][:-2].split('\t')).astype(int)


        motion_data = data[4:4+text_inf[1]]


        count = 0
        new_set = np.array([])
        for row in motion_data:

            sub_set = np.array(row.split('\t')[:-1]).astype(float).astype(int)
            try:
                new_set = np.vstack((new_set,sub_set))
            except ValueError:
                new_set = sub_set

        return new_set,text_inf

    def adjust_frame_length(self,motion_data,frame_num = 50):

        for side in ['left','right']:
            for dkey in ['command','wristpoint','hippoint']:
                try:
                    command_set = motion_data[dkey][side].T
                    if command_set.ndim == 1:
                        command_set = np.vstack((command_set,command_set)).T
                    if len(command_set[0]) == 1:
                        command_set = np.hstack((command_set,command_set))
                    new_set = np.array([])
                    for row in command_set:

                        try:
                            new_sub = TimeSeriesResampler(frame_num).fit_transform(row)[0].T[0]
                        except IndexError:
                            input()
                        except ValueError:
                            input()


                        try :
                            new_set = np.vstack((new_set,new_sub))
                        except ValueError:
                            new_set = new_sub
                    motion_data[dkey][side] = new_set.T
                except KeyError:
                    pass

        for side in ['left','right']:
            try:
                motion_data[side]['phase'] = self.make_phase_set(motion_data[side]['wristpoint'],motion_data[side]['hippoint'],70)

            except IndexError:
                input()

        return motion_data

    def make_phase_set(self,wristpoint_set,hippoint_set,threshold):

        phase_set = []
        for wp,hp in zip(wristpoint_set,hippoint_set):
            if hp[2]+threshold < wp[2]:
                phase_set.append('stroke')
            else:
                phase_set.append('home')

        return phase_set

    def check_arm_inf(self,arm_inf):
        """
        arm_infのフレーム数が１の時に，フレーム数をふ安関数．add_motionの時にバグが生じる
        そもそもcut_gestureか，make_commandでこのようなmotionの削除処理をするべき．

        """

        command_set_left = arm_inf['left']['command']

        if command_set_left.ndim == 1:
            print('1 frame motion')

            for side in ['left','right']:
                for dkey in ['command', 'hippoint', 'wristpoint']:

                    tmp = arm_inf[side][dkey]
                    arm_inf[side][dkey] = np.vstack((tmp,tmp))


                arm_inf[side]['phase'] = np.hstack((arm_inf[side]['phase'],arm_inf[side]['phase']))

            print('check')

        return arm_inf



    def add_gesture2motiondata(self,motion_data,arm_inf,ges_start,ges_len = None,holdtime = None):
        """
        ジェスチャーの開始から最後まで、motion_dataが足りない時の対処を書いておく必要がある

        入力情報
        motion_data:発話に伴うモーション情報（ロボットに送信する情報）
        arm_inf:ジェスチャー情報　['left','right']['command','angle','wristpoint','phase']
        ges_len:ジェスチャーの継続フレーム数（Noneの場合はフルジェスチャー）
        holdtime:ジェスチャーが終わった後に加えるホールドの時間

        出力情報:引数のmotion_dataにジェスチャー情報を格納する

        """

        self.check_arm_inf(arm_inf)

        if ges_len is None:
            ges_len = len(arm_inf['left']['command'])

        for dkey in ['command','phase']:
            for side in ['left','right']:

                for frame in range(ges_len):
                    if len(motion_data[side][dkey]) < int(ges_start) + frame:
                        print('over frame')
                        print('motion_data length = {}'.format(len(motion_data[side][dkey])))
                        print('ges_start + gesture = {}'.format(int(ges_start) + len(arm_inf[side][dkey][frame])))
                        break
                    try:
                        motion_data[side][dkey][int(ges_start)+frame] = arm_inf[side][dkey][frame]
                    except IndexError:
                        print('len(motion_data[side][dkey]) = {}'.format(len(motion_data[side][dkey])))
                        print('int(ges_start) + frame = {}'.format(int(ges_start) + frame))

                if not holdtime is None:

                    if dkey == 'command':
                        hold_inf = arm_inf[side][dkey][frame]

                    if dkey == 'phase':
                        if arm_inf[side][dkey][frame] == 'stroke':
                            hold_inf = 'hold'
                        else:
                            hold_inf = 'home'

                    for h in range(holdtime):

                        try:
                            motion_data[side][dkey][int(ges_start)+frame+1+h] = hold_inf
                        except IndexError:
                            print('out of range')

        return motion_data

    def get_hand_default(self,handside):
        """
        入力情報
        handside:left or right

        出力情報:引数の側の手の指令値情報
        """
        armstart = self.robot_class.robot_inf[handside]['armstart']-1
        arm_len = self.robot_class.robot_inf[handside]['armlength'] - self.robot_class.robot_inf[handside]['fingerlength']
        return self.robot_class.default_command[armstart:armstart+arm_len]

    def get_sigmoid(self,frame_num,para = 1):
        """
        入力情報
        frame_num：出力として得られるsigmoid関数のステップ数
        para：sigomid関数のパラメータ　曲線の緩やかさを決定する

        出力情報：引数のステップ数に設定したsigmoid関数の値
        """
        def cal_sigmoid(x,a = 1):
            return 1.0/(1.0 + np.exp(-a*x))

        x = np.arange(-5,5,0.01)
        y = cal_sigmoid(x, para)

        #===============================================================================================================
        # plt.xlabel("x_")
        # plt.ylabel("f(x)")
        # plt.plot(x,y)
        # plt.show()
        #===============================================================================================================

        step = math.floor(len(y)/frame_num)
        y = y[::step][:frame_num]
        y = y/y[-1]


        #===============================================================================================================
        # x = np.arange(0,len(y),1)
        # plt.xlabel("x_")
        # plt.ylabel("f(x)")
        # plt.plot(x,y)
        # plt.show()
        #===============================================================================================================

        return y[:,np.newaxis]

    def add_pre_ret(self,motion_data,handside):
        """
        preparationとretractionの追加を行うための関数
        finish_modをする場合はこの関数内でするかしないかを選択できるようにしておく

        入力情報
        motion_data:gestureの情報　[left,right][command,phase]
        hand_side:'left' or 'right'
        last_mod:ジェスチャーの最後のフレームが'home'でない場合に'home'で終わるように設定するかどうか　True or False

        出力情報:preparationとretractionを含んだmotiondata

        """
        ftime = -1
        thre_count = 0
        check_first = False

        value_diff = 0


        hand_default = self.get_hand_default(handside)
        hand_com = hand_default

        y_pre = self.get_sigmoid(self.pre_frame,self.pre_p)
        y_ret = self.get_sigmoid(self.ret_frame,self.ret_p)

        for ind in range(len(motion_data[handside]['command'])):
            print('step:{}'.format(ind))
            if thre_count == 0 and motion_data[handside]['phase'][ind] != 'home':
                "stroke -> stroke"
                print("frame{}:stroke -> stroke ".format(ind))
                ftime += 1
                thre_count = 0
                check_first = True

            elif thre_count == 0 and motion_data[handside]['phase'][ind] == 'home':
                if ind != 0:
                    "stroke -> retaction"
                    print("frame{}:stroke -> retaction".format(ind))
                    #checkpoint
                    hand_com = np.mean(motion_data[handside]['command'][ftime-2:ftime+1],axis = 0)#[0:self.ges_file_inf['command']])
                    value_diff = hand_default - hand_com
                    y2_ret = y_ret*value_diff
                    motion_data[handside]['command'][ind] = hand_com+y2_ret[thre_count]
                    motion_data[handside]['phase'][ind] = 'retraction'
                    thre_count += 1
                    ftime += 1
                else :
                    "step = 0(first step is under the threshold) -> home"
                    print("frame{}:step = 0(first step is under the threshold) -> home".format(ind))
                    motion_data[handside]['command'][ind] = hand_default
                    thre_count += 1
                    ftime += 1

            elif thre_count != 0 and motion_data[handside]['phase'][ind] != 'home':
                """
                home or retraction -> stroke
                previous motion position is under the threshold value,
                now the position is above the threshold value.
                """
                hand_com = motion_data[handside]['command'][ind]

                if thre_count > int(self.ret_frame + self.pre_frame):
                    """
                    previous phase of gesture is home -> stroke
                    it is enough time to preparate from home position
                    """
                    print("frame{}:previous phase of gesture is home -> stroke".format(ind))
                    stime = ftime  - self.pre_frame
                    value_diff =  hand_com - hand_default
                    y2_pre = y_pre*value_diff

                    motion_data[handside]['command'][stime+1:ftime+1] = hand_default + y2_pre
                    motion_data[handside]['phase'][stime+1:ftime+1] = 'preparation'


                elif thre_count <= int(self.ret_frame + self.pre_frame) and self.pre_frame <= thre_count:
                    """
                    the phase when this preparation is started is retraction or home and it has enough time to preparation.
                    because of retraction time(maybe and home ) > preparation time
                    """
                    print("frame{}:the phase when this preparation is started is retraction or home and it has enough time to preparation.".format(ind))
                    stime = ftime  - self.pre_frame
                    hand_com2 = motion_data[handside]['command'][stime+1]
                    value_diff =  hand_com - hand_com2
                    y2_pre = y_pre*value_diff
                    motion_data[handside]['command'][stime+1:ftime+1] = hand_com2 + y2_pre
                    motion_data[handside]['phase'][stime+1:ftime+1] = 'preparation'

                elif thre_count <= self.pre_frame and check_first is False:
                    """
                    the phase when this preparation is started is home and it doesn't have enough time to preparate stroke phase.
                    And it is first time of stroke
                    """
                    print("frame{}:the phase when this preparation is started is home and it doesn't have enough time to preparate stroke phase.".format(ind))
                    x_pre_short = np.arange(int(-thre_count/2),int(math.ceil(thre_count/2)),1)
                    value_diff = hand_com - hand_default
                    y_pre_short = self.get_sigmoid(len(x_pre_short),self.pre_p)
                    y2_pre = y_pre_short*value_diff
                    stime = ftime - len(x_pre_short)
                    motion_data[handside]['command'][stime+1:ftime+1] = hand_default + y2_pre
                    motion_data[handside]['phase'][stime+1:ftime+1] = 'preparation'

                else :
                    """
                    the phase when this preparation is started is retraction and it doesn't have enough time to preparate stroke phase.
                    And it is not first time of stroke
                    """
                    print("frame{}:the phase when this preparation is started is retraction and it doesn't have enough time to preparate stroke phase.".format(ind))
                    #check point この配列の長さだけ知れれば良いarange使う必要なし
                    x_pre_short = np.arange(int(-thre_count/2),int(math.ceil(thre_count/2)),1)
                    hand_com2 = motion_data[handside]['command'][ftime-thre_count+1]
                    value_diff = hand_com - hand_com2
                    y_pre_short = self.get_sigmoid(len(x_pre_short),self.pre_p)
                    y2_pre = y_pre_short*value_diff
                    #check point
                    stime = ftime - len(x_pre_short)
                    print('checking flow')
                    motion_data[handside]['command'][stime+1:ftime+1] = hand_com + y2_pre
                    motion_data[handside]['phase'][stime+1:ftime+1] = 'preparation'

                thre_count = 0
                ftime += 1
                check_first = True
            else:
                """
                under the threshold value -> under the threshold value
                """
                if thre_count >= self.ret_frame-2 or check_first is False:
                    "the phase of gesture is home"
                    print("frame{}:the phase of gesture is home".format(ind))
                    motion_data[handside]['command'][ind] = hand_default

                else :
                    "the phase of gesture is retraction"
                    print("frame{}:the phase of gesture is retraction".format(ind))

                    motion_data[handside]['command'][ind] = hand_com + y2_ret[thre_count]
                    motion_data[handside]['phase'][ind] = 'retraction'
                thre_count += 1
                ftime += 1

        return motion_data

    def save_motiontext(self,motion_data,file_name):
        """
        入力情報

        motion_data:
        file_name:
        interval:(msec)

        出力情報:なし
        引数のファイル名でmotiontextを保存する
        """

        try:
            os.mkdir('./create-data/motiontext')
        except FileExistsError:
            pass

        try:
            os.mkdir('./create-data/motiontext/{}'.format(self.robot_class.robot_name))
        except FileExistsError:
            pass

        savefile_path = './create-data/motiontext/{}/{}'.format(self.robot_class.robot_name,file_name)

        leftarm_length = self.robot_class.robot_inf['left']['armlength'] - self.robot_class.robot_inf['left']['fingerlength']
        rightarm_length = self.robot_class.robot_inf['right']['armlength'] - self.robot_class.robot_inf['right']['fingerlength']

        leftarm_start = self.robot_class.robot_inf['left']['armstart']-1
        leftarm_end = leftarm_start + leftarm_length
        rightarm_start = self.robot_class.robot_inf['right']['armstart']-1
        rightarm_end = rightarm_start + rightarm_length

        motiontext_data = np.array([])
        step_num = len(motion_data['left']['command'])

        for frame in range(step_num):
            command_default = copy.copy(self.robot_class.default_command)
            command_default[leftarm_start:leftarm_end] = motion_data['left']['command'][frame]
            command_default[rightarm_start:rightarm_end] = motion_data['right']['command'][frame]
            try:
                motiontext_data = np.vstack((motiontext_data,command_default))
            except ValueError:
                motiontext_data = command_default


        f = open(savefile_path + ".txt",'w')
        f.write("comment:2nd row is default values: 3rd is axis number: 4th are motion inverval, motion steps, motion axes\n")

        msg2 = ''
        msg3 = ''
        for ind,com in enumerate(self.robot_class.default_command):
            msg2 += "{}\t".format(com)
            msg3 += "{}\t".format(ind+1)
        msg2 = msg2[:-1] + '\n'
        msg3 = msg3[:-1] + '\n'
        f.write(msg2+msg3)

        interval = int(1/self.motion_fps*1000)
        f.write('{}\t{}\t{}\n'.format(interval,step_num,len(self.robot_class.default_command)))

        for ind in range(step_num):
            for i in range(len(motiontext_data[ind])):
                f.write(str(int(motiontext_data[ind][i])) + "\t")
            f.write("\n")

        f.write("* 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111 111\n")
        f.close()
        print('create motion_text !!')

    
    def get_arm_index(self, side):

        if side == 'left':
            length = self.robot_class.robot_inf['left']['armlength'] - self.robot_class.robot_inf['left']['fingerlength']
            start = self.robot_class.robot_inf['left']['armstart']-1
            end = start + length

        elif side == 'right':
            length = self.robot_class.robot_inf['right']['armlength'] - self.robot_class.robot_inf['right']['fingerlength']
            start = self.robot_class.robot_inf['right']['armstart']-1
            end = start + length

        return start, end


    def command2angle(self, text_path):
        '''
        Convert motion editor text file to angles of arm position.

        Arguments:
            text_path -- Path of motion editor .txt file.
        
        Returns:
            angle -- Dict for left and right arm joint angle. The order on each arm is th1 -> th8.
        '''

        angle = {}

        arr, _ = self.read_text_data(text_path)

        side = 'left'
        # Calculate angle
        arm_start, arm_end = self.get_arm_index(side)
        arm_arr = arr[:, arm_start:arm_end]
        ang_arr = np.empty(shape=(0, 8)) # 8: T1 -> T8
        for arr_frame in arm_arr:
            ang_arr = np.append(ang_arr, np.expand_dims(self.robot_class.command2angle_set(arr_frame, side), 0), axis=0)
        
        angle['left'] = ang_arr.copy()

        side = 'right'
        # Calculate angle
        arm_start, arm_end = self.get_arm_index(side)
        arm_arr = arr[:, arm_start:arm_end]
        ang_arr = np.empty(shape=(0, 8)) # 8: T1 -> T8
        for arr_frame in arm_arr:
            ang_arr = np.append(ang_arr, np.expand_dims(self.robot_class.command2angle_set(arr_frame, side), 0), axis=0)
        
        angle['right'] = ang_arr.copy()

        return angle