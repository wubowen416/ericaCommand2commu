# -*- coding: utf-8 -*-


import numpy as np
import copy

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class ERICAClass():
    def __init__(self):
        self.robot_name = 'ERICA'

        self.l0 = 28
        self.l1 = 25
        self.l15 = 19
        self.l2 = 21.5
        self.l3 = 74.5
        self.l4 = 109.5
        self.l5 = 129.8
        self.l6 = 87.2
        self.l7 = 139.2
        self.l8 = 175

        #phaseの推定に使用
        self.threshold = 150

        self.joint_point =np.array([[   33,    0,  508],
                                    [   33,   28,  508],
                                    [    8,   28,  508],
                                    [    8,   47,  508],
                                    [    8, 68.5,  508],
                                    [    8,  143,  503],
                                    [    8,154.5,393.5],
                                    [ 18.5,162.1,263.7],
                                    [ 14.6,175.2,176.5],
                                    [ 27.4,222.7, 37.3]])

        self.forearm_direction = {'vertical':250, 'side':108, 'lick':180}
        self.read_robotConfig()

        shoulder_len = self.l0+self.l15+self.l2 + self.l3
        forearm = self.l4 + self.l5
        upperarm = self.l6 + self.l7
        self.arm_length = forearm + upperarm
        finger_len = 175
        self.length_list = np.array([shoulder_len,forearm,upperarm,finger_len])

    def read_robotConfig(self):
        # print('robot mode {}'.format(self.robot_name))
        f = open('./config/{}Config.txt'.format(self.robot_name))
        gf = f.read().split('\n')
        f.close()
        self.robot_inf = {}
        self.limit_angle = {}
        for ind in range(len(gf)):
            if 'default command' in gf[ind]:
                self.default_command = np.array(gf[ind].split('\t')[1].split(',')).astype(np.int64)

            elif 'robot_inf' in gf[ind]:
                bodyparts = gf[ind].split('\t')[1]
                gf_sub = gf[ind].split('\t')[2:]
                self.robot_inf[bodyparts] = {}
                for v in gf_sub:
                    self.robot_inf[bodyparts][v.split(' ')[0]] = int(v.split(' ')[1])
            elif 'elbow_joint' in gf[ind]:
                self.elbow_command_num = {}
                gf_sub = gf[ind].split('\t')[1:]
                for v in gf_sub:
                    self.elbow_command_num[v.split(' ')[0]] = int(v.split(' ')[1])
            elif 'uparm_joint' in gf[ind]:
                self.uparm_command_num = {}
                gf_sub = gf[ind].split('\t')[1:]
                for v in gf_sub:
                    self.uparm_command_num[v.split(' ')[0]] = int(v.split(' ')[1])
            elif 'forearm_joint' in gf[ind]:
                self.forearm_command_num = {}
                gf_sub = gf[ind].split('\t')[1:]
                for v in gf_sub:
                    self.forearm_command_num[v.split(' ')[0]] = int(v.split(' ')[1])

            elif 'limit angle left' in gf[ind]:
                count = int(gf[ind].split('\t')[1])
                self.limit_angle['left'] = np.array([])

                for i in range(count):
                    tmp  = np.array(gf[ind+i+1].split('\t')).astype(float)
                    try:
                        self.limit_angle['left'] = np.vstack((self.limit_angle['left'],tmp))
                    except ValueError:
                        self.limit_angle['left'] = tmp

            elif 'limit angle right' in gf[ind]:
                count = int(gf[ind].split('\t')[1])
                self.limit_angle['right'] = np.array([])

                for i in range(count):
                    tmp  = np.array(gf[ind+i+1].split('\t')).astype(float)
                    try:
                        self.limit_angle['right'] = np.vstack((self.limit_angle['right'],tmp))
                    except ValueError:
                        self.limit_angle['right'] = tmp

    def DH_calculation(self,th,details = False,jointnum = None,bodypoint = False,bodyarray = False,finger = False):
        """
        DH法でジョイントの計算を行う関数。
        入力情報
        th:各関節の角度情報
        details:すべての関節を見るかどうか
        joint_num:ジョイントの行列を返す際にどのジョイントの情報を返すかの番号を入力
        bodypoint:関節の3次元点の位置情報を返す設定
        bodyarray:対応するジョイントの行列を返す設定
        finger:指先の情報を含めるかどうか
        ...

        """
        angle3 = 3/180*np.pi
        angle6 = 10/180*np.pi

        A = np.array([[1,0,0,self.joint_point[0][0]],
                       [0,1,0,self.joint_point[0][1]],
                       [0,0,1,self.joint_point[0][2]],
                       [0,0,0,1]])
        Aa = np.array([[1,0,0,0],
                       [0,1,0,self.l0],
                       [0,0,1,0],[0,0,0,1]])
        A1 = np.array([[-np.sin(-th[0]),0,np.cos(-th[0]),-self.l1*np.cos(-th[0])],
                       [np.cos(-th[0]),0,np.sin(-th[0]),-self.l1*np.sin(-th[0])],
                       [0,1,0,0],
                       [0,0,0,1]])
        Ab = np.array([[1,0,0,self.l15],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1]])
        A2 = np.array([[-np.sin(th[1]),0,np.cos(th[1]),self.l2*np.cos(th[1])],
                       [np.cos(th[1]),0,np.sin(th[1]),self.l2*np.sin(th[1])],
                       [0,1,0,0],
                       [0,0,0,1]])
        A3 = np.array([[-np.cos(-th[2]),0,-np.sin(-th[2]),0],
                       [-np.sin(-th[2]),0,np.cos(-th[2]),0],
                       [0,1,0,self.l3],
                       [0,0,0,1]])
        A4 = np.array([[-np.sin(th[3]+angle3),0,np.cos(th[3]+angle3),self.l4*np.cos(th[3]+angle3)],
                       [np.cos(th[3]+angle3),0,np.sin(th[3]+angle3),self.l4*np.sin(th[3]+angle3)],
                       [0,1,0,0],
                       [0,0,0,1]])
        A5 = np.array([[-np.sin(th[4]),0,np.cos(th[4]),0],
                       [np.cos(th[4]),0,np.sin(th[4]),0],
                       [0,1,0,self.l5],
                       [0,0,0,1]])
        A6 = np.array([[-np.cos(th[5]),0,np.sin(th[5]),self.l6*np.sin(th[5])],
                       [np.sin(th[5]),0,np.cos(th[5]),self.l6*np.cos(th[5])],
                       [0,1,0,0],
                       [0,0,0,1]])
        A7 = np.array([[-np.cos(th[6]),0,-np.sin(th[6]),0],
                       [-np.sin(th[6]),0,np.cos(th[6]),0],
                       [0,1,0,self.l7],
                       [0,0,0,1]])
        A8 = np.array([[np.cos(th[7]),np.sin(th[7]),0,self.l8*np.sin(th[7])],
                       [-np.sin(th[7]),np.cos(th[7]),0,self.l8*np.cos(th[7])],
                       [0,0,1,0],
                       [0,0,0,1]])

        Ac = np.array([[1,0,0,0],
                       [0,np.cos(angle6),-np.sin(angle6),0],
                       [0,np.sin(angle6),np.cos(angle6),0],
                       [0,0,0,1]])

        T1 = np.dot(A,Aa)
        T2 = np.dot(T1,A1)
        T3 = np.dot(T2,Ab)
        T4 = np.dot(T3,A2)
        T5 = np.dot(T4,A3)
        T6 = np.dot(T5,A4)
        T7 = np.dot(T6,A5)
        T7c = np.dot(T7,Ac)
        T8 = np.dot(T7c,A6)#T7->T7c
        T9 = np.dot(T8,A7)
        T10 = np.dot(T9,A8)

        if bodypoint:
            if details:
                arr = np.array([A[0:3,3],
                                T1[0:3,3],
                                T2[0:3,3],
                                T3[0:3,3],
                                T4[0:3,3],
                                T5[0:3,3],
                                T6[0:3,3],
                                T7[0:3,3],
                                T8[0:3,3],
                                T9[0:3,3]])
            else:
                arr = np.array([A[0:3,3],
                                T1[0:3,3],
                                T3[0:3,3],
                                T4[0:3,3],
                                T5[0:3,3],
                                T6[0:3,3],
                                T7[0:3,3],
                                T8[0:3,3],
                                T9[0:3,3]])
            if finger:
                arr = np.vstack((arr,T10[0:3,3]))

            return arr

        if bodyarray:
            T_all = np.array([A,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10])
            return T_all[jointnum]

    def scale2robot_coord(self,point):
        x = point[0]*self.arm_length+self.joint_point[0][0]
        y = point[1]*self.arm_length+self.joint_point[0][1]
        z = point[2]*self.arm_length+self.joint_point[0][2]
        return np.array([x,y,z])

    def translation_shoulder(self,armpoint_set, shoulder_point):
        """
        th1とth2の計算を終えた後に、肩の位置を届く範囲へと平行移動する関数。
        平行移動する点は指定された側の腕の３点（肩、肘、手首）を対象としている。

        入力情報
        body_set: 腕の関節の三次元座標の配列　array[joiny num][x,y,z]
        shoulder_point: 平行移動させる肩の点の３次元座標

        出力情報: 肩の点を移動させた後の腕の関節点の配列

        """
        vector_ref = shoulder_point - armpoint_set[1]

        new_set = copy.copy(armpoint_set)
        new_set[1:] += vector_ref
        return new_set

    def calculate_th1(self,shoulder_point,handside,body_set = None):
        """
        肩の点から、th1を求める関数。

        """
        sh_p = copy.copy(shoulder_point)

        if handside == 'right':
            sh_p[1] = -sh_p[1]

        #self.plot_bodypoint(bodypoint = sh_p, defaultpoint = True, body_set = body_set)
        #print('sh_p = {}'.format(sh_p))

        vector_sh = sh_p-self.joint_point[2]
        #vector_sh[2] = 0

        vector2d_sh = vector_sh[0:2]
        #bl = np.array([self.joint_point[2],self.joint_point[2]+vector_sh])
        #self.plot_bodypoint(bodypoint = bl, defaultpoint = True, body_set = body_set)

        vector2d_sh = vector2d_sh/np.linalg.norm(vector2d_sh)
        vector_ref = (self.joint_point[3] - self.joint_point[2])[0:2]
        vector_ref = vector_ref/np.linalg.norm(vector_ref)

        angle_goal = np.arccos(np.sum(vector2d_sh*vector_ref))

        th = [angle_goal,0,0,0,0,0,0,0]
        bp_p = self.DH_calculation(th, details = False, bodypoint = True)
        th = [-angle_goal,0,0,0,0,0,0,0]
        bp_m = self.DH_calculation(th, details = False, bodypoint = True)

        diff_p = sh_p - bp_p[4]
        diff_m = sh_p - bp_m[4]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        #th = [angle_goal,0,0,0,0,0,0,0]
        #bp = self.DH_calculation(th, details = True, bodypoint = True)
        #self.plot_bodypoint(bodypoint = bp, defaultpoint = True, body_set = body_set)

        return angle_goal

    def calculate_th2(self,shoulder_point,handside,body_set = None):
        """
        肩の点から、th2を求める関数。

        """
        sh_p = copy.copy(shoulder_point)
        if handside == 'right':
            sh_p[1] = -sh_p[1]

        #self.plot_bodypoint(bodypoint = sh_p, defaultpoint = True, body_set = body_set)
        #print('sh_p = {}'.format(sh_p))

        vector_sh = sh_p-self.joint_point[3]
        vector_sh[0] = 0

        vector2d_sh = vector_sh[1:3]
        #bl = np.array([self.joint_point[3],self.joint_point[3]+vector_sh])
        #self.plot_bodypoint(bodypoint = bl, defaultpoint = True, body_set = body_set)

        vector2d_sh = vector2d_sh/np.linalg.norm(vector2d_sh)

        vector_ref = (self.joint_point[4] - self.joint_point[3])[1:3]
        vector_ref = vector_ref/np.linalg.norm(vector_ref)

        angle_goal = np.arccos(np.sum(vector2d_sh*vector_ref))

        th = [0, angle_goal,0,0,0,0,0,0]
        bp_p = self.DH_calculation(th, details = False, bodypoint = True)
        th = [0,-angle_goal,0,0,0,0,0,0]
        bp_m = self.DH_calculation(th, details = False, bodypoint = True)

        diff_p = sh_p - bp_p[4]
        diff_m = sh_p - bp_m[4]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        #th = [0,angle_goal,0,0,0,0,0,0]
        #bp = self.DH_calculation(th, details = True, bodypoint = True)
        #self.plot_bodypoint(bodypoint = bp, defaultpoint = True, body_set = body_set)

        return angle_goal

    def calculate_th3(self,th_list,armpoint,handside):

        #構造的な違いの埋め合わせを行うため。ただただこれをすると脇を開けすぎになる可能性あり。
        thr_p = self.joint_point[2]
        sh_p = copy.copy(armpoint[0])
        el_p = copy.copy(armpoint[1])
        wr_p = copy.copy(armpoint[2])

        if handside == 'right':
            sh_p[1] = -sh_p[1]
            el_p[1] = -el_p[1]
            wr_p[1] = -wr_p[1]


        vector_sh2el_data = el_p - sh_p
        vector_sh2el_data = vector_sh2el_data/np.linalg.norm(vector_sh2el_data)
        bp = self.DH_calculation(th_list, details = False, bodypoint = True)

        vector_sh2el_robot = bp[6] - bp[4]
        vector_sh2el_robot = vector_sh2el_robot/np.linalg.norm(vector_sh2el_robot)

        #angle_goal = np.arccos(np.sum(vector_sh2el_data*vector_sh2el_robot))
        #print('angle_goal th3 = {}'.format(angle_goal))

        vector_ref = (sh_p - thr_p)/np.linalg.norm(sh_p - thr_p)
        vector_sh2el_data_vertical = np.cross(vector_sh2el_data,vector_ref)
        vector_sh2el_robot_vertical = np.cross(vector_sh2el_robot,vector_ref)
        vector_sh2el_data_vertical = vector_sh2el_data_vertical/np.linalg.norm(vector_sh2el_data_vertical)
        vector_sh2el_robot_vertical = vector_sh2el_robot_vertical/np.linalg.norm(vector_sh2el_robot_vertical)

        angle_goal = np.arccos(np.sum(vector_sh2el_data_vertical*vector_sh2el_robot_vertical))
        #print('angle_goal th3(cross.ver) = {}'.format(angle_goal))

        th = copy.copy(th_list)
        th[2] = angle_goal
        bp_p = self.DH_calculation(th, details = False, bodypoint = True)
        #debug
        #self.plot_bodypoint(bodypoint = bp_p, figname = 'bp_p', defaultpoint = True, handside = handside,bodypoint2=armpoint)
        th[2] = -angle_goal
        bp_m = self.DH_calculation(th, details = False, bodypoint = True)
        #debug
        #self.plot_bodypoint(bodypoint = bp_m, figname = 'bp_m', defaultpoint = True, handside = handside,bodypoint2=armpoint)


        diff_p = wr_p - bp_p[6]
        diff_m = wr_p - bp_m[6]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        return angle_goal

    def calculate_th4(self,armpoint,handside):
        """
        脇の開閉を計算する関数。

        input
        armpoint: 腕の関節点 array[shoulder,elbow,wrist(,finger)]
        """


        #構造的な違いの埋め合わせを行うため。ただただこれをすると脇を開けすぎになる可能性あり。
        thr_p = copy.copy(self.joint_point[2])
        sh_p = copy.copy(armpoint[0])
        el_p = copy.copy(armpoint[1])


        if handside == 'right':
            sh_p[1] = -sh_p[1]
            el_p[1] = -el_p[1]

        #===============================================================================================================
        # self.plot_bodypoint(bodypoint = thr_p, defaultpoint = True, figname = 'thr')
        # self.plot_bodypoint(bodypoint = sh_p, defaultpoint = True, figname = 'sh')
        # self.plot_bodypoint(bodypoint = el_p, defaultpoint = True, figname = 'el')
        #===============================================================================================================

        vector1 = thr_p - sh_p
        vector2 = el_p - sh_p
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = vector2/np.linalg.norm(vector2)

        angle_goal = np.arccos(np.sum(vector1*vector2))
        #print('angle_goal(th4) = {}'.format(angle_goal/np.pi*180))

        return angle_goal - np.pi*(90+3)/180

    def calculate_th5(self,th_list,armpoint,handside):
        el_p = copy.copy(armpoint[1])
        wr_p = copy.copy(armpoint[2])

        if handside == 'right':
            el_p[1] = -el_p[1]
            wr_p[1] = -wr_p[1]

        vector_el2wr = wr_p - el_p
        T8 = self.DH_calculation(th_list, jointnum = 8, bodyarray = True)

        vector_cal = T8[0:3,2]/np.linalg.norm(T8[0:3,2])
        vector_el2wr = vector_el2wr/np.linalg.norm(vector_el2wr)
        #angle_goal = np.arccos(np.sum(vector_cal*vector_el2wr))
        #print('angle_goal(th5) = {}'.format(angle_goal/np.pi*180))


        bp = self.DH_calculation(th_list, details = False, bodypoint = True)
        vector_ref = (bp[5] - bp[4])/np.linalg.norm(bp[5] - bp[4])
        vector_cal_vertical = np.cross(vector_cal,vector_ref)
        vector_el2wr_vertical = np.cross(vector_el2wr,vector_ref)

        vector_cal_vertical = vector_cal_vertical/np.linalg.norm(vector_cal_vertical)
        vector_el2wr_vertical = vector_el2wr_vertical/np.linalg.norm(vector_el2wr_vertical)
        angle_goal = np.arccos(np.sum(vector_el2wr_vertical*vector_cal_vertical))
        #print('angle_goal(th5 cross.ver) = {}'.format(angle_goal/np.pi*180))

        th = copy.copy(th_list)
        th[4] = angle_goal
        bp_p = self.DH_calculation(th, details = False, bodypoint = True)
        th[4] = -angle_goal
        bp_m = self.DH_calculation(th, details = False, bodypoint = True)

        #self.plot_bodypoint(bodypoint = bp_p, body_set = body_set, figname = 'th5 test', defaultpoint = True)
        #self.plot_bodypoint(bodypoint = bp_m, body_set = body_set, figname = 'th5 test', defaultpoint = True)
        diff_p = wr_p - bp_p[-1]
        diff_m = wr_p - bp_m[-1]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        return angle_goal

    def calculate_th6(self,armpoint,handside):

        sh_p = armpoint[0]
        el_p = armpoint[1]
        wr_p = armpoint[2]

        #===============================================================================================================
        # self.plot_bodypoint(bodypoint = sh_p, defaultpoint = True, figname = 'sh')
        # self.plot_bodypoint(bodypoint = el_p, defaultpoint = True, figname = 'el')
        # self.plot_bodypoint(bodypoint = wr_p, defaultpoint = True, figname = 'wr')
        #===============================================================================================================

        vector1 = sh_p - el_p
        vector2 = wr_p - el_p
        vector1 = vector1/np.linalg.norm(vector1)
        vector2 = vector2/np.linalg.norm(vector2)

        angle_goal = np.arccos(np.sum(vector1*vector2))
        #print('angle_goal = {}'.format(angle_goal/np.pi*180))

        return np.pi - angle_goal

    def calculate_th7(self,armpoint,wrist_matrix,th_list,handside):
        """
        input
        armpoint: 腕の関節点 array[shoulder,elbow,wrist, finger]
        wrist_matrix: 手首の座標系の行列
        th_list: 前腕以外の関節角度の配列 array[joint_num]
        handside: left or right

        """

        wrist_dh = self.DH_calculation(th = th_list,jointnum = 9, bodyarray = True)

        vector_cal = copy.copy(wrist_dh[0:3,0])
        vector_data = -copy.copy(wrist_matrix[0:3,2])
        if handside == 'right':
            vector_data[1] = -vector_data[1]

        vector_cal /= np.linalg.norm(vector_cal)
        vector_data /= np.linalg.norm(vector_data)

        #self.plot_bodypoint(bodypoint = armpoint, figname = 'cal7', defaultpoint = True, coordiante = wrist_matrix)
        #bp = self.DH_calculation(th = th_list, details = True, bodypoint = True, finger = True)
        #self.plot_bodypoint(bodypoint = bp, figname = 'cal7 DH', defaultpoint = True, coordiante = wrist_dh,handside=handside)

        #===============================================================================================================
        # vector_ref = (wr_p - el_p)/np.linalg.norm(wr_p - el_p)
        # vector_cal_vertical = np.cross(vector_cal,vector_ref)
        # vector_wr2fi_vertical = np.cross(vector_wr2fi,vector_ref)
        # vector_cal_vertical = vector_cal_vertical/np.linalg.norm(vector_cal_vertical)
        # vector_wr2fi_vertical = vector_wr2fi_vertical/np.linalg.norm(vector_wr2fi_vertical)
        #===============================================================================================================
        angle_goal = np.arccos(np.sum(vector_cal*vector_data))

        th = copy.copy(th_list)
        th[6] = angle_goal
        #print('th7 = {}'.format(angle_goal*180/np.pi))
        #bp_p = self.DH_calculation(th, details = False, bodypoint = True,finger=True)
        cm_p = self.DH_calculation(th, jointnum = 9, bodyarray = True)

        th[6] = -angle_goal
        #bp_m = self.DH_calculation(th, details = False, bodypoint = True,finger=True)
        cm_m = self.DH_calculation(th, jointnum = 9, bodyarray = True)

        #self.plot_bodypoint(bodypoint = bp_p, figname = 'cal7 plus', defaultpoint = True, coordiante = cm_p,handside=handside)
        #self.plot_bodypoint(bodypoint = bp_m, figname = 'cal7 minus', defaultpoint = True, coordiante = cm_m,handside=handside)

        diff_p = vector_data - cm_p[0:3,0]
        diff_m = vector_data - cm_m[0:3,0]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        return angle_goal

    def calculate_th7_old(self,armpoint,th_list,handside):
        """
        input
        armpoint: 腕の関節点 array[shoulder,elbow,wrist, finger]

        th_list: 前腕以外の関節角度の配列 array[joint_num]
        handside: left or right


        """
        el_p = copy.copy(armpoint[1])
        wr_p = copy.copy(armpoint[2])
        fi_p = copy.copy(armpoint[3])

        if handside == 'right':
            el_p[1] = -el_p[1]
            wr_p[1] = -wr_p[1]
            fi_p[1] = -fi_p[1]

        T10 = self.DH_calculation(th_list, jointnum = 10, bodyarray = True, finger = True)
        fi_d = T10[0:3,3]
        #===============================================================================================================
        # bp = self.DH_calculation(th = th_list, details = True, bodypoint = True, finger = True)
        # self.plot_bodypoint(bodypoint = bp, figname = 'cal th7', defaultpoint = True, handside = 'left', coordiante = T10)
        # arm_set = np.array([el_p,wr_p,fi_p])
        # self.plot_bodypoint(bodypoint = arm_set, figname = 'cal th7', defaultpoint = True, handside = 'left', coordiante = T10)
        #===============================================================================================================

        vector_cal = (fi_d - wr_p)/np.linalg.norm(fi_d - wr_p)
        vector_wr2fi = (fi_p - wr_p)/np.linalg.norm(fi_p - wr_p)

        vector_ref = (wr_p - el_p)/np.linalg.norm(wr_p - el_p)
        vector_cal_vertical = np.cross(vector_cal,vector_ref)
        vector_wr2fi_vertical = np.cross(vector_wr2fi,vector_ref)

        vector_cal_vertical = vector_cal_vertical/np.linalg.norm(vector_cal_vertical)
        vector_wr2fi_vertical = vector_wr2fi_vertical/np.linalg.norm(vector_wr2fi_vertical)
        angle_goal = np.arccos(np.sum(vector_wr2fi_vertical*vector_cal_vertical))

        th = copy.copy(th_list)
        th[6] = angle_goal
        bp_p = self.DH_calculation(th, details = False, bodypoint = True,finger=True)
        th[6] = -angle_goal
        bp_m = self.DH_calculation(th, details = False, bodypoint = True,finger=True)

        diff_p = fi_p - bp_p[-1]
        diff_m = fi_p - bp_m[-1]
        if np.linalg.norm(diff_m) < np.linalg.norm(diff_p):
            angle_goal = -angle_goal

        return angle_goal


    def adjust_elbow(self,p5,p7,p9,prev97,prev95):
        v95b = np.sqrt(prev95[0]**2+prev95[1]**2+prev95[2]**2)
        v97b = np.sqrt(prev97[0]**2+prev97[1]**2+prev97[2]**2)
        ve95 = prev95/v95b
        ve97 = prev97/v97b
        verticalToplane = np.cross(ve97,ve95)
        temp = (v95b**2+(self.l6+self.l7)**2-(self.l4+self.l5)**2)/(2*v95b*(self.l6+self.l7))
        #print(temp)
        if temp >= 1:
            print("OverOver")
            temp = 0.99
        elif temp <= -1:
            print("mOverOver")
            temp = -0.99
        thb = np.arccos(temp)
        #===============================================================================================================
        # if thb == np.nan:
        #     thb = 0.01
        #===============================================================================================================
        point = p9 + ve95*(self.l7+self.l6)*np.cos(thb)
        verticalTo95 = np.cross(ve95,verticalToplane)
        verticalTo95b = np.sqrt(verticalTo95[0]**2+verticalTo95[1]**2+verticalTo95[2]**2)
        verticalTo95e = verticalTo95/verticalTo95b
        goal1 = point + verticalTo95e*(self.l7+self.l6)*np.sin(thb)
        goal2 = point - verticalTo95e*(self.l7+self.l6)*np.sin(thb)

        err1 = np.linalg.norm(p7-goal1)
        err2 = np.linalg.norm(p7-goal2)

        if err1<err2:
            return goal1
        else :
            return goal2

    def set_fixed_value(self,joint_num):
        if joint_num == 7:
            return self.forearm_direction['side']
        elif joint_num == 8:
            return 125

    def angle2command(self,th,i,handside):
        """
        角度から、ロボットの指令値に変換する関数。
        ８番目（手首）の計算方法はfilter_limit_angleと同様に再検討が必要。


        """
        #thがstringの時に固定値を返す
        if type(th) is np.str_:
            try:
                th = float(th)
            except ValueError:
                #print('th = {}'.format(th))
                return self.set_fixed_value(joint_num = i)

        if i<1 or 8<i:
            return -1
        elif i == 8:
            th = np.pi*self.limit_angle[handside][i-1][1]/180-th

        thd = th/np.pi*180
        cua = 255/(self.limit_angle[handside][i-1][1]-self.limit_angle[handside][i-1][0])
        com = int(cua*(thd-self.limit_angle[handside][i-1][0]))

        if -1 < com and com < 256:
            pass
        elif com <= -1:
            #print("limit"+str(i)+"previous value :" + str(com))
            com = 0
        else:
            #print("limit"+str(i)+"previous value :" + str(com))
            com = 255

        return com

    def command2angle(self,com,i,handside):
        """
        指令値から、角度へと変換する関数。
        """

        if i == 9:
            i = 8

        if i<1 or 8<i:
            return -1

        cua = 255/(self.limit_angle[handside][i-1][1]-self.limit_angle[handside][i-1][0])
        thd = com/cua + self.limit_angle[handside][i-1][0]
        th = thd*np.pi/180
        if i == 8:
            th = -np.pi*self.limit_angle[handside][i-1][1]/180 + th
        return th

    def angle2command_set(self,angle_set,handside,finger_value = None):
        """
        角度の配列を指令値の配列に変換する。
        出力した指令値情報は、ERICAの指令値順で出力する。


        input
        angle_set: 角度計算時の順序の角度の配列 array[joint_num]
        handside: left or right
        finger_value:指の指令値　numpyのarray[thumb,index finger, other finger]

        output:指令値順の並びの指令値の情報の配列 array[joint_num]
        """
        com_set = np.array([])

        for ind,ang in enumerate(angle_set):
            temp = self.angle2command(ang, ind+1, handside)
            com_set = np.append(com_set,temp)
        com_set = self.angle2command_order(com_set)

        if not finger_value is None:
            com_set = np.append(com_set,finger_value)

        return com_set

    def command2angle_set(self,command_set,handside):
        """
        指令値の配列を角度の配列に変換する。
        出力した角度情報は、ERICAのジョイントの角度計算時の順で出力される。

        input
        command_set: ERICAの指令値順の配列(指の情報はない想定)
        handside: left or right

        output: ジョイント構造順の角度情報の配列 array[joint_num]
        """

        command_set = self.command2angle_order(command_set)

        angle_set = np.array([])
        for ind,com in enumerate(command_set):
            temp = self.command2angle(com, ind+1, handside)
            angle_set = np.append(angle_set,temp)
        #print('angle_set = {}'.format(angle_set))

        return angle_set

    def angle2command_order(self,angle_order):
        """
        実際の構造（角度計算の時）の順序から、指令値送信の時の順序へと変換する関数。
        """
        ao = angle_order
        return np.array([ao[1],ao[0],ao[3],ao[2],ao[4],ao[5],ao[6],ao[7],ao[7]])

    def command2angle_order(self,command_order):
        """
        指令値の順序から、実際の構造（角度計算時）の順序へと変換する関数。
        指の情報が入っていても、関係なし。手首のジョイントまでの情報を返す。
        """

        co = command_order
        return np.array([co[1],co[0],co[3],co[2],co[4],co[5],co[6],int(co[7]+co[8])])

    def plot_bodypoint(self,bodypoint = None,figname = None,defaultpoint = False,handside = None,coordiante = None,bodypoint2 = None):
        """
        ...

        """

        fig = plt.figure()
        ax = Axes3D(fig)
        limit_value = {'x':np.array([-100,400]),
                       'y':np.array([-250,250]),
                       'z':np.array([   0,500])}

        if not figname is None:
            ax.set_title(figname)

        ax.set_xlim(limit_value['x'])
        ax.set_ylim(limit_value['y'])
        ax.set_zlim(limit_value['z'])

        if not bodypoint is None:

            if handside is None or handside == 'left':
                ax.scatter3D(np.ravel(bodypoint.T[0]),np.ravel(bodypoint.T[1]),np.ravel(bodypoint.T[2]),color = 'red')
                ax.plot_wireframe(bodypoint.T[0],bodypoint.T[1],bodypoint.T[2],color = 'green')
            elif handside == 'right':
                ax.scatter3D(np.ravel(bodypoint.T[0]),np.ravel(-bodypoint.T[1]),np.ravel(bodypoint.T[2]),color = 'red')
                ax.plot_wireframe(bodypoint.T[0],-bodypoint.T[1],bodypoint.T[2],color = 'green')

        if not bodypoint2 is None:
            ax.scatter3D(np.ravel(bodypoint2.T[0]),np.ravel(bodypoint2.T[1]),np.ravel(bodypoint2.T[2]),color = 'gold')
            ax.plot_wireframe(bodypoint2.T[0],bodypoint2.T[1],bodypoint2.T[2],color = 'skyblue')

        if not coordiante is None:
            axis_x = coordiante[0:3,0]*20 + coordiante[0:3,3]
            axis_y = coordiante[0:3,1]*20 + coordiante[0:3,3]
            axis_z = coordiante[0:3,2]*20 + coordiante[0:3,3]

            axis_x = np.array([coordiante[0:3,3],axis_x]).T
            axis_y = np.array([coordiante[0:3,3],axis_y]).T
            axis_z = np.array([coordiante[0:3,3],axis_z]).T

            if handside is None or handside == 'left':
                #ax.scatter3D(np.ravel(axis_x[0]),np.ravel(axis_x[1]),np.ravel(axis_x[2]),color = 'red')
                ax.plot_wireframe(axis_x[0],axis_x[1],axis_x[2],color = 'red')
                #ax.scatter3D(np.ravel(axis_y[0]),np.ravel(axis_y[1]),np.ravel(axis_y[2]),color = 'red')
                ax.plot_wireframe(axis_y[0],axis_y[1],axis_y[2],color = 'yellow')
                #ax.scatter3D(np.ravel(axis_z[0]),np.ravel(axis_z[1]),np.ravel(axis_z[2]),color = 'red')
                ax.plot_wireframe(axis_z[0],axis_z[1],axis_z[2],color = 'blue')
            else:
                #ax.scatter3D(np.ravel(axis_x[0]),np.ravel(-axis_x[1]),np.ravel(axis_x[2]),color = 'red')
                ax.plot_wireframe(axis_x[0],-axis_x[1],axis_x[2],color = 'red')
                #ax.scatter3D(np.ravel(axis_y[0]),np.ravel(-axis_y[1]),np.ravel(axis_y[2]),color = 'red')
                ax.plot_wireframe(axis_y[0],-axis_y[1],axis_y[2],color = 'yellow')
                #ax.scatter3D(np.ravel(axis_z[0]),np.ravel(-axis_z[1]),np.ravel(axis_z[2]),color = 'red')
                ax.plot_wireframe(axis_z[0],-axis_z[1],axis_z[2],color = 'blue')



        if defaultpoint:
            ax.scatter3D(np.ravel( self.joint_point.T[0]),
                         np.ravel( self.joint_point.T[1]),
                         np.ravel( self.joint_point.T[2]),color = 'orange')
            ax.plot_wireframe( self.joint_point.T[0],
                               self.joint_point.T[1],
                               self.joint_point.T[2],color = 'pink')
            ax.scatter3D(np.ravel( self.joint_point.T[0]),
                         np.ravel(-self.joint_point.T[1]),
                         np.ravel( self.joint_point.T[2]),color = 'orange')
            ax.plot_wireframe( self.joint_point.T[0],
                              -self.joint_point.T[1],
                               self.joint_point.T[2],color = 'pink')
        plt.show()

    def get_T0(self):
        return np.array([
            [1, 0, 0, 33],
            [0, 1, 0, 0],
            [0, 0, 1, 508],
            [0, 0, 0, 1]
        ])

    def get_A1(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, self.l0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_A2(self, theta1):
        return np.array([
            [-np.sin(theta1), 0, np.cos(theta1), -self.l15*np.sin(theta1) - self.l1*np.cos(theta1)],
            [np.cos(theta1), 0, np.sin(theta1), self.l15*np.cos(theta1) - self.l1*np.sin(theta1)],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    def get_A3(self, theta2):
        return np.array([
            [-np.sin(theta2), 0, np.cos(theta2), self.l2*np.cos(theta2)],
            [np.cos(theta2), 0, np.sin(theta2), self.l2*np.sin(theta2)],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    def get_A4(self, theta3):
        return np.array([
            [-np.cos(theta3), 0, -np.sin(theta3), 0],
            [-np.sin(theta3), 0, np.cos(theta3), 0],
            [0, 1, 0, self.l3],
            [0, 0, 0, 1]
        ])

    def get_A5(self, theta4):
        return np.array([
            [-np.sin(theta4), 0, np.cos(theta4), self.l4*np.cos(theta4)],
            [np.cos(theta4), 0, np.sin(theta4), self.l4*np.sin(theta4)],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    def get_A6(self, theta5):
        return np.array([
            [-np.sin(theta5), 0, np.cos(theta5), 0],
            [np.cos(theta5), 0, np.sin(theta5), 0],
            [0, 1, 0, self.l5],
            [0, 0, 0, 1]
        ])

    def get_A7(self, theta6):
        return np.array([
            [-np.cos(theta6), 0, np.sin(theta6), self.l6*np.sin(theta6)],
            [np.sin(theta6)*np.cos(np.pi/18), -np.size(np.pi/18), np.cos(theta6)*np.cos(np.pi/18), self.l6*np.cos(theta6)*np.cos(np.pi/18)],
            [np.sin(theta6)*np.sin(np.pi/18), np.cos(np.pi/18), np.cos(theta6)*np.sin(np.pi/18), self.l6*np.cos(theta6)*np.sin(np.pi/18)],
            [0, 0, 0, 1]
        ])

    def get_A8(self, theta7):
        return np.array([
            [-np.cos(theta7), 0, -np.sin(theta7), 0],
            [-np.sin(theta7), 0, np.cos(theta7), 0],
            [0, 1, 0, self.l7],
            [0, 0, 0, 1]
        ])
        
    
    def angle2position(self, ang, side):
        '''
        Convert single arm joint angle to joint position.

        Arguments:
            ang -- 1D array. thetas from th1 -> th8 of one arm. In radius

        Returns:
            pos -- 2D array. (num_joint, 3(x,y,z))
        '''

        # Calculate transformation matrix
        A1 = self.get_A1()
        A2 = self.get_A2(ang[0])
        A3 = self.get_A3(ang[1])
        A4 = self.get_A4(ang[2])
        A5 = self.get_A5(ang[3])
        A6 = self.get_A6(ang[4])
        A7 = self.get_A7(ang[5])
        A8 = self.get_A8(ang[6])

        # Forward kinematics
        T0 = self.get_T0()
        T1 = np.dot(T0, A1)
        T2 = np.dot(T1, A2)
        T3 = np.dot(T2, A3)
        T4 = np.dot(T3, A4)
        T5 = np.dot(T4, A5)
        T6 = np.dot(T5, A6)
        T7 = np.dot(T6, A7)
        T8 = np.dot(T7, A8)

        # Get position data
        pos = np.array([
            T0[:3, -1],
            T1[:3, -1],
            T2[:3, -1], 
            T3[:3, -1], 
            T4[:3, -1], 
            T5[:3, -1], 
            T6[:3, -1], 
            T7[:3, -1], 
            T8[:3, -1],
        ])

        if side == 'left':
            return pos
        elif side == 'right':
            pos[:, 1] = - pos[:, 1].copy()
            return pos
        

    def angle2position_set(self, angle):
        '''
        Convert both arm joint angle to joint position.

        Arguments:
            angle -- Dict for left and right arm joint angle of all timesteps. The order on each arm is th1 -> th8. In rad

        Returns:
            pos_d -- Dict for left and right arm joint position. The order on each arm is T0 -> T8.
        '''

        pos_d = {}

        side = 'left'
        ang = angle['left']
        pos = np.empty(shape=(0, 9, 3)) # 9: T0 -> T8; 3: x, y, z
        for a in ang:
            p = self.angle2position(a, side)
            pos = np.append(pos, np.expand_dims(p, 0), axis=0)
        pos_d[side] = pos

        side = 'right'
        ang = angle['right']
        pos = np.empty(shape=(0, 9, 3)) # 9: T0 -> T8; 3: x, y, z
        for a in ang:
            p = self.angle2position(a, side)
            pos = np.append(pos, np.expand_dims(p, 0), axis=0)
        pos_d[side] = pos

        return pos_d
