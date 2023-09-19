#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import time
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from collections import deque
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from tf.transformations import *
import itertools

import cv2
from sensor_msgs.msg import Image,CompressedImage
import sys
import ros_numpy

class Env():

    #ゴール画像処理用
    goal_blue_lower = np.array([85, 100, 50])
    goal_blue_upper = np.array([130, 255, 255])

    goal_green_lower = np.array([55, 100, 60])
    goal_green_upper = np.array([85, 255, 255])

    # orange_lower = np.array([0, 150, 90])
    # orange_upper = np.array([20, 255, 255])
    # yellow_lower = np.array([15, 150, 90])
    # yellow_upper = np.array([35, 255, 255])

    white = [255, 255, 255]
    wall = [0, 0, 0]
    lightgreen = [80, 190, 158]

    def __init__(self,mode,input_list,task):
        self.mode = mode
        self.input_list = input_list
        self.task = task
        self.flag = 'both'
        self.collision2 = 0
        self.pastlist_cam = deque([])
        self.pastlist_lidar = deque([])
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        if ('cam'in self.input_list) or ('pastcam'in self.input_list) :
            # self.sub_img=rospy.Subscriber('usb_cam/image_raw',Image,self.pass_data,queue_size=10) #生データ（通信量が多い）
            self.sub_img=rospy.Subscriber('usb_cam/image_raw/compressed',CompressedImage,self.pass_data,queue_size=10) #圧縮データ#これらがないと上手く画像取得できない
        self.sub_scan=rospy.Subscriber('scan', LaserScan,self.pass_data,queue_size=10)

        self.lidar_max=1.8 #対象のworldにおいて取りうるlidar最大値(simの貫通対策や正規化に使用)
        self.lidar_min=0.12 #lidarの最小測距値(m)
        self.range_margin = self.lidar_min+0.01 # >0.12 衝突判定値(m) ＊lidarの最小測距値(self.lidar_min)+指定判定値
        self.display_image = True # 入力画像を表示する

    def pass_data(self,data):#画像正常取得用callback
        pass

    def get_lidar(self,restart=False): #lidar情報取得
        data = None
        data_range = []
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=1)
            except:
                # 停止
                vel_cmd = Twist()
                vel_cmd.linear.x = 0 #直進方向0.15m/s
                vel_cmd.angular.z =0  #回転方向 rad/s
                self.pub_cmd_vel.publish(vel_cmd) #実行
                print('waiting lidar')
                pass
        for i in range(len(data.ranges)):
            if data.ranges[i] == float('Inf'):#最大より遠い(inf)なら3.5
                data_range.append(3.5)
            if np.isnan(data.ranges[i]):#欠落値 最小より近いなら0
                data_range.append(0)
            
            if self.mode == 'sim':
                if data.ranges[i] > self.lidar_max:#貫通防止
                    data_range.append(0)
                else:
                    data_range.append(data.ranges[i])
            else:#real
                data_range.append(data.ranges[i])

        if restart:
            #lidar値を360degから45deg刻み8方向に変更（リスタートで用いる）
            use_list=[]
            for i in range(8):
                index=(len(data_range)//8)*i
                data=max(data_range[index-2],data_range[index-1],data_range[index],data_range[index+1],data_range[index+2]) #実機の飛び値対策
                use_list.append(data)
        else:
            #lidar値を360degから15deg刻み24方向に変更（観測情報）
            use_list=[]
            for i in range(24):
                index=(len(data_range)//24)*i
                data=max(data_range[index-2],data_range[index-1],data_range[index],data_range[index+1],data_range[index+2]) #実機の飛び値対策
                use_list.append(data)
        
        data_range = use_list
        return data_range

    def get_camera(self): #camera画像取得
        img = None
        while img is None:
            try:
                # img = rospy.wait_for_message('usb_cam/image_raw', Image, timeout=1) #生データ
                img = rospy.wait_for_message('usb_cam/image_raw/compressed', CompressedImage, timeout=1)#圧縮データ
            except:
                # 停止
                vel_cmd = Twist()
                vel_cmd.linear.x = 0 #直進方向0.15m/s
                vel_cmd.angular.z =0  #回転方向 rad/s
                self.pub_cmd_vel.publish(vel_cmd) #実行
                print('waiting cam')
                pass
        
        #生データ用
        # img = ros_numpy.numpify(img) #py3ではcvbridgeが使えないためros_numpyで代用
        #圧縮データ用
        # '''
        img = np.frombuffer(img.data,np.uint8)
        img = cv2.imdecode(img,cv2.IMREAD_COLOR)#BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # '''

        raw_img=img
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #グレースケール
        img = cv2.resize(img,(48,27))
        disp_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#cv2の標準フォーマットはBGR
        # cv2.imwrite('img.png',disp_img)
        # print('saved')

        # if self.display_image:
        #     cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('camera', 480, 270)
        #     cv2.imshow('camera', disp_img) 
        #     cv2.waitKey(1)
        return img

    def process_image(self,img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if self.flag == 'blue' or self.flag == 'green':
            if self.flag == 'blue':            
                no_goal_mask = cv2.inRange(img_hsv, self.goal_green_lower, self.goal_green_upper)#緑ゴールのマスクを取得
            if self.flag == 'green':
                no_goal_mask = cv2.inRange(img_hsv, self.goal_blue_lower, self.goal_blue_upper)#青ゴールのマスクを取得

            goal_rgb = cv2.cvtColor(no_goal_mask, cv2.COLOR_GRAY2RGB)#マスクをRGBに
            goal_rgb[np.where((goal_rgb == self.white).all(axis=2))] = self.wall#白い部分(ゴール)を壁の色に
            back_mask = cv2.bitwise_not(no_goal_mask) #背景(反転)マスクを取得
            back_rgb = cv2.cvtColor(back_mask, cv2.COLOR_GRAY2RGB)#背景マスクをRGBに
            masked_img = cv2.bitwise_and(img, back_rgb) #ゴールを切り取った画像
            process1 = cv2.addWeighted(masked_img, 1, goal_rgb, 1, 0)  #合成

            process1_hsv = cv2.cvtColor(process1, cv2.COLOR_RGB2HSV)
            if self.flag == 'blue':
                goal_mask = cv2.inRange(process1_hsv, self.goal_blue_lower, self.goal_blue_upper)#青ゴールのマスクを取得
            if self.flag == 'green':
                goal_mask = cv2.inRange(process1_hsv, self.goal_green_lower, self.goal_green_upper)#緑ゴールのマスクを取得

            goal_rgb = cv2.cvtColor(goal_mask, cv2.COLOR_GRAY2RGB)#マスクをRGBに
            goal_rgb[np.where((goal_rgb == self.white).all(axis=2))] = self.lightgreen#白い部分(ゴール)を明るい緑色に
            back_mask = cv2.bitwise_not(goal_mask) #背景(反転)マスクを取得
            back_rgb = cv2.cvtColor(back_mask, cv2.COLOR_GRAY2RGB)#背景マスクをRGBに
            masked_img = cv2.bitwise_and(process1, back_rgb) #ゴールを切り取った画像
            process2 = cv2.addWeighted(masked_img, 1, goal_rgb, 1, 0)  #合成
            processed = process2
        
        elif self.flag == 'both':            
            goal_mask = cv2.inRange(img_hsv, self.goal_blue_lower, self.goal_blue_upper)#青ゴールのマスクを取得

            goal_rgb = cv2.cvtColor(goal_mask, cv2.COLOR_GRAY2RGB)#マスクをRGBに
            goal_rgb[np.where((goal_rgb == self.white).all(axis=2))] = self.lightgreen#白い部分(ゴール)を明るい緑色に
            back_mask = cv2.bitwise_not(goal_mask) #背景(反転)マスクを取得
            back_rgb = cv2.cvtColor(back_mask, cv2.COLOR_GRAY2RGB)#背景マスクをRGBに
            masked_img = cv2.bitwise_and(img, back_rgb) #ゴールを切り取った画像
            process1 = cv2.addWeighted(masked_img, 1, goal_rgb, 1, 0)  #合成

            process1_hsv = cv2.cvtColor(process1, cv2.COLOR_RGB2HSV)
            goal_mask = cv2.inRange(process1_hsv, self.goal_green_lower, self.goal_green_upper)#緑ゴールのマスクを取得

            goal_rgb = cv2.cvtColor(goal_mask, cv2.COLOR_GRAY2RGB)#マスクをRGBに
            goal_rgb[np.where((goal_rgb == self.white).all(axis=2))] = self.lightgreen#白い部分(ゴール)を明るい緑色に
            back_mask = cv2.bitwise_not(goal_mask) #背景(反転)マスクを取得
            back_rgb = cv2.cvtColor(back_mask, cv2.COLOR_GRAY2RGB)#背景マスクをRGBに
            masked_img = cv2.bitwise_and(process1, back_rgb) #ゴールを切り取った画像
            process2 = cv2.addWeighted(masked_img, 1, goal_rgb, 1, 0)  #合成
            processed = process2

        else:
            processed = img

        if self.display_image:
            disp_img = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)#標準フォーマットBGR
            cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('camera', 480, 270)
            cv2.imshow('camera', disp_img)
            cv2.waitKey(1)
        return processed #RGB
        

    def getState(self, scan):#情報取得
        collision = False

        if self.range_margin >=min(scan):#衝突
                collision = True

        ########goal判定#############################################
        goal=False
        goal_count = 0
        goal_count_middle = 0
        near_goal = False #ゴールに近い時、lidarの値を最大に
        if self.task == 'goal':
            img = self.get_camera() #RGB
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #HSV
            goal_blue = cv2.inRange(img_hsv, self.goal_blue_lower, self.goal_blue_upper)
            goal_green = cv2.inRange(img_hsv, self.goal_green_lower, self.goal_green_upper)
            blue_count = np.count_nonzero(goal_blue)
            green_count = np.count_nonzero(goal_green)
            img_hsv_middle = img_hsv[:,len(img_hsv[0])//3:len(img_hsv[0])*2//3]
            goal_blue_middle = cv2.inRange(img_hsv_middle, self.goal_blue_lower, self.goal_blue_upper)
            goal_green_middle = cv2.inRange(img_hsv_middle, self.goal_green_lower, self.goal_green_upper)
            blue_count_middle = np.count_nonzero(goal_blue_middle)
            green_count_middle = np.count_nonzero(goal_green_middle)

            if self.flag == 'blue' or self.flag == 'both':
                goal_count=blue_count
                goal_count_middle = blue_count_middle                
                if blue_count >300:
                    goal=True
                    # print('blue goal!')
                    self.flag = 'green'
                # if blue_count >200:
                #     near_goal = True
            if self.flag == 'green' or self.flag == 'both':
                goal_count=green_count
                goal_count_middle = green_count_middle
                if green_count >300:
                    goal=True
                    # print('green goal!')
                    self.flag = 'blue'
                # if green_count >200:
                #     near_goal = True
            if self.flag == 'both':
                goal_count =blue_count+green_count
                goal_count_middle=blue_count_middle+green_count_middle
        ######################################################################

        ########LiDAR情報の前処理#############################################
        # scan:24方向15度刻み

        scan_list=[]
        #前方7方向30度刻み
        # scan_list.append(scan[0]) #正面
        # for i in range(3):
        #     scan_list.append(scan[2*i+2])
        #     scan_list.append(scan[-(2*i+2)])
        #12方向30度刻み
        for i in range(12):
            scan_list.append(scan[2*i])

        #報酬値計算用、正規化前のデータを用いる
        reward_scan=scan_list
        
        #lidar値の正規化
        input_scan=[]
        if near_goal:
            for i in range(len(scan_list)):
                input_scan.append(self.lidar_max)
        else:
            for i in range(len(scan_list)):
                input_scan.append((scan_list[i]-self.range_margin)/(self.lidar_max-self.range_margin))
        ######################################################################

        return input_scan,reward_scan,collision,goal,goal_count,goal_count_middle

    def setReward(self, state, collision, goal, goal_count, goal_count_middle, t, action):
        reward =0
        if self.task == 'goal':
            if goal:
                reward+=500-t
                # reward += 1000
            # else:
            #     reward += np.sqrt(300*goal_count)/(300*1)
            #     reward += goal_count_middle/10
            #     reward-=2

        if collision:
            reward += -100
        # if min(state)>=0.3:#壁から遠い時
        #     reward +=1
        # if min(state)<=0.2:#壁から近い時
        #     reward += -1
        
        return reward

    def step(self, t, action, evaluate=False):
        vel_cmd = Twist()#steptime:0.2s
        if action == 0:
            vel_cmd.linear.x = 0.20 #直進方向 m/s
            vel_cmd.angular.z = 1.4 #回転方向 rad/s

        if action == 1:
            vel_cmd.linear.x = 0.15 #直進方向 m/s 
            vel_cmd.angular.z = 0 #回転方向 rad/s
        
        if action == 2:
            vel_cmd.linear.x = 0.20 #直進方向 m/s
            vel_cmd.angular.z = -1.4 #回転方向 rad/s
        self.pub_cmd_vel.publish(vel_cmd) #実行
        data = self.get_lidar()
        input_scan,reward_scan,collision,goal,goal_count,goal_count_middle = self.getState(data)
        reward = self.setReward(reward_scan, collision, goal, goal_count, goal_count_middle, t, action)
        
        if not evaluate:
            if goal:
                self.restart(goal)
            elif collision:
                self.restart()
        
        ########入力情報の準備#############################################
        state_list = []
        if ('cam'in self.input_list) or ('pastcam'in self.input_list) :
            img = self.get_camera()
            if self.task == 'goal':
                img = self.process_image(img) #RGB
            img = np.asarray(img, dtype=np.float32)
            img /=255.0
            img = np.asarray(img.flatten())
            img = img.tolist()
            state_list = state_list+img
            if 'pastcam'in self.input_list:
                self.pastlist_cam.append(img)
                if len(self.pastlist_cam)>2:# 常に[過去 現在]の状態にする
                    self.pastlist_cam.popleft() #左端の要素を削除(kako2)
                past_cam = self.pastlist_cam[0]
                state_list = state_list+past_cam

        if ('lidar'in self.input_list) or ('pastlidar'in self.input_list) :
            state_list = state_list+input_scan
            if 'pastlidar' in self.input_list:
                self.pastlist_lidar.append(input_scan)
                if len(self.pastlist_lidar)>2:# 常に[過去 現在]の状態にする
                    self.pastlist_lidar.popleft() #左端の要素を削除(kako2)
                past_scan = self.pastlist_lidar[0]
                state_list = state_list+past_scan
        ###################################################################

        return np.array(state_list), reward ,collision,goal

    def reset(self):
        data = self.get_lidar()
        input_scan,_,collision,_,_,_ = self.getState(data)
        
        ########入力情報の準備#############################################
        state_list = []
        if ('cam'in self.input_list) or ('pastcam'in self.input_list) :
            img = self.get_camera()
            if self.task == 'goal':
                img = self.process_image(img) #RGB
            img = np.asarray(img, dtype=np.float32)
            img /=255.0
            img = np.asarray(img.flatten())
            img = img.tolist()
            state_list = state_list+img
            if 'pastcam'in self.input_list:
                self.pastlist_cam.append(img)
                if len(self.pastlist_cam)>2:# 常に[過去 現在]の状態にする
                    self.pastlist_cam.popleft() #左端の要素を削除(kako2)
                past_cam = self.pastlist_cam[0]
                state_list = state_list+past_cam

        if ('lidar'in self.input_list) or ('pastlidar'in self.input_list) :
            state_list = state_list+input_scan
            if 'pastlidar' in self.input_list:
                self.pastlist_lidar.append(input_scan)
                if len(self.pastlist_lidar)>2:# 常に[過去 現在]の状態にする
                    self.pastlist_lidar.popleft() #左端の要素を削除(kako2)
                past_scan = self.pastlist_lidar[0]
                state_list = state_list+past_scan
        ###################################################################
        return np.array(state_list)

    def restart(self,goal=False):
        # 停止
        vel_cmd = Twist()
        vel_cmd.linear.x = 0 #直進方向0.15m/s
        vel_cmd.angular.z =0  #回転方向 rad/s
        self.pub_cmd_vel.publish(vel_cmd) #実行

        data_range = self.get_lidar(restart=True)

        while True:
            while True:
                vel_cmd.linear.x = 0 #直進方向0m/s
                vel_cmd.angular.z =pi/4  #回転方向 rad/s
                self.pub_cmd_vel.publish(vel_cmd) #実行
                data_range = self.get_lidar(restart=True)
                
                if goal:
                    if data_range.index(max(data_range)) == 0:
                        wall='goal'
                        break
                else:
                    if data_range.index(min(data_range))== 0:#正面
                        wall='front'
                        break
                    if data_range.index(min(data_range))== round(len(data_range)/2): #背面
                        wall='back'
                        break
            
            # 停止
            vel_cmd.linear.x = 0 #直進方向　m/s
            vel_cmd.angular.z =0  #回転方向 rad/s
            self.pub_cmd_vel.publish(vel_cmd) #実行

            if wall == 'goal':
                # while data_range[round(len(data_range)/2)] <= self.range_margin:# ゴール後の衝突を避ける
                while min(data_range) <=self.range_margin:
                    vel_cmd.linear.x = 0.10 #直進方向0.15m/s
                    vel_cmd.angular.z =0  #回転方向 rad/s
                    self.pub_cmd_vel.publish(vel_cmd) #実行
                    data_range = self.get_lidar(restart=True)
                break

            elif wall =='front':
                while data_range[0] < self.range_margin + 0.10: # 衝突値＋10cm
                    vel_cmd.linear.x = -0.10 #直進方向0.15m/s
                    vel_cmd.angular.z =0  #回転方向 rad/s
                    self.pub_cmd_vel.publish(vel_cmd) #実行
                    data_range = self.get_lidar(restart=True)

            elif wall =='back':
                while data_range[round(len(data_range)/2)] < self.range_margin + 0.10:# 衝突値＋10cm
                    vel_cmd.linear.x = 0.10 #直進方向0.15m/s
                    vel_cmd.angular.z =0  #回転方向 rad/s
                    self.pub_cmd_vel.publish(vel_cmd) #実行
                    data_range = self.get_lidar(restart=True)

            # 停止
            vel_cmd.linear.x = 0 #直進方向　m/s
            vel_cmd.angular.z =0  #回転方向 rad/s
            self.pub_cmd_vel.publish(vel_cmd) #実行
            
            side_list = [round(len(data_range)/4),round(len(data_range)*3/4)] #側面
            num = np.random.randint(0,1)
            while True:
                vel_cmd.linear.x = 0 #直進方向0m/s
                if num == 0:
                    vel_cmd.angular.z =pi/4  #回転方向 rad/s
                else:
                    vel_cmd.angular.z =-pi/4  #回転方向 rad/s
                self.pub_cmd_vel.publish(vel_cmd) #実行
                data_range = self.get_lidar(restart=True)
                if data_range.index(min(data_range)) in side_list:
                    break
                
            # 停止
            vel_cmd.linear.x = 0 #直進方向　m/s
            vel_cmd.angular.z =0  #回転方向 rad/s
            self.pub_cmd_vel.publish(vel_cmd) #実行
            data_range = self.get_lidar(restart=True)
            if min(data_range)>self.range_margin+0.05: #restart直後に衝突判定にならないように最低5cm余裕
                break

    def set_robot(self,num=0): #指定位置にロボットを移動させる
        start_time = time.time()
        while True:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0 #直進方向m/s
            vel_cmd.angular.z =0.0  #回転方向 rad/s
            self.pub_cmd_vel.publish(vel_cmd) #実行
            now=time.time()
            if now-start_time>=0.5:
                break
        if num == 0:
            XYZyaw = [-0.4,1.4,0.0,0.0]#初期位置
        if num == 1:
            XYZyaw = [0.0,1.4,0.0,0.0]
        if num == 2:
            XYZyaw = [0.0,1.4,0.0,3.14]
        if num == 3:
            XYZyaw = [0.0,0.4,0.0,0.0]
        if num == 4:
            XYZyaw = [0.0,0.4,0.0,3.14]
        if num == 5:
            XYZyaw = [-0.7,0.9,0.0,0.0]

        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_burger'
        state_msg.pose.position.x = XYZyaw[0]
        state_msg.pose.position.y = XYZyaw[1]
        state_msg.pose.position.z = XYZyaw[2]
        q=quaternion_from_euler(0,0,XYZyaw[3])
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state( state_msg )