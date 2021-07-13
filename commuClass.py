import numpy as np
from numpy.lib.function_base import angle
from scipy import signal

from tools.JsonConfig import JsonConfig


class CommUClass:

    def __init__(self):

        # print("Initialize CommuCommander..")
        
        config = JsonConfig('./config/CommUConfig.json')

        self.max_speed = config.Commu.max_speed
        self.denominator = config.Commu.denominator
        self.data_fps = config.Data.fps
        self.send_fps = config.Commu.fps
        self.command = config.Command.command

        self.constraint = config.Constraint

      
    def angle2command(self, angle_data):

        length = len(angle_data['left']['theta'])

        # FPS
        if self.send_fps != self.data_fps:
            mag = self.send_fps // self.data_fps
            for side, ang_names in angle_data.items():
                for ang_name, vec in ang_names.items():
                    angle_data[side][ang_name] = signal.resample(vec, mag*length)

        left_thetas = angle_data['left']['theta']
        left_phis = angle_data['left']['phi']
        right_thetas = angle_data['right']['theta']
        right_phis = angle_data['right']['phi']

        # Adjust rotation direction for CommU

        def right_yoko_map(x):
            x_abs = np.abs(x)
            if  x_abs < 25:
                return - (25 - x_abs)
            elif 25 < x_abs:
                return x_abs - 25

        def left_yoko_map(x):
            x_abs = np.abs(x)
            if x_abs < 25:
                return 25 - x_abs
            elif x_abs > 25:
                return -(x_abs - 25)

        left_yoko_euler = np.array(list(map(left_yoko_map, left_thetas))).astype(int)
        left_tate_euler = (90-np.abs(left_phis)).astype(int)
        right_yoko_euler = np.array(list(map(right_yoko_map, right_thetas))).astype(int)
        right_tate_euler = (np.abs(right_phis)-90).astype(int)

        # Constraint
        # 1 Add constraint
        left_yoko_euler[left_yoko_euler > self.constraint.Left.Yoko.upper] = self.constraint.Left.Yoko.upper
        left_yoko_euler[left_yoko_euler < self.constraint.Left.Yoko.lower] = self.constraint.Left.Yoko.lower
        right_yoko_euler[left_yoko_euler > self.constraint.Right.Yoko.upper] = self.constraint.Right.Yoko.upper
        right_yoko_euler[left_yoko_euler < self.constraint.Right.Yoko.lower] = self.constraint.Right.Yoko.lower

        # 2 Collide with body
        # 2-A left hand
        def f(x):
            return -0.75 * x + 37.5
        def inv_f(y):
            return (37.5 - y) / 0.75

        for i, (x, y) in enumerate(zip(left_tate_euler, left_yoko_euler)):

            if x > 10 and x < 50 and y > 0:
                y_of_x = f(x)
                if y >= y_of_x: # Above the line
                    x_of_y = inv_f(y)
                    y = (y + y_of_x) / 2
                    x = (x + x_of_y) / 2

            if x >= 50 and y > 0:
                y = 0
            
            if x < -40 and y > 40:
                x_dis = -x - 40
                y_dis = y - 40
                if x_dis < y_dis:
                    x = -40
                else:
                    y = 40

            left_tate_euler[i] = int(x)
            left_yoko_euler[i] = int(y) 

        # 2-B right hand
        def f(x):
            return -0.75 * x - 37.5
        def inv_f(y):
            return -(37.5 + y) / 0.75

        for i, (x, y) in enumerate(zip(right_tate_euler, right_yoko_euler)):

            if x < -50 and y < 0 :
                y = 0

            if x < -10 and y < 0 and x > -50:
                y_of_x = f(x)
                if y <= y_of_x: # Above the line
                    x_of_y = inv_f(y)
                    y = (y + y_of_x) / 2
                    x = (x + x_of_y) / 2
            
            if x > 40 and y < -40:
                x_dis = x - 40
                y_dis = - y - 40
                if x_dis < y_dis:
                    x = 40
                else:
                    y = -40
            
            right_yoko_euler[i] = int(y)
            right_tate_euler[i] = int(x)


        # Write command
        sheet = np.full(shape=(14, length), fill_value=-10000, dtype=int)

        # lines = []

        for i in range(length):

            # line = self.command

            if i == 0:
                speed_2 = self.max_speed
                speed_3 = self.max_speed
                speed_4 = self.max_speed
                speed_5 = self.max_speed

            else:
                speed_2 = int(np.abs((left_tate_euler[i] - left_tate_euler[i-1]) / (1 / self.send_fps))) // self.denominator
                speed_3 = int(np.abs((left_yoko_euler[i] - left_yoko_euler[i-1]) / (1 / self.send_fps))) // self.denominator
                speed_4 = int(np.abs((right_tate_euler[i] - right_tate_euler[i-1]) / (1 / self.send_fps))) // self.denominator
                speed_5 = int(np.abs((right_yoko_euler[i] - right_yoko_euler[i-1]) / (1 / self.send_fps))) // self.denominator

            if speed_2 != 0:
                # line += f" 2 {left_tate_euler[i]} {speed_2}"
                sheet[2, i] = left_tate_euler[i]

            if speed_3 != 0:
                # line += f" 3 {left_yoko_euler[i]} {speed_3}"
                sheet[3, i] = left_yoko_euler[i]

            if speed_4 != 0:
                # line += f" 4 {right_tate_euler[i]} {speed_4}"
                sheet[4, i] = right_tate_euler[i]

            if speed_5 != 0:
                # line += f" 5 {right_yoko_euler[i]} {speed_5}"
                sheet[5, i] = right_yoko_euler[i]

            # if speed_2 == 0 and speed_3 == 0 and speed_4 == 0 and speed_5 == 0:
                # line = "skip"
                # pass

            if speed_2 == 0:
                if i == 0:
                    sheet[2, i] = 80
                else:
                    sheet[2, i-1] = left_tate_euler[i-1]
            if speed_3 == 0:
                if i == 0:
                    sheet[3, i] = -4
                else:
                    sheet[3, i-1] = left_yoko_euler[i-1]
            if speed_4 == 0:
                if i == 0:
                    sheet[4, i] = -80
                else:
                    sheet[4, i-1] = right_tate_euler[i-1]
            if speed_5 == 0:
                if i == 0:
                    sheet[4, i] = 4
                else:
                    sheet[5, i-1] = right_yoko_euler[i-1]

            # lines.append(line + '\n')

        return sheet
    