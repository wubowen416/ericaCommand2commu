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
            x = np.abs(x)
            if x < 25:
                return x - 25
            else:
                return 70 - x

        def left_yoko_map(x):
            x = np.abs(x)
            if x < 25:
                return 25 - x
            else:
                return x - 70

        
        left_yoko_euler = np.array(list(map(left_yoko_map, left_thetas))).astype(int)
        left_tate_euler = (90-np.abs(left_phis)).astype(int)
        right_yoko_euler = np.array(list(map(right_yoko_map, right_thetas))).astype(int)
        right_tate_euler = (np.abs(right_phis)-90).astype(int)

        # Add constraint
        left_yoko_euler[left_yoko_euler > self.constraint.Left.Yoko.upper] = self.constraint.Left.Yoko.upper
        left_yoko_euler[left_yoko_euler < self.constraint.Left.Yoko.lower] = self.constraint.Left.Yoko.lower
        right_yoko_euler[left_yoko_euler > self.constraint.Right.Yoko.upper] = self.constraint.Right.Yoko.upper
        right_yoko_euler[left_yoko_euler < self.constraint.Right.Yoko.lower] = self.constraint.Right.Yoko.lower

        # Write command

        sheet = np.full(shape=(14, length), fill_value=-1, dtype=int)

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

            if speed_2 == 0 and speed_3 == 0 and speed_4 == 0 and speed_5 == 0:
                # line = "skip"
                pass

            # lines.append(line + '\n')

        return sheet
    