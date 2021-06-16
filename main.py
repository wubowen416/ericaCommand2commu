"""Batch script.

Convert all motion editor files in input_dir to commu command files,
save in output_dir with same filename.

For ERICA and commu near to SHINTANI.

<start_joint> -- Start joint index for calculating angles of CommU
<end_joint> -- End joint index for calculating angles of CommU
For joint index configs, refer to "./memo/joint_index.PNG"

Example: python main inputs outputs 3 8

Usage:
    main.py <input_dir> <output_dir> <start_joint> <end_joint> 
"""

from commandmanager import CommandManager
from commuClass import CommUClass
from angleCalculator import angleCalculator
from tools.functions import prepend_line
from docopt import docopt
import os
import numpy as np


def convert(input_path, output_path, START_JOINT, END_JOINT):

    cm = CommandManager('ERICA')
    ac = angleCalculator()
    cc = CommUClass()

    angle_data = cm.command2angle(input_path)
    position_data = cm.robot_class.angle2position_set(angle_data)
    ang_data = ac(position_data, START_JOINT, END_JOINT)
    sheet = cc.angle2command(ang_data).T

    # Write to file
    np.savetxt(output_path, sheet.astype(int), delimiter="\t", fmt='%i')

    # Prepend header
    line_0 = "comment:2nd row is default values: 3rd is axis number: 4th are motion inverval, motion steps, motion axes"
    line_1 = "0\t0\t80\t-4\t-80\t4\t0\t0\t0\t0\t0\t0\t-5\t0"
    line_2 = "0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13"
    line_3 = "10\t{}\t14".format(len(sheet))
    prepend_line(output_path, line_3)
    prepend_line(output_path, line_2)
    prepend_line(output_path, line_1)
    prepend_line(output_path, line_0)

    # Append last line
    # Open a file with access mode 'a'
    last_line = "*" + " 111" * 14
    with open(output_path, "a") as f:
    # Append 'hello' at the end of file
        f.write(last_line)


def batch_process(input_dir, output_dir, sj, ej):
    filenames = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, filename in enumerate(filenames):
        print(f"({i+1}/{len(filenames)}) Processing {filename}...")
        convert(os.path.join(input_dir, filename), os.path.join(output_dir, filename), sj, ej)
    

if __name__ == "__main__":

    args = docopt(__doc__)
    batch_process(args["<input_dir>"], args["<output_dir>"], int(args["<start_joint>"]), int(args["<end_joint>"]))



# from commandmanager import CommandManager
# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt

# cm = CommandManager('ERICA')
# angle_data = cm.command2angle('test.txt')

# position_data = cm.robot_class.angle2position_set(angle_data)

# fig = plt.figure(dpi=200)
# ax = fig.add_subplot(projection='3d')

# pos = position_data['left'][0]
# for p in pos:
#     x, y, z = p
#     ax.scatter(x, y, z)
# for i in range(len(pos)-1):
#     x, y, z = pos[i+1]
#     xp, yp, zp = pos[i]
#     ax.plot([xp, x], [yp, y], [zp, z])

# pos = position_data['right'][0]
# for p in pos:
#     x, y, z = p
#     ax.scatter(x, y, z)
# for i in range(len(pos)-1):
#     x, y, z = pos[i+1]
#     xp, yp, zp = pos[i]
#     ax.plot([xp, x], [yp, y], [zp, z])


# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_xlim((-100, 400))
# ax.set_ylim((-250, 250))
# ax.set_zlim((0, 500))

# plt.show()

