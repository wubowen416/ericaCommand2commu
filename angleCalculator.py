import numpy as np

class angleCalculator:

    def __init__(self) -> None:
        
        pass

    def __call__(self, position_data, si, ei):

        '''
        Calculate theta and phi for selected joint.

        Arguments:
            position_data -- Dict for left and right arm joint position. The order on each arm is T0 -> T8. (T(timestep), 9(T0->T8), 3(xyz))
            si -- start point/joint index
            ei -- end point/joint index
        '''

        data = {
            'left': {
                'theta': np.array([]),
                'phi': np.array([])
            },
            'right': {
                'theta': np.array([]),
                'phi': np.array([])
            }
        }

        side = 'left'
        pos = position_data[side]
        pos_s = pos[:, si, :]
        pos_e = pos[:, ei, :] # (T, 3)
        vec = pos_e - pos_s # (T, 3)
        vec = self.normalize(vec)
        for v in vec: # (3,)
            x, y, z = v
            theta = self.calculate_theta(y, z)
            phi = self.calculate_phi(x, z)
            data[side]['theta'] = np.append(data[side]['theta'], theta)
            data[side]['phi'] = np.append(data[side]['phi'], phi)

        side = 'right'
        pos = position_data[side]
        pos_s = pos[:, si, :]
        pos_e = pos[:, ei, :] # (T, 3)
        vec = pos_e - pos_s # (T, 3)
        vec = self.normalize(vec)
        for v in vec: # (3,)
            x, y, z = v
            theta = self.calculate_theta(y, z)
            phi = self.calculate_phi(x, z)
            data[side]['theta'] = np.append(data[side]['theta'], theta)
            data[side]['phi'] = np.append(data[side]['phi'], phi)

        return data


    def calculate_theta(self, y, z):

        y_abs = np.abs(y)
        z_abs = np.abs(z)

        if z < 0 and y < 0:
            theta = - np.rad2deg(np.arctan(y_abs/z_abs))
        elif z < 0 and y > 0:
            theta = np.rad2deg(np.arctan(y_abs/z_abs))
        elif z > 0 and y < 0:
            theta = - (90 + np.rad2deg(np.arctan(z_abs/y_abs)))
        else:
            theta = 90 + np.rad2deg(np.arctan(z_abs/y_abs))

        return theta

    def calculate_phi(self, x, z):

        x_abs = np.abs(x)
        z_abs = np.abs(z)

        if x > 0 and z < 0:
            phi = -np.rad2deg(np.arctan(x_abs/z_abs))
        elif x > 0 and z > 0:
            phi = -90 - np.rad2deg(np.arctan(z_abs/x_abs))
        elif x < 0 and z < 0:
            phi = np.rad2deg(np.arctan(x_abs/z_abs))
        else:
            phi = 90 + np.rad2deg(np.arctan(z_abs/x_abs))

        return phi


    def normalize(self, arr, axis=-1, order=2):
        l2 = np.linalg.norm(arr, ord=order, axis=axis, keepdims=True)
        l2[l2==0] = 1
        return arr / l2
    

