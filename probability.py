import numpy as np


def calc_joint_probability(transition_matrix, p, char_arr):
    n, m = transition_matrix.shape
    joint_matrix = np.zeros((n, m))

    str_res = ''

    for i in range(n):
        for j in range(m):
            str_res += f'p(X_i = {char_arr[j]} | X_i+1 = {char_arr[i]}) = p({char_arr[j]}) * p({char_arr[i]}|{char_arr[j]}) = '
            str_res += f'{p[j]:.3f}  *  {transition_matrix[i][j]} = '
            joint_matrix[i][j] = round(transition_matrix[i][j] * p[j], 3)
            str_res += f'{joint_matrix[i][j]:.3f}\n'

    return str_res, joint_matrix

