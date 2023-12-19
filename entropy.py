import math

import numpy as np


def binary_entropy(p, char_arr):
    n = p.shape[0]
    str_res = 'H(X_i) = -('
    str_num = ''
    H_X = 0
    for i in range(n):
        str_res += f'p({char_arr[i]}) * log(p({char_arr[i]}))'
        tmp = np.round(math.log2(p[i]), 3)
        H_X += np.round(p[i] * tmp, 3)
        str_num += f'{p[i]:.3f} * ({tmp:.3f})'
        if i < n - 1:
            str_res += ' + '
            str_num += ' + '
    H_X *= (-1)
    H_X = np.round(H_X, 3)
    str_res += ') = -(' + str_num + ') = ' + str(H_X) + '\n'
    return str_res, H_X


def calc_joint_entropy(joint_matrix, char_arr):
    n, m = joint_matrix.shape
    str_res = 'H(X_i X_i+1) = -('
    str_num = ''
    H_joint = 0
    for i in range(n):
        for j in range(m):
            str_res += f'p(X_i = {char_arr[i]} | X_i+1 = {char_arr[j]}) * log(p(X_i = {char_arr[i]} | X_i+1 = {char_arr[j]}))'
            tmp = round(math.log2(joint_matrix[i][j]), 3)
            H_joint += joint_matrix[i][j] * tmp
            str_num += str(joint_matrix[i][j]) + ' * (' + str(tmp) + ')'
            if j < m - 1:
                str_res += ' + '
                str_num += ' + '
        if i < n - 1:
            str_res += ' + '
            str_num += ' + '
    H_joint *= (-1)
    H_joint = round(H_joint, 3)
    str_res += ') = -(' + str_num + ') = ' + str(H_joint) + '\n'
    return str_res, H_joint


def full_conditional_entropy(H_joint, H_X):
    H_full_cond = H_joint - H_X
    str_res = f'H_X_i(X_i+1) = H(X_i X_i+1) - H(X_i) = {H_joint} - {H_X} = {H_full_cond}'
    return str_res, H_full_cond


def conditional_entropy(p, transition_matrix):
    n = p.shape[0]
    str_res = 'H(X_i) = -('
    str_num1 = ''
    str_num2 = ''
    H_cond = 0
    for j in range(n):
        str_num1 += f'{p[j]:.3f} * ('
        H_cond_tmp = 0
        for k in range(n):
            str_num1 += f'{transition_matrix[j][k]} * log({transition_matrix[j][k]})'
            tmp = math.log2(transition_matrix[j][k])
            H_cond_tmp += transition_matrix[j][k] * tmp
            #str_num2 += f'{H_cond_tmp}'
            if k < n - 1:
                str_num1 += ' + '

        str_num1 += ')'
        str_num2 += f'{p[j]:.3f} * {H_cond_tmp:.3f}'
        H_cond += p[j] * H_cond_tmp
        if j < n - 1:
            str_num1 += ' + '
            str_num2 += ' + '
    H_cond *= (-1)
    str_res += str_num1 + ') = -(' + str_num2 + ') = ' f'{H_cond:.3f}\n'
    return str_res, H_cond
