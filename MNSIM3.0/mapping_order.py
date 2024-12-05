from IPython import embed
import torch
import sys
import os
import math
import configparser as cp
import numpy as np

def generate_normal_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(row):
        for j in range(column):
            matrix[i][j] = start
            pos[start][0]=i
            pos[start][1]=j
            start += 1
    return matrix,pos

def generate_snake_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(row):
        for j in range(column):
            if i % 2:
                matrix[i][column - j - 1] = start
                pos[start][0]=i
                pos[start][1]=column - j - 1
            else:
                matrix[i][j] = start
                pos[start][0]=i
                pos[start][1]=j
            start += 1
    return matrix,pos

def generate_zigzag_matrix(row, column):
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(row * column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                if j < column - 1:
                    j += 1
                    matrix[i][j] = start
                else:
                    i += 1
                    matrix[i][j] = start
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                matrix[i][j] = start
                step += 1
                if i == row - 1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < row - 1:
                    i += 1
                    matrix[i][j] = start
                else:
                    j += 1
                    matrix[i][j] = start
                state = 3
            elif state == 3:
                j += 1
                i -= 1
                matrix[i][j] = start
                step += 1
                if j == column - 1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        pos[start][0]=i
        pos[start][1]=j
        start += 1
    return matrix,pos

def generate_normal_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_normal_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(int(row/c)):
        for j in range(int(column/c)):
            for m in range(c):
                for n in range(c):
                    matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                    pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                    pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
            start += 1
    return matrix,pos

def generate_snake_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_snake_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    start = 0
    for i in range(int(row/c)):
        for j in range(int(column/c)):
            for m in range(c):
                for n in range(c):
                    if i % 2:
                        matrix[i*c+m][(int(column/c) - j - 1)*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=(int(column/c) - j - 1)*c+n
                    else:
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
            start += 1
    return matrix,pos

def generate_zigzag_matrix_cmesh(row, column, c=2):
    matrix_min,pos_min=generate_zigzag_matrix(c,c)
    matrix = np.zeros([row, column])
    pos=np.zeros([row*column,2])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(int(row/c) * int(column/c)):
        if x == 0:
            for m in range(c):
                for n in range(c):
                    matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                    pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                    pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
        else:
            if state == 0:
                if j < int(column/c) - 1:
                    j += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                else:
                    i += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                step += 1
                if i == int(row/c) - 1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < int(row/c) - 1:
                    i += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                else:
                    j += 1
                    for m in range(c):
                        for n in range(c):
                            matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                            pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                            pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                state = 3
            elif state == 3:
                j += 1
                i -= 1
                for m in range(c):
                    for n in range(c):
                        matrix[i*c+m][j*c+n] = start*c**2+int(matrix_min[m][n])
                        pos[start*c**2+int(matrix_min[m][n])][0]=i*c+m
                        pos[start*c**2+int(matrix_min[m][n])][1]=j*c+n
                step += 1
                if j == int(column/c) - 1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        start += 1
    return matrix,pos

matrix, pos = generate_normal_matrix_cmesh(8,8,2)
print(pos)
#print(pos)