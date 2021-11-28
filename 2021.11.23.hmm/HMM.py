# 导入包
from enum import Flag
import random
import argparse
import codecs
import os
import re
from tkinter.constants import NO
from typing import KeysView
import numpy as np
from numpy.core.fromnumeric import argmax, size
from numpy.lib.function_base import select
from numpy.random.mtrand import gamma
import pandas as pd
import copy
# observations
np.printoptions(precision=5)


class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


def load_observations(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    if len(lines) % 2 == 1:  # remove extra lines
        lines = lines[:len(lines)-1]
    return [Observation(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]


class HMM:

    @property
    def A(self):
        # 处理pi和矩阵a，a就是状态转换矩阵
        A = pd.DataFrame(index=[x for x in self.states], columns=[
                         x for x in self.states])  # 创建一个空的dataframe
        for state_i in self.states:
            for state_j in self.states:
                A.loc[state_i, state_j ] = float(self.transitions[state_i][state_j])
        return A

    @property
    def PI(self):
        trans_dic = copy.deepcopy(self.transitions)
        _pi = trans_dic['#']
        _pi = pd.DataFrame(_pi, index=['p']).T
        return _pi

    @property
    def B(self):
        states = self.states
        # 获取所有的观测值
        outputs = self.outputs
        for state_j in states:
            for output in self.emissions[state_j].keys():
                outputs.add(output)
        # 获取所有状态到观测值的概率，没有则写为0
        p = []
        for state in states:
            temp = []
            for output in outputs:
                temp.append(float(self.emissions[state].get(output,0)))
            p.append(temp)
        B = np.array(p)
        B = pd.DataFrame(B,index=states, columns=list(outputs))

        return B

    def __init__(self, transitions=None, emissions=None):
        """creates a model from transition and emission probabilities"""
        self.transitions = transitions
        self.emissions = emissions
        if self.emissions:
            self.states = self.emissions.keys()

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities if given.
        Initializes probabilities randomly if unspecified."""
        # TODO: fill in for section a
        trans_file_name = str(basename) + '.trans'
        emit_file_name = str(basename) + '.emit'
        # 下面处理状态转移矩阵
        trans_dic = {}
        states = set()
        trans_dic_need_init = False
        with open(trans_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                line_after = re.sub(r' +', ' ', line)
                line_after = line_after.split(' ')  # 按空格分割后的数组
                tran_01 = line_after
                if len(tran_01) == 2: # 如果处理后长度没有3，则认为没有概率,并且需要初始化概率，注意是整个初始化
                    trans_dic_need_init = True
                    tran_01.append(0)
                if tran_01[0] not in trans_dic.keys():
                    trans_dic[tran_01[0]] = {}
                trans_dic[tran_01[0]][tran_01[1]] = float(tran_01[2])
                states.add(tran_01[0])
                states.add(tran_01[1])
        states.remove('#')
        # 需要初始化概率
        if trans_dic_need_init:
            for state_i in trans_dic.keys():
                num = len(trans_dic[state_i])
                p = np.random.randint(1000,size=(num))
                p = p/p.sum()
                p_gen  = (x for x in p)
                for state_j in trans_dic[state_i].keys():
                    trans_dic[state_i][state_j] = next(p_gen)
        # 将缺失的状态转移，补足为0
        # 补足缺失的pi及以state作为i状态的字典
        for state in states:
            trans_dic['#'][state] = trans_dic['#'].get(state,0) #若没有，则赋值为0
            if state not in trans_dic.keys():
                trans_dic[state] = {}
        # 补足缺失的aij
        for state_i in states:
            for state_j in states:
                trans_dic[state_i][state_j] = trans_dic[state_i].get(state_j,0)
        
        self.transitions = trans_dic
        self.states = trans_dic['#'].keys()
        
        # 下面处理发射矩阵
        emit_dic = {}
        outputs = set()
        emit_dic_need_init =False
        with open(emit_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                line_after = re.sub(r' +', ' ', line)
                line_after = line_after.split(' ')  # 按空格分割后的数组
                p_01 = line_after
                if len(p_01) == 2: # 如果处理后长度没有3，则认为没有概率,并且需要初始化概率，注意是整个初始化
                    emit_dic_need_init = True
                    p_01.append(0)
                if p_01[0] not in emit_dic.keys():
                    emit_dic[p_01[0]] = {}
                emit_dic[p_01[0]][p_01[1]] = float(p_01[2])
                # outputs.add(p_01[0])
                #print(line_after)
                outputs.add(p_01[1])
        # 需要初始化概率
        if emit_dic_need_init:
            for state_j in emit_dic.keys():
                num = len(emit_dic[state_j])
                p = np.random.randint(1000,size=(num))
                p = p/p.sum()
                p_gen  = (x for x in p)
                for output in emit_dic[state_j].keys():
                    trans_dic[state_j][output] = next(p_gen)
        
        # 下面补足
        for state_j in states:
            if state_j not in emit_dic.keys():
                emit_dic[state_j] = {} #这里会出现若是某个状态到观测值的概率全部没有的情况下，都被初始化为0了
            for output in outputs:
                emit_dic[state_j][output] = emit_dic[state_j].get(output,0)
        
        self.emissions = emit_dic
        self.outputs = outputs
        print('load success!')

    def dump(self, basename):
        """store HMM model parameters in basename.trans and basename.emit"""
        # TODO: fill in for section a
        trans_file_name = str(basename)+'.trans'
        emit_file_name = str(basename)+'.emit'
        with open(trans_file_name, 'w', encoding='utf-8') as f:
            for keys, items in self.transitions.items():  # {a:{b:f,c:f}}
                for key, item in items.items():
                    if item != 0:
                        f.writelines(keys+' '+key+' '+str(item)+'\n')

        with open(emit_file_name, 'w', encoding='utf-8') as f:
            for keys, items in self.emissions.items():  # {a:{b:f,c:f}}
                for key, item in items.items():
                    if item != 0:
                        f.writelines(keys+' '+key+' '+str(item)+'\n')
        print('dump success!')

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM.
        """
        # TODO: fill in for section c
        # 先利用A矩阵随机出N个状态
        # 第一步，随机π_i
        # 获取π的概率
        # _pi_p = []
        # _pi_state = []
        # for state,p in self.transitions['#'].items():
        #     _pi_state.append(state)
        #     _pi_p.append(float(p))
        # π_i_state = np.random.choice(a=_pi_state,size=1,replace=False,p=_pi_p)[0]

        state_seq = []
        # a_i_state = π_i_state
        # state_seq.append(a_i_state)
        a_i_state = '#'
        for i in range(1, n+1):
            aj_state = [] # 存储aj的状态
            aj_p = [] # 存储aj的概率
            for state, p in self.transitions[a_i_state].items():
                aj_state.append(state) # 获取状态
                aj_p.append(float(p))  # 获取概率
            a_i_state = np.random.choice(
                a=aj_state, size=1, replace=True, p=aj_p)[0]
            state_seq.append(a_i_state)

        # 通过状态链，去随机观测值
        out_seq = []
        for state in state_seq:
            # 获取该状态下的观测值
            out_result = []
            out_p = []
            for out, p in self.emissions[state].items():
                # print(out,p)
                out_result.append(out)
                out_p.append(float(p))
            # print('-----------')
            out_result = np.random.choice(
                a=out_result, size=1, replace=True, p=out_p)[0]
            # print(out_result)
            out_seq.append(out_result)
        # print(state_seq)
        return Observation(state_seq, out_seq)

    def viterbi(self, observation):
        """given an observation,
        set its state sequence to be the most likely state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # TODO: fill in for section d
        # 下面开始初始化delta_1 下，各个状态的概率
        obs = observation.outputseq
        delta_1 = []
        for state,p in self.transitions['#'].items(): # 所有状态的初始概率
            delta_1.append(p*self.emissions[state][obs[0]])

        delta_1 = np.array(delta_1)
        # 下面开始初始化psai_1 下，概率最大的路径节点（全部为0）
        psai_1 = np.zeros(len(self.transitions['#'].keys()))

        delta = []
        psai = []
        delta.append(delta_1)
        psai.append(psai_1)

        # 下面开始迭代
        delta_t_1 = delta_1
        for output in obs[1:]:
            temp_delta_t_1 = [] #用来临时存储，每轮迭代后，delta_i的计算结果,不能直接用delta_t_1存储，否则会改变它的值
            temp_psai_t = [] # psai的值不参与迭代，是直接取的结果
            for state_i in self.states: # 遍历所有的状态，这些状态作为第i个状态
                a_j_i = [] # 用来存储aji,状态为j的情况下，转移到i的概率
                for state_j in self.states: # 遍历所有状态，做为第j个状态
                    a_j_i.append(self.transitions[state_j][state_i])
                b_i_t = self.emissions[state_i][output] # 用来存储状态为i时，观测值为ot的概率
                a_j_i = np.array(a_j_i)
                delta_t_1_i = (delta_t_1*a_j_i).max()*b_i_t
                # 这里是找寻 (delta_t_1*a_j_i).max()中，
                # 先获取索引,索引应该不是唯一的，但是np的函数返回的是唯一的，该问题待优化
                index = (delta_t_1*a_j_i).argmax() # 返回最大的参数的位置，也就是概率最大的路径当中，第t-i个节点，在所有隐状态中的索引
                # pasai_t = [x for x in self.transitions['#'].keys()][index] # 返回所有状态，因为字典没有改变过，所有顺序不会变，转成列表，按索引取值即可
                pasai_t = index  # + 1 # 因为ndarray中的索引是从0开始的，但是我们定义“设定”的是从1开始的
                temp_delta_t_1.append(delta_t_1_i)
                temp_psai_t.append(pasai_t)
            delta_t_1 = np.array(temp_delta_t_1)
            temp_psai_t = np.array(temp_psai_t)

            psai.append(temp_psai_t) # 加入到psai中，便于后续核对结果
            delta.append(delta_t_1) # 加入到delta中，便于后续核对结果
        
        #迭代完成后，终止时求的最大概率和最后一个节点
        max_p = delta[-1].max()
        
        # 将状态赋值给stateseq
        observation.stateseq = []
        states = [x for x in self.transitions['#'].keys()] # 因为字典的key的顺序不会改变，所以返回的结果的顺序也依然不变
        # 选择终点
        # 创建状态索引列表，
        i_s_l = [] # index state list
        index = delta[-1].argmax()
        i_s_l.append(index) # 在索引的时候需要
        # 下面开始前向迭代
        rever_psai = psai[:0:-1] # 列表反转,这样保证是从后往前迭代的，第0个元素是不要的
        for k in rever_psai:
            index = k[index]
            i_s_l.append(index)
        i_s_l = i_s_l[::-1] #再反转，因为迭代是从后往前的，所以反转了就是从前往后的状态
        for i in i_s_l:
            observation.stateseq.append(states[i])
        return delta,psai,max_p,i_s_l

    def forward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the forward algorithm parameters alpha_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        all_alpha = []
        # 第一步，计算初值
        obs = observation.outputseq
        alpha_1 = []
        for state in self.states:
            # print(state)
            # print(obs[0])
            # print(self.emissions[state][obs[0]])
            t = self.transitions['#'][state]*self.emissions[state].get(obs[0], 0)
            alpha_1.append(t)
        alpha_1 = np.array(alpha_1)  # 转为ndarray，做乘法
        all_alpha.append(alpha_1)

        # 第二步，迭代
        alpha_i = alpha_1  # 将初始值赋值给alpha_i，参见统计学习方法
        for t in range(2, len(obs)+1):  # 从第二部迭代到最后一个观测值
            temp_alpha_i = []
            for i_state in self.states:  # 这个循环标志a_ji当中的i状态，参见统计学习方法. HMM
                a_ji = []  # 用它来标志aji，表示j状态，转移到i状态的概率
                for j_state in self.states:  # 这个循环标志a_ji当中的j状态，参见统计学习方法. HMM
                    a_ji.append(float(self.transitions[j_state][i_state]))
                # 将b_ji 转为ndarray属性，便于计算
                a_ji = np.array(a_ji)
                # alpha_i = alpha_i*b_ji
                # 下面找到状态t到观测值的概率
                # print('state : ',i_state,' ',obs[t-1])
                b_i_t = float(self.emissions[i_state].get(obs[t-1], 0))

                temp_alpha_i.append((alpha_i*a_ji).sum()*b_i_t)
            alpha_i = np.array(temp_alpha_i)
            all_alpha.append(alpha_i)
        # 最后一轮alpha的值
        # print(alpha_i)

        # 下面计算概率
        p_o = alpha_i.sum()
        # print(p_o)
        return np.array(all_alpha)
        # for keys,items

    def forward_probability(self, observation):
        """return probability of observation, computed with forward algorithm.
        """
        all_alpha = self.forward(observation)
        alpha = all_alpha[-1]
        return alpha.sum()

    def backward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the backward algorithm parameters beta_i(t)
        for all 0<=t<T and i HMM states.
        """
        all_beta = []
        # 第一步，计算初值,也就是beta_t的初始值
        obs = observation.outputseq
        
        beta_t = []
        for __ in self.states:
            t = 1            # 全部初始化为1
            beta_t.append(t)
        beta_t = np.array(beta_t)  # 转为ndarray，方便做乘法
        all_beta.append(beta_t)

        # 第二步，迭代
        beta_t_plus_1 = beta_t  # 将初始值赋值给βt+1,便于计算βt，参见统计学习方法
        for t in range(2, len(obs)+1):  # 从倒数第二个观测值，迭代到第一个观测值
            temp_beta_t_plus_1 = []
            for i_state in self.states:  # 这个循环标志β_t_i当中的i状态，参见统计学习方法. HMM
                a_ij = []  # 用来标志a_ij表示从i状态，转移到j状态的概率
                b_j_t_plus_1 = []  # 用它来标志b_j_t，表示j状态下，观测值为t+1的概率
                for j_state in self.states:  # 这个循环标志b_ji当中的j状态，参见统计学习方法. HMM
                    #print('i state : ',i_state,' ',self.transitions[i_state][j_state],' j :state ',j_state,' ',self.emissions[j_state][obs[len(obs)-t]],' t :',obs[len(obs)-t], ' t+1: ',obs[len(obs)-t+1])
                    a_ij.append(
                        float(self.transitions[i_state][j_state]))  # 逐个添加概率
                    b_j_t_plus_1.append(
                        float(self.emissions[j_state].get(obs[len(obs)-t+1], 0)))  # 逐个添加概率
                # 将a_ij 转为ndarray属性，便于计算
                a_ij = np.array(a_ij)
                b_j_t_plus_1 = np.array(b_j_t_plus_1)
                temp_beta_t_plus_1.append(
                    (a_ij*b_j_t_plus_1*beta_t_plus_1).sum())
                #print('i_state:',i_state,' ,β is :',(a_ij*b_j_t_plus_1*beta_t_plus_1).sum())
            beta_t_plus_1 = np.array(temp_beta_t_plus_1)
            all_beta.append(beta_t_plus_1)
        
        all_beta = all_beta[::-1]
        return np.array(all_beta)

    def backward_probability(self, observation):
        """return probability of observation, computed with backward algorithm.
        """
        # TODO: fill in for section e
        all_beta = self.backward(observation)
        _pi = []
        _b_i_1 = []
        for state, pro in self.transitions['#'].items():
            # print(state)
            _pi.append(float(pro))
            _b_i_1.append(
                float(self.emissions[state].get(observation.outputseq[0], 0)))
        _pi = np.array(_pi)
        _b_i_1 = np.array(_b_i_1)

        return (_pi*_b_i_1*all_beta[0]).sum()

    def learn_supervised(self, corpus, emitlock=False, translock=False):
        """Given a corpus, which is a list of observations
        with known state sequences,
        set the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        """
        # TODO: fill in for section b
        # 初始化状态转移矩阵及发射矩阵
        transitions_set = set()
        emit_dic = {}
        for obs in corpus:
            # print(obs.stateseq,obs.outputseq)
            # 获取所有的state
            for state, output in zip(obs.stateseq, obs.outputseq):
                transitions_set.add(state)  # 将状态加到set
                if state in emit_dic.keys():  # 判断状态是否在发射矩阵的字典中
                    emit_dic[state][output] = ''  # 如果是，则直接添加对应的观测值
                else:
                    # 如果不在字典中，则先创建对应的键后，将观测值转为字典，作为键值对的值，进行赋值
                    emit_dic[state] = {output: ''}
        transitions_dic = {}
        # 初始概率用#作为父节点
        pi_dic = {}
        for state_i in transitions_set:
            pi_dic[state_i] = ''
        transitions_dic['#'] = pi_dic
        for state_i in transitions_set:
            temp = {}
            for state_j in transitions_set:
                temp[state_j] = ''
            transitions_dic[state_i] = temp

        self.transitions = transitions_dic
        self.emissions = emit_dic

        # 下面开始训练，训练的参数结果将根据设定，来确定是否更新，算法的公式推导参见《统计学习方法第二版》P203 and 204
        # 深拷贝
        num_transitions_dic = copy.deepcopy(transitions_dic)
        num_emittionns_dic = copy.deepcopy(emit_dic)
        # 将字典中的值初始化为0，便于后续累加，计算频数
        for states, items in num_transitions_dic.items():
            for state, item in items.items():
                num_transitions_dic[states][state] = 0
        for states, items in num_emittionns_dic.items():
            for output, item in items.items():
                num_emittionns_dic[states][output] = 0
        # 下面开始计算频数
        for obs in corpus:
            # print(obs.stateseq,obs.outputseq)
            # 获取所有的state
            s_s = obs.stateseq
            o_t = obs.outputseq
            for i in range(len(s_s)-1):
                if i == 0:
                    num_transitions_dic['#'][s_s[i]] += 1
                num_transitions_dic[s_s[i]][s_s[i+1]] += 1
                num_emittionns_dic[s_s[i]][o_t[i]] += 1
            num_emittionns_dic[s_s[-1]][o_t[-1]] += 1
        # 下面开始计算概率
        for states,items in num_emittionns_dic.items():
            total = sum(items.values())
            for output,item in items.items():
                num_emittionns_dic[states][output] = num_emittionns_dic[states][output]/total

        #先处理pi

        # pi = num_transitions_dic.pop('#')# 计算转移概率时，需要将pi踢出去，不然会有除0错误
        for states,items in num_transitions_dic.items():
            total = sum(items.values())
            for state,item in items.items():
                num_transitions_dic[states][state] = num_transitions_dic[states][state]/total
        
        
        # num_transitions_dic['#'] = pi
        #是否改变
        if not emitlock:
            self.emissions = num_emittionns_dic
        if not translock:
            self.transitions = num_transitions_dic


    def learn_unsupervised(self, corpus, convergence=0.001, emitlock=False, translock=False, restarts=0):
        """Given a corpus,
        which is a list of observations with the state sequences unknown,
        apply the Baum Welch EM algorithm
        to learn the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        Stop when the log likelihood changes less than the convergence threhshold,
        and return the final log likelihood.
        If restarts>0, re-run EM with random initializations.
        """
        # TODO: fill in for section f

        train_model = []

        for i in corpus:
            obs = i
            if len(obs.outputseq) < 2 :
                print(obs.outputseq)
                print(obs)
                print('序列长度小于2，不参与训练')
                continue
            
            # 获取α
            alpha = self.forward(obs)
            # 获取β
            beta = self.backward(obs)

            #计算γ
            gama = alpha*beta
            gama = gama/gama.sum(axis=1).reshape(-1,1) # γ[t,i]

            # 下面计算π
            # 深拷贝
            # trans_dic = copy.deepcopy(self.transitions)
            index  = [x for x in range(len(self.transitions['#']))]
            # for state_index,state in zip(index,self.transitions['#'].keys()):
            #     trans_dic['#'][state] = gama[0,state_index]
            
            # 下面开始计算A矩阵
            # 下面 transition这个字典，转换为A矩阵
            A = self.A.values
            B = self.B
            # 下面开始计算不同时刻t(从1到T-1)下的epsilon
            epsilon = [] # ε[t,i,j]
            for i in range(len(obs.outputseq)-1):
                t = alpha[i,].reshape(-1,1)*A*B[obs.outputseq[i]].values.reshape(1,-1)*beta[i+1,].reshape(1,-1)
                print(t)
                epsilon.append(t)
            epsilon = np.array(epsilon)
            # epsilon还要除以对应的时刻下面的所有的加和
            episilon_t_sum = epsilon.sum(axis=(1,2)).reshape(-1,1,1)
            # print('-----------epsilon_t_sum------------')
            # print(episilon_t_sum)
            epsilon = epsilon/episilon_t_sum

            # print('-----------epsilon------------')
            # print(epsilon)
            # print(epsilon.shape)


            # ε[t,i,j]在t方向上求和
            epsilon_sum = epsilon.sum(axis=0)
            # γ[t-1,i]在t方向上求和, 两者做除法可求出矩阵A
            gama_sum_t_1 = gama[:-1].sum(axis=0)
            # print('---------------------episilon sum------------')
            # print(epsilon_sum)
            # print('----------------gama------------')
            # print(gama_sum_t_1)


            # new_A = epsilon_sum/gama_sum_t_1

            # # 计算新的a_ij
            # for state_i_index,state_i in zip(index,self.transitions['#'].keys()):
            #     for state_j_index,state_j in zip(index,self.transitions['#'].keys()):
            #         ## print(epsilon[:,state_i_index,state_j_index].sum()/gama[:-1,state_i_index].sum())
            #         trans_dic[state_i][state_j] = epsilon[:,state_i_index,state_j_index].sum()/gama[:-1,state_i_index].sum()
            
            
            
            # γ在t方向上求和
            # 求和后可作为bjk的分母
            gama_sum_t = gama.sum(axis=0)
            # 计算b_j_t，注意这里计算的是bjt,bjk还要通过求和来实现
            # b_j_t = gama/gama_sum_t
            # 找到重复值,找到在观测序列中，每个观测值的索引

            output_index_dic = {}
            for k,index in zip(obs.outputseq,range(len(obs.outputseq))):
                if k not in output_index_dic:
                    output_index_dic[k] = []
                if k in obs.outputseq:
                    output_index_dic[k].append(index)
            
            emit_dic = copy.deepcopy(self.emissions)
            # 计算b矩阵当中的分子
            newB_num = []
            for state_j,state_index in zip(self.states,range(len(self.states))):
                j_k_p = []
                for output in self.outputs:
                    if output in output_index_dic.keys():
                        index = output_index_dic[output] # 返回每个观测值的索引
                        p = 0
                        for i in index:
                            p += gama[i,state_index]
                        emit_dic[state_j][output] = p 
                    else:
                        emit_dic[state_j][output] = 0
                    j_k_p.append(emit_dic[state_j][output])
                newB_num.append(j_k_p)
            newB_num = np.array(newB_num)
            # # 对于不在本次更新的b_j_k全部重置为0，
            # for output in self.outputs:
            #     if output not in output_index_dic.keys():
            #         trans_dic[]
            # emit_dic = copy.deepcopy(self.emissions)
            # # 计算新的b_j_k
            # for state_j_index,state_j in zip(index,self.transitions['#'].keys()):
            #     for output in self.emissions[state_j].keys():
            #         # 获取那些在obs中，观测值为output的索引
            #         output_index = []
            #         for i in range(len(obs.outputseq)):
            #             if obs.outputseq[i] == output:
            #                 output_index.append(i)
            #         numerator = 0
            #         for i in output_index:
            #             print(i,state_j_index)
            #             numerator += gama[i,state_j_index]
            #         emit_dic[state_j][output] = numerator/gama[:,state_j_index].sum()
            train_model.append([gama,[epsilon_sum,gama_sum_t_1],[newB_num,gama_sum_t]])
            # print(epsilon_sum)
            # print(gama_sum_t_1)
        
        # 下面计算tran_model中的值
        gama_sum  = np.zeros((1,len(self.states)))

        epsilon_sum = np.zeros((len(self.states),len(self.states)))
        gama_sum_t_1 = np.zeros((1,len(self.states)))

        newB_num_sum = np.zeros((len(self.states),len(self.outputs)))
        gama_sum_t = np.zeros((1,len(self.states)))
        for i in range(len(train_model)):
            gama_sum = gama_sum +train_model[i][0][0,]
            # print('episilon in model shape : ',train_model[i][1][0].shape)
            # print('episilon sum in model shape : ',epsilon_sum.shape)
            # print(train_model[i][1][0])
            epsilon_sum = epsilon_sum+train_model[i][1][0]
            gama_sum_t_1 = gama_sum_t_1 +train_model[i][1][1]
            newB_num_sum = newB_num_sum +train_model[i][2][0]
            gama_sum_t = gama_sum_t + train_model[i][2][1]
        
        # print('episilon sum ',epsilon.shape)
        new_PI = gama_sum[0,]/len(corpus)
        new_A = epsilon_sum/gama_sum_t_1
        new_B = newB_num_sum/gama_sum_t.reshape(-1,1)

        # emit_dic = copy.deepcopy(self.emissions)
        # # 计算新的b_j_k
        # for 
        # for state_j_index,state_j in zip(index,self.transitions['#'].keys()):
        #     for output in self.emissions[state_j].keys():
        #         # 获取那些在obs中，观测值为output的索引
        #         output_index = []
        #         for i in range(len(obs.outputseq)):
        #             if obs.outputseq[i] == output:
        #                 output_index.append(i)
        #         numerator = 0
        #         for i in output_index:
        #             print(i,state_j_index)
        #             numerator += gama[i,state_j_index]
        #         emit_dic[state_j][output] = numerator/gama[:,state_j_index].sum()

        return new_PI,new_A,new_B