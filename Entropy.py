import copy
import math
import itertools
import matplotlib.pyplot as plt
from scipy.stats import *
import math

O = {'A': 119.0, 'B': 66.0, 'C': 720.0} # Sum of transportation origin data
D = {'A': 112.0, 'B': 51.0, 'C': 742.0} # Sum of transportation destination data

fee = {('A', 'A'): 3.0, ('A', 'B'): 4.0, ('A', 'C'): 9.0,
       ('B', 'A'): 4.0, ('B', 'B'): 3.0, ('B', 'C'): 6.0,
       ('C', 'A'): 9.0, ('C', 'B'): 6.0, ('C', 'C'): 3.0}

data = {('A', 'A'): 50.0, ('A', 'B'): 20.0, ('A', 'C'): 100.0,
       ('B', 'A'): 20.0, ('B', 'B'): 7.0, ('B', 'C'): 6.0,
       ('C', 'A'): 1.0, ('C', 'B'): 8.0, ('C', 'C'): 1000.0}

budget = {('A', 'A'): 7.0, ('A', 'B'): 9.0, ('A', 'C'): 2.0,
          ('B', 'A'): 9.0, ('B', 'B'): 10.0, ('B', 'C'): 4.0,
          ('C', 'A'): 2.0, ('C', 'B'): 4.0, ('C', 'C'): 20.0}
B = 34

class Entropy: # Entropy maximization model
    def __init__(self, O, D, fee, data):
        self.O = O
        self.D = D
        self.fee = fee
        self.beta = 0.948
        self.data = data

    def calPQ(self): # Calculate coefficients
        count = 0
        P = {}
        Q = {}
        for i in self.O.keys():
            Q[i] = 1
        while count < 100:
            for i in self.O.keys():
                divider = 0
                for j in self.D.keys():
                    divider += Q[j] * self.D[j] * math.exp(- self.beta * self.fee[(i, j)])
                P[i] = 1.0 / divider
            for j in self.D.keys():
                divider = 0
                for i in self.O.keys():
                    divider += P[i] * self.O[i] * math.exp(- self.beta * self.fee[(i, j)])
                Q[j] = 1.0 / divider
            count += 1
        return P, Q

    def calT(self): # Calculate transportation
        T = {}
        P, Q = self.calPQ()
        for i in self.O.keys():
            for j in self.D.keys():
                T[(i, j)] = P[i] * Q[j] * self.O[i] * self.D[j] * math.exp(- self.beta * self.fee[(i, j)])
        return T

    def normalize(self, data):
        for i in data.keys():
            list = []
            divider = sum(data[i])
            for j in data[i]:
                list.append(j / divider)
            data[i] = list
        return data

    def poisson(self, update, count):
        T = self.calT()
        distribution_x = {}
        distribution_y = {}
        for i in T.keys():
            upper_limit = min(self.O[i[0]], self.D[i[1]])
            distribution_x[i] = list(range(0, int(upper_limit) + 1))
        if update == False:
            for i in distribution_x.keys():
                probability_list = []
                for j in distribution_x[i]:
                    probability = distributions.poisson.pmf(j, T[i])
                    probability_list.append(probability)
                distribution_y[i] = probability_list
        else:
            for i in distribution_x.keys():
                probability_list = []
                for j in distribution_x[i]:
                    probability = distributions.poisson.pmf(j, T[i]) * distributions.poisson.pmf(self.data[i], j)**count
                    probability_list.append(probability)
                distribution_y[i] = probability_list
        distribution_y = self.normalize(distribution_y)
        for i in itertools.combinations(distribution_y.keys(), 2):
            self.comparison(distribution_y, i[0], i[1])
        return distribution_x, distribution_y

    def comparison(self, data, first, second): # data is usually distribution_y, first is the first object to compare, second is the second object to compare
        first_index = len(data[first])
        second_index = len(data[second])
        first_bigger = 0
        second_bigger = 0
        tie = 0
        s_i = 0
        for f_i in range(0, first_index):
            for s_i in range(0, second_index):
                if  f_i > s_i:
                    first_bigger += data[first][f_i] * data[second][s_i]
                elif f_i < s_i:
                    second_bigger += data[first][f_i] * data[second][s_i]
                else:
                    tie += data[first][f_i] * data[second][s_i]
                s_i += 1
            f_i += 1

        print ('%s: %f, %s: %f, tie: %f'%(first, first_bigger, second, second_bigger, tie))

class Graph:
    def __init__(self, O, D, fee, data, update, count=1):
        self.O = O
        self.D = D
        self.fee = fee
        self.data = data
        self.update = update
        self.count = count

    def makeGraph(self):
        entropy = Entropy(self.O, self.D, self.fee, self.data)
        distribution_x, distribution_y = entropy.poisson(self.update, self.count)

        fig = plt.figure()

        ax1 = fig.add_subplot(3, 3, 1)
        ax2 = fig.add_subplot(3, 3, 2)
        ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 3, 4)
        ax5 = fig.add_subplot(3, 3, 5)
        ax6 = fig.add_subplot(3, 3, 6)
        ax7 = fig.add_subplot(3, 3, 7)
        ax8 = fig.add_subplot(3, 3, 8)
        ax9 = fig.add_subplot(3, 3, 9)

        ax1.bar(distribution_x[('A', 'A')], distribution_y[('A', 'A')])
        ax2.bar(distribution_x[('A', 'B')], distribution_y[('A', 'B')])
        ax3.bar(distribution_x[('A', 'C')], distribution_y[('A', 'C')])
        ax4.bar(distribution_x[('B', 'A')], distribution_y[('B', 'A')])
        ax5.bar(distribution_x[('B', 'B')], distribution_y[('B', 'B')])
        ax6.bar(distribution_x[('B', 'C')], distribution_y[('B', 'C')])
        ax7.bar(distribution_x[('C', 'A')], distribution_y[('C', 'A')])
        ax8.bar(distribution_x[('C', 'B')], distribution_y[('C', 'B')])
        ax9.bar(distribution_x[('C', 'C')], distribution_y[('C', 'C')])
        ax7.set_xlim([0, 20])
        ax9.set_xlim([700, 750])
        plt.show()

def freq(distribution_x, distribution_y):
    result = {}
    for k, v in distribution_y.items():
        result[k] = distribution_x[k][v.index(max(v))]
    return result

def knapsack(i, B, budget, trans):
    if i < 0 or B <= 0:
        return 0, [], 0
    if budget[i] > B:
        value, prev, b = knapsack(i - 1, B, budget, trans)
        curr = copy.deepcopy(prev)
        return value, curr, b
    else: # w[i] <= W
        left, prev_left, b_left = knapsack(i - 1, B, budget, trans)
        right, prev_right, b_right = knapsack(i - 1, B - budget[i], budget, trans)
        if left >= trans[i] + right:
            curr = copy.deepcopy(prev_left)
            return left, curr, b_left
        else:
            curr = copy.deepcopy(prev_right)
            curr.append(i)
            return trans[i] + right, curr, b_right + budget[i]

def sum_distribution(distribution_x1, distribution_y1, distribution_x2, distribution_y2):
    result_x = [x for x in range(0, len(distribution_y1) + len(distribution_y2))]
    result_y = [0.0 for x in range(0, len(distribution_y1) + len(distribution_y2))]
    for idx1 in range(len(distribution_x1)):
        for idx2 in range(len(distribution_x2)):
            x1 = distribution_x1[idx1]
            x2 = distribution_x2[idx2]
            sum = round(x1 + x2)
            result_y[sum] += distribution_y1[idx1] * distribution_y2[idx2]
    return result_x, result_y

def draw_graph(distribution_x, distribution_y, lower_lim, upper_lim):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(distribution_x, distribution_y)
    ax.set_xlim([lower_lim, upper_lim])
    plt.show()

if __name__ == '__main__':
    entropy = Entropy(O, D, fee, data)
    print(entropy.calT())
    graph = Graph(O, D, fee, data, True)
    graph.makeGraph()

    distribution_x, distribution_y = entropy.poisson(True, 1)
    frequent_values = freq(distribution_x, distribution_y)

    frequent_values_list = list(frequent_values.values())
    count = len(frequent_values) - 1
    budget_list = list(budget.values())

    trans_sum, trans_list, trans_budget = knapsack(count, B, budget_list, frequent_values_list)
    selected_zone_list = list(map(lambda i: list(budget.keys())[i], trans_list))

    distribution_sum_x = distribution_x[selected_zone_list[0]]
    distribution_sum_y = distribution_y[selected_zone_list[0]]
    for i in range(1, len(selected_zone_list)):
        distribution_sum_x, distribution_sum_y = sum_distribution(distribution_sum_x, distribution_sum_y, distribution_x[selected_zone_list[i]], distribution_y[selected_zone_list[i]])
    print(trans_budget)
    print(trans_sum)
    print(selected_zone_list)
    draw_graph(distribution_sum_x, distribution_sum_y, 800, 860)




