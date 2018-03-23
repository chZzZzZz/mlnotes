# coding=utf-8
import sys
import os
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import pyplot as plt
def read_lines(file_path: object) -> object:
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print ('file not exist: ' + file_path)
        return None
ecs_infor_array = read_lines(".\data_2015_1.txt")#读取训练数据到数组中

#读取升序排列的时间和flavor
ecs_user_array = read_lines(".\data_user.txt")
print(ecs_user_array)
flavor_basic = []
time_basic = []
for item in ecs_user_array:
    values = item.split('\t')
    values[1] = values[1].rstrip('\n')
    time_basic.append(values[0])
    if (len(values[1])>5):
        flavor_basic.append(values[1])
print(flavor_basic)
print(time_basic)


uuid=[]
flavorName=[]
createTime=[]
for item in ecs_infor_array:
        values = item.split("\t")
        uuid.append(values[0])
        flavorName.append(values[1])
        createTime.append(values[2])
date = []
for each in createTime:
    split_list = each.split(' ')
    date.append(split_list[0])

#print(ecs_infor_array)
print(flavorName)#虚拟机
#print(createTime)
print(date)#只包含年月日的日期

#计算每天各种型号的虚拟机申请总数
day = 0#时间 百位十位个位分别代表年月日
sumdate = []#每个日期出现的次数
sumday = [0,]
flavor = [[] for a in range(15)]

for j in range(0, len(time_basic)):
    num = 0  # 某个具体日期出现次数
    for each in date:
        if each == time_basic[j]:
            num += 1
            day += 1
    sumdate.append(num)
    sumday.append(day)

i=0
while(i<15):
    for d in range(0, len(sumday)-1):
        f=0
        for k in range(sumday[d], sumday[d+1]):
            if flavorName[k] == flavor_basic[i]:
                f += 1
        flavor[i].append(f)
    i += 1

print(sumday)
print(sumdate)
print(flavor)

#使用sklearn 线性回归预测
x_train = [[]for i in range(7)]
#x_train = []
x_test = [[7],[8],[9],[10],[11],[12],[13]]
#x_test = [31,32,33,34,35,36,37]
for i in range(0,7):
    x_train[i].append(1)
    x_train[i].append(i)




#def trans(m):
    #return list(map(list,zip(*m)))

y = flavor
y_seven = []
for each in y:
     y_seven.append(each[-7:len(each)])

#y_ = trans(y)
#model = svm.SVR(C=5)
#model = linear_model.LinearRegression()
#model = KNeighborsRegressor(weights='distance')
#model.fit(x_train, y[7][-7:len(y[7])])
#result = model.predict(x_test)
#result = model.predict(x_test)
# for i in range(len(result)):
#     if result[i] > 0.5:
#         result[i] = 1
#     else:
#         result[i] = 0
# print(result)
# print(model.score(x_train,y[7][-7:len(y[7])]))
# print(y[7])
print(x_train)
print(y_seven)

'''线性回归算法实现'''
theta = [1, 1]
def computeCost(x_train,y,theta):
    h = []
    J = []
    m = 7
    for u in range(len(y)):
        for v in range(7):
            J_sum = 0
            pt = [a*b for a, b in zip(x_train[v], theta)]#在python2中需要变化
            h.append(pt[0]+pt[1])
            J_sum += pow((h[v]-y[u][v]), 2)/2
        J.append(J_sum/m)
    return J

J_history = []#15台虚拟机每个虚拟机的J
J_history = computeCost(x_train,y,theta)
print(J_history)

def gradientDecent(x_train,y,theta,alpha,num_iters):
    sumtemp = [[]for i in range(len(y))]#存储每次迭代计算的theta
    cha = []
    m = 7
    for u in range(len(y)):
        for i in range(num_iters):
            h = []
            t1 = []
            t_ = []
            for v in range(7):
                pt = [a * b for a, b in zip(x_train[v], theta)]  # 在python2中需要变化
                h.append(pt[0]+pt[1])  # h为7*1矩阵=x*theta
            cha.append([a - b for a, b in zip(h, y[u])])  # 15个h-y的7*1矩阵
            for vv in range(7):
                t = [a * cha[u][vv] for a in x_train[vv]]  # t list (h-y)*x 如[1,2]
                t_.append(t)  # t_=[[]*7]
            a = t_[0][0] + t_[1][0] + t_[2][0] + t_[3][0] + t_[4][0] + t_[5][0] + t_[6][0]
            b = t_[0][1] + t_[1][1] + t_[2][1] + t_[3][1] + t_[4][1] + t_[5][1] + t_[6][1]
            t1.append(a)
            t1.append(b)
            t1 = [a * (alpha / m) for a in t1]  # t1 = [0.9,0.8]
            theta_ = [a - b for a, b in [theta, t1]]
            sumtemp[u].append(theta_)
            theta = sumtemp[u][i]
    return sumtemp

theta = gradientDecent(x_train, y_seven, theta, 0.01, 200)
print(theta[0][199])






















