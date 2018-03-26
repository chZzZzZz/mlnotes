import ecs
def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    ecs_lines = ecs.read_lines("E:\sdk-python\src\ecs\TrainData_2015.1.1_2015.2.19.txt")  # 读取训练数据到数组中
    input_lines = ecs.read_lines("E:\sdk-python\src\ecs\input_5flavors_cpu_7days.txt")
    # 把输入数据文件中的各个数据放到对应的变量里
    fuwuqi = input_lines[0]  # 服务器
    m = int(input_lines[2])  # 需要放置的虚拟机个数
    fulist = fuwuqi.split(' ')
    cpu = int(fulist[0])  # 服务器的cpu核数
    mem = int(fulist[1]) * 1024  # 服务器的内存
    xuniji = input_lines[3:m + 3]  # 虚拟机

    input_flavor = []
    input_flavor_heshu = []
    input_flavor_neicun = []
    for each in xuniji:
        xulist = each.split(' ')
        input_flavor.append(xulist[0])
        input_flavor_heshu.append(int(xulist[1]))
        input_flavor_neicun.append(int(xulist[2]))
    input_flavor_cpu_dict = dict(zip(input_flavor, input_flavor_heshu))  # 需要配置的虚拟机和其核数的字典
    input_flavor_mem_dict = dict(zip(input_flavor, input_flavor_neicun))  # 需要配置的虚拟机和其内存的字典
    # input_flavor_dict(map(lambda x,y:[x,y],input_flavor,input_flavor_num))
    # input_flavor_dict(zip(input_flavor,input_flavor_num))


    # 把txt文件中的数据读入到数组中
    uuid = []
    flavorName = []
    createTime = []
    for item in ecs_lines:
        values = item.split("\t")
        uuid.append(values[0])
        flavorName.append(values[1])
        createTime.append(values[2])
    date = []
    for each in createTime:
        split_list = each.split(' ')
        date.append(split_list[0])

    # 删除数组中的重复数据
    def remove_duplicates(ori):
        new = []  # 新建一个列表，以防止原来的列表被损坏
        for i in ori:  # 历遍原来的列表
            if i not in new:  # 如果这个元素不在新表内，则加入
                new.append(i)
        return new

    time_basic = remove_duplicates(date)
    flavor_basic = ['flavor1', 'flavor2', 'flavor3', 'flavor4', 'flavor5', 'flavor6', 'flavor7', 'flavor8', 'flavor9',
                    'flavor10', 'flavor11', 'flavor12', 'flavor13', 'flavor14', 'flavor15']

    # 计算每天各种型号的虚拟机申请总数
    day = 0  # 时间 百位十位个位分别代表年月日
    sumdate = []  # 每个日期出现的次数
    sumday = [0, ]
    flavor = [[] for a in range(15)]

    for j in range(0, len(time_basic)):
        num = 0  # 某个具体日期出现次数
        for each in date:
            if each == time_basic[j]:
                num += 1
                day += 1
        sumdate.append(num)
        sumday.append(day)

    i = 0
    while (i < 15):
        for d in range(0, len(sumday) - 1):
            f = 0
            for k in range(sumday[d], sumday[d + 1]):
                if flavorName[k] == flavor_basic[i]:
                    f += 1
            flavor[i].append(f)
        i += 1

    # 使用sklearn 线性回归预测
    x_train = [[] for i in range(7)]
    # x_train = []
    x_test = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7]]
    # x_test = [31,32,33,34,35,36,37]
    for i in range(0, 7):
        x_train[i].append(1)
        x_train[i].append(i)




        # def trans(m):
        # return list(map(list,zip(*m)))

    y = flavor
    y_seven = []
    for each in y:
        y_seven.append(each[-7:len(each)])


    '''线性回归算法实现'''
    theta = [1, 1]

    def computeCost(x_train, y, theta):
        h = []
        J = []
        m = 7
        for u in range(len(y)):
            for v in range(7):
                J_sum = 0
                pt = [a * b for a, b in zip(x_train[v], theta)]  # 在python2中需要变化
                h.append(pt[0] + pt[1])
                J_sum += pow((h[v] - y[u][v]), 2) / 2
            J.append(J_sum / m)
        return J

    J_history = []  # 15台虚拟机每个虚拟机的J
    J_history = computeCost(x_train, y, theta)

    def gradientDecent(x_train, y, theta, alpha, num_iters):
        sumtemp = [[] for i in range(len(y))]  # 存储每次迭代计算的theta
        cha = []
        m = 7
        for u in range(len(y)):

            for i in range(num_iters):
                h = []
                t1 = []
                t_ = []
                for v in range(7):
                    pt = [a * b for a, b in zip(x_train[v], theta)]  # 在python2中需要变化
                    h.append(pt[0] + pt[1])  # h为7*1矩阵=x*theta
                cha.append([a - b for a, b in zip(h, y[u])])  # 15个h-y的7*1矩阵
                for vv in range(7):
                    t = [a * cha[u * num_iters + i][vv] for a in x_train[vv]]  # t list (h-y)*x 如[1,2]
                    t_.append(t)  # t_=[[]*7]
                a = t_[0][0] + t_[1][0] + t_[2][0] + t_[3][0] + t_[4][0] + t_[5][0] + t_[6][0]
                b = t_[0][1] + t_[1][1] + t_[2][1] + t_[3][1] + t_[4][1] + t_[5][1] + t_[6][1]
                t1.append(a)
                t1.append(b)
                t2 = [a * (alpha / m) for a in t1]  # t1 = [0.9,0.8]
                theta_ = [a - b for a, b in zip(theta, t2)]
                sumtemp[u].append(theta_)
                theta = sumtemp[u][i]
        return sumtemp

    theta = gradientDecent(x_train, y_seven, theta, 0.01, 200)


    f = []  # flavor各天的申请量
    for j in range(15):
        sum = 0
        for i in range(7):
            f_list = [a * b for a, b in zip(x_test[i], theta[j][199])]
            f_add = f_list[0] + f_list[1]
            if f_add < 0.5:
                f_add = 0
            else:
                f_add = round(f_add)
            sum += f_add
        f.append(sum)
    dict_result = {'flavor1': f[0],
                   'flavor2': f[1],
                   'flavor3': f[2],
                   'flavor4': f[3],
                   'flavor5': f[4],
                   'flavor6': f[5],
                   'flavor7': f[6],
                   'flavor8': f[7],
                   'flavor9': f[8],
                   'flavor10': f[9],
                   'flavor11': f[10],
                   'flavor12': f[11],
                   'flavor13': f[12],
                   'flavor14': f[13],
                   'flavor15': f[14]}


    '分配虚拟机 背包问题'
    # 把待分配虚拟机的名称,核数,内存,个数存到一个列表里
    k1 = input_flavor_cpu_dict.keys()
    k2 = input_flavor_mem_dict.keys()
    k3 = dict_result.keys()
    comment = k1 & k2 & k3
    comment_list = list(comment)  # 待分配的虚拟机名称列表
    list_fenpei = [[] for i in range(len(comment_list))]
    sumflavor = 0  # 预测的虚拟机总数
    for i in range(len(comment_list)):
        list_fenpei[i].append(comment_list[i])
        list_fenpei[i].append(input_flavor_cpu_dict[comment_list[i]])
        list_fenpei[i].append(input_flavor_mem_dict[comment_list[i]])
        list_fenpei[i].append(dict_result[comment_list[i]])
        sumflavor += list_fenpei[i][3]


    # 背包算法实现
    def fit(list_fenpei, cpu, mem):
        tai = 1
        ncpu = cpu
        nmem = mem
        h = []
        for i in range(len(list_fenpei)):
            geshu = list_fenpei[i][3]
            heshu = list_fenpei[i][1]
            neicun = list_fenpei[i][2]
            mingzi = list_fenpei[i][0]
            for j in range(geshu):
                ncpu = ncpu - heshu
                nmem = nmem - neicun
                if (ncpu >= 0 & nmem >= 0):
                    h.append(mingzi)
                else:
                    h.append('\n')
                    tai = tai + 1
                    ncpu = cpu
                    nmem = mem
        return h, tai

    jieguo, tai = fit(list_fenpei, cpu, mem)
    strjieguo = " ".join(jieguo)
    splice = strjieguo.split('\n')
    daifenpei = []
    for each in splice:
        split_list = each.split(' ')
        daifenpei.append(split_list)

    dict_feipei_result = [{} for i in range(len(splice))]
    for i in range(tai):
        for item in daifenpei[i]:
            if len(item) > 2:
                if item not in dict_feipei_result[i]:
                    dict_feipei_result[i][item] = 1
                else:
                    dict_feipei_result[i][item] += 1

    result = []
    dict_result_zhen = {key: value for key, value in dict_result.items() if key in comment}
    # result添加预测结果
    result.append(str(sumflavor) + '\n')
    for key, value in dict_result_zhen.items():
        result.append("%s %s\n" % (key, value))  # 字典转换为列表
    result.append('\n')  # 这里的'\n'是空行
    result.append(str(tai) + '\n')  # 这里的'\n'是换行
    # result添加分配结果
    for i in range(tai):
        result.append(str(i + 1) + ' ')
        for key, value in dict_feipei_result[i].items():
            result.append("%s %s " % (key, value))
        result.append('\n')

    if ecs_lines is None:
        print('ecs information is none')
        return result
    if input_lines is None:
        print('input file information is none')
        return result
    return result
