import geatpy as ea
import numpy as np
import time
import pandas as pd
import joblib

LAMBDA = 0.5

class GA:
    def __init__(self, model_path, n_dim, lb, ub, goal = 'max', size_pop=50, max_iter=5, prob_cross = 0.9, prob_mut=0.01, precision=1, encoding = 'BG', selectStyle=None, recStyle=None, mutStyle=None, seed=None):
        
        np.random.seed(seed)
        self.predictor = joblib.load(model_path)

        self.dim = n_dim
        self.lb = lb
        self.ub = ub
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.encoding = encoding
        self.selectStyle = selectStyle #采用二元锦标赛
        self.recStyle =recStyle #采用多点交叉
        self.mutStyle =mutStyle #采用二进制染色体的变异算子
        self.pc= prob_cross #交叉概率
        self.pm = prob_mut #变异概率
        self.goal = np.array([-1]) if goal == 'max' else np.array([1]) #表示目标函数是最小化，元素为-1则表示对应的目标函数是最大化

        self.ranges = np.array([lb,ub])
        lbin, ubin = [1] * self.dim, [1] * self.dim # 决策变量下边界, 决策变量上边界
        self.borders=np.array([lbin,ubin])
        self.varTypes = np.array([precision] * self.dim) #决策变量的类型，0表示连续，1表示离散
        self.Field = self.__set_chromosome()
        
        Lind =int(np.sum(self.Field[0, :]))#计算染色体长度
        self.obj_trace = np.zeros((self.max_iter, 2)) #定义目标函数值记录器
        self.var_trace = np.zeros((self.max_iter, Lind)) #染色体记录器，记录历代最优个体的染色

    
    def set_env(self, workloads: list, svcs: list, bottlenecks: list, r: dict):
        if len(bottlenecks) != self.dim:
            raise Exception('the action dim must equal the length of bottlencks')
        self.workloads = workloads
        self.svcs = svcs
        self.bottlenecks = bottlenecks
        self.r = r
        
    
    def __set_chromosome(self):
        '''
            染色体编码设置
        '''
        codes = [1] * self.dim#决策变量的编码方式，两个1表示变量均使用格雷编码
        precisions =[6] * self.dim#决策变量的编码精度，表示解码后能表示的决策变量的精度可达到小数点后6位
        scales = [0] * self.dim #0表示采用算术刻度，1表示采用对数刻度
        #调用函数创建译码矩阵
        Field =ea.crtfld(self.encoding, self.varTypes, self.ranges, self.borders, precisions,codes,scales)
        return Field
    
    def fitness(self, action):
        x = []
        index = 0
        for i in range(len(self.svcs)):
            svc = self.svcs[i]
            if svc in self.bottlenecks:
                x.extend([i, self.workloads[i], action[index]])
                index += 1
            else:
                x.extend([i, self.workloads[i], self.r[svc]])
        x = np.array(x).reshape(1, -1)
        R1 = self.predictor.predict(x).tolist()[0] 
        R2 = (1 - (np.sum(action) / np.sum(self.ub)))
        return [LAMBDA * R1 + (1 - LAMBDA) * R2]


    def __aim(self, Phen):
        return np.apply_along_axis(self.fitness, 1, Phen)


    def evolve(self):
        Chrom = ea.crtpc(self.encoding, self.size_pop, self.Field) #生成种群染色体矩阵
        variable = ea.bs2ri(Chrom, self.Field) #对初始种群进行解码
        ObjV = self.__aim(variable) #计算初始种群个体的目标函数值
        best_ind = np.argmax(ObjV) #计算当代最优个体的序号
        X_best = 0

        n_elites = int(self.size_pop / 2)

        for gen in range(self.max_iter):
            FitnV = ea.ranking(self.goal * ObjV) #根据目标函数大小分配适应度值
            SelCh = Chrom[ea.selecting(self.selectStyle, FitnV, self.size_pop - n_elites), :]#选择
            SelCh = ea.recombin(self.recStyle, SelCh, self.pc) #重组
            SelCh = ea.mutate(self.mutStyle, self.encoding, SelCh, self.pm)#变异
            #把父代精英个体与子代的染色体进行合并，得到新一代种群
            top_k_idx=ObjV.flatten().argsort()[::-1][0:n_elites]
            Chrom = np.vstack([Chrom[top_k_idx, :], SelCh])
            Phen = ea.bs2ri(Chrom, self.Field)#对种群进行解码(二进制转十进制)

            ObjV = self.__aim(Phen)#求种群个体的目标函数值
            #记录
            best_ind = np.argmax(ObjV)#计算当代最优个体的序号
            self.obj_trace[gen,0]=np.sum(ObjV)/ObjV.shape[0]#记录当代种群的目标函数均值
            X_best = ObjV[best_ind] if ObjV[best_ind] > X_best else X_best
            # obj_trace[gen,1]=ObjV[best_ind]#记录当代种群最优个体目标函数值
            self.obj_trace[gen,1]=X_best
            self.var_trace[gen,:]=Chrom[best_ind,:]#记录当代种群最优个体的染色体
            pass
        #进化完成
        ea.trcplot(self.obj_trace, [['average fitness of population','max fitness of population']])

        res = []
        best_gen = np.argmax(self.obj_trace[:, [1]])
        # print('最优解的目标函数值：', self.obj_trace[best_gen, 1])
        variable = ea.bs2ri(self.var_trace[[best_gen], :], self.Field)#解码得到表现型（即对应的决策变量值）
        # print('最优解的决策变量值为：')
        for i in range(variable.shape[1]):
            # print('x'+str(i)+'=',variable[0, i])
            res.append(variable[0, i])

        return res


# if __name__ == '__main__':
#     workloads = [0, 0, 0, 0, 0, 2.622222222222222, 4.5777777777777775, 0, 40.02222222222223, 1.2888888888888888, 4.0, 0, 0, 2.2666666666666666, 0, 0, 5.022222222222222, 2.2444444444444445, 0.0, 0, 0, 3.2666666666666666, 4.955555555555555, 0, 0, 0, 1.8444444444444443, 0, 0, 76.37777777777777, 3.977777777777778, 0, 51.911111111111104, 0, 40.3111111111111, 7.822222222222222, 0, 13.177777777777775, 10.755555555555556, 42.66666666666666, 0.0, 0, 0]
#     svcs = ['ts-admin-basic-info-service', 'ts-admin-order-service', 'ts-admin-route-service', 'ts-admin-travel-service', 'ts-admin-user-service', 'ts-assurance-service', 'ts-auth-service', 'ts-avatar-service', 'ts-basic-service', 'ts-cancel-service', 'ts-config-service', 'ts-consign-price-service', 'ts-consign-service', 'ts-contacts-service', 'ts-delivery-service', 'ts-execute-service', 'ts-food-map-service', 'ts-food-service', 'ts-inside-payment-service', 'ts-news-service', 'ts-notification-service', 'ts-order-other-service', 'ts-order-service', 'ts-payment-service', 'ts-preserve-other-service', 'ts-preserve-service', 'ts-price-service', 'ts-rebook-service', 'ts-route-plan-service', 'ts-route-service', 'ts-seat-service', 'ts-security-service', 'ts-station-service', 'ts-ticket-office-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-plan-service', 'ts-travel-service', 'ts-travel2-service', 'ts-ui-dashboard', 'ts-user-service', 'ts-verification-code-service', 'ts-voucher-service']
#     bottlenecks = ['ts-auth-service', 'ts-ui-dashboard']
#     dim = len(bottlenecks)
#     max_num, min_num = 15, 1
#     ub, lb = [max_num] * dim, [min_num] * dim
#     r = dict.fromkeys(svcs, 1)

#     opter = GA('/home/ubuntu/xsy/experiment/autoscaling/simulation/train_ticket/RandomForestClassify.model', dim, lb, ub, 'max', size_pop=50, max_iter=5, prob_cross=0.9, prob_mut=0.01, precision=1, encoding='BG', selectStyle='tour', recStyle='xovdp', mutStyle='mutbin', seed=1)
#     opter.set_env(workloads, svcs, bottlenecks, r)

#     opter.evolve()

#     print(opter.fitness([3,1]))
