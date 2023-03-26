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
        self.selectStyle = selectStyle # binary tournament
        self.recStyle =recStyle # two-point crossover operator
        self.mutStyle =mutStyle # binary-chromosome mutation operator
        self.pc= prob_cross # crossover probability
        self.pm = prob_mut # mutation probability

        # 1 means to minimize the target function
        # -1 means to maximize objective function
        self.goal = np.array([-1]) if goal == 'max' else np.array([1])

        self.ranges = np.array([lb,ub])
        lbin, ubin = [1] * self.dim, [1] * self.dim # Lower and upper boundary of the decision variable
        self.borders=np.array([lbin,ubin])
        self.varTypes = np.array([precision] * self.dim) # type of the decision variable，[0 means continuous and 1 means discrete]
        self.Field = self.__set_chromosome()
        
        Lind =int(np.sum(self.Field[0, :])) # Calculate chromosome length
        self.obj_trace = np.zeros((self.max_iter, 2)) # record the value of object function  
        self.var_trace = np.zeros((self.max_iter, Lind)) # record the best individuals in history

    
    def set_env(self, workloads: list, svcs: list, bottlenecks: list, r: dict):
        if len(bottlenecks) != self.dim:
            raise Exception('the action dim must equal the length of bottlencks')
        self.workloads = workloads
        self.svcs = svcs
        self.bottlenecks = bottlenecks
        self.r = r
        
    
    def __set_chromosome(self):
        '''
            set the configuration of chromosome
        '''
        codes = [1] * self.dim # Gray code
        precisions =[6] * self.dim # Encoding precision of decision variables
        scales = [0] * self.dim # arithmetic scale

        # decoding matrix
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
        # Generate the matrix for chromosome population
        Chrom = ea.crtpc(self.encoding, self.size_pop, self.Field)
        # Decode the initial population
        variable = ea.bs2ri(Chrom, self.Field)
        # Calculate the value of objective function for the initial population individual
        ObjV = self.__aim(variable)
        # Record the index of the best individuals
        best_ind = np.argmax(ObjV)
        X_best = 0

        n_elites = int(self.size_pop / 2)

        for gen in range(self.max_iter):
            # Assign fitness values according to the value of the objective function
            FitnV = ea.ranking(self.goal * ObjV)
            # Select
            SelCh = Chrom[ea.selecting(self.selectStyle, FitnV, self.size_pop - n_elites), :]
            # Recombine
            SelCh = ea.recombin(self.recStyle, SelCh, self.pc)
            # Mutate
            SelCh = ea.mutate(self.mutStyle, self.encoding, SelCh, self.pm)
            # A new generation population is obtained by merging the chromosomes of the elite parent and offspring
            top_k_idx=ObjV.flatten().argsort()[::-1][0:n_elites]
            Chrom = np.vstack([Chrom[top_k_idx, :], SelCh])
            # Decode the population (binary to decimal)
            Phen = ea.bs2ri(Chrom, self.Field)

            # Calculate objective function value for population
            ObjV = self.__aim(Phen)
            # Record the index of the best individuals
            best_ind = np.argmax(ObjV)
            # Record the mean of the objective function value for the population
            self.obj_trace[gen,0]=np.sum(ObjV)/ObjV.shape[0]
            X_best = ObjV[best_ind] if ObjV[best_ind] > X_best else X_best
            self.obj_trace[gen,1]=X_best
            self.var_trace[gen,:]=Chrom[best_ind,:]
            pass
        # Finish
        ea.trcplot(self.obj_trace, [['average fitness of population','max fitness of population']])

        res = []
        best_gen = np.argmax(self.obj_trace[:, [1]])
        # print('The value of the optimal solution is：', self.obj_trace[best_gen, 1])
        variable = ea.bs2ri(self.var_trace[[best_gen], :], self.Field) # decode
        # print('The decision variable value of the optimal solution are')
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
