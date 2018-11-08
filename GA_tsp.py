# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:14:45 2018

@author: 24681

tsp bays29
"""
import numpy as np
import matplotlib.pyplot as plt
#输入29个城市的距离矩阵
dis_matrix=np.array([[0,107,241,190,124,80,316,76,152,157,283,133,113,297,228,129,348,276,188,150,65,341,184,67,221,169,108,45,167],
                     [107,0,148,137,88,127,336,183,134,95,254,180,101,234,175,176,265,199,182,67,42,278,271,146,251,105,191,139,79],
                     [241,148,0,374,171,259,509,317,217,232,491,312,280,391,412,349,422,356,355,204,182,435,417,292,424,116,337,273,77],
                     [190,137,374,0,202,234,222,192,248,42,117,287,79,107,38,121,152,86,68,70,137,151,239,135,137,242,165,228,205],
                     [124,88,171,202,0,61,392,202,46,160,319,112,163,322,240,232,314,287,238,155,65,366,300,175,307,57,220,121,97],
                     [80,127,259,234,61,0,386,141,72,167,351,55,157,331,272,226,362,296,232,164,85,375,249,147,301,118,188,60,185],
                     [316,336,509,222,392,386,0,233,438,254,202,439,235,254,210,187,313,266,154,282,321,298,168,249,95,437,190,314,435],
                     [76,183,317,192,202,141,233,0,213,188,272,193,131,302,233,98,344,289,177,216,141,346,108,57,190,245,43,81,243],
                     [152,134,217,248,46,72,438,213,0,206,365,89,209,368,286,278,360,333,284,201,111,412,321,221,353,72,266,132,111],
                     [157,95,232,42,160,167,254,188,206,0,159,220,57,149,80,132,193,127,100,28,95,193,241,131,169,200,161,189,163],
                     [283,254,491,117,319,351,202,272,365,159,0,404,176,106,79,161,165,141,95,187,254,103,279,215,117,359,216,308,322],
                     [133,180,312,287,112,55,439,193,89,220,404,0,210,384,325,279,415,349,285,217,138,428,310,200,354,169,241,112,238],
                     [113,101,280,79,163,157,235,131,209,57,176,210,0,186,117,75,231,165,81,85,92,230,184,74,150,208,104,158,206],
                     [297,234,391,107,322,331,254,302,368,149,106,384,186,0,69,191,59,35,125,167,255,44,309,245,169,327,246,335,288],
                     [228,175,412,38,240,272,210,233,286,80,79,325,117,69,0,122,122,56,56,108,175,113,240,176,125,280,177,266,243],
                     [129,176,349,121,232,226,187,98,278,132,161,279,75,191,122,0,244,178,66,160,161,235,118,62,92,277,55,155,275],
                     [348,265,422,152,314,362,313,344,360,193,165,415,231,59,122,244,0,66,178,198,286,77,362,287,228,358,299,380,319],
                     [276,199,356,86,287,296,266,289,333,127,141,349,165,35,56,178,66,0,112,132,220,79,296,232,181,292,233,314,253],
                     [188,182,355,68,238,232,154,177,284,100,95,285,81,125,56,66,178,112,0,128,167,169,179,120,69,283,121,213,281],
                     [150,67,204,70,155,164,282,216,201,28,187,217,85,167,108,160,198,132,128,0,88,211,269,159,197,172,189,182,135],
                     [65,42,182,137,65,85,321,141,111,95,254,138,92,255,175,161,286,220,167,88,0,299,229,104,236,110,149,97,108],
                     [341,278,435,151,366,375,298,346,412,193,103,428,230,44,113,235,77,79,169,211,299,0,353,289,213,371,290,379,332],
                     [184,271,417,239,300,249,168,108,321,241,279,310,184,309,240,118,362,296,179,269,229,353,0,121,162,345,80,189,342],
                     [67,146,292,135,175,147,249,57,221,131,215,200,74,245,176,62,287,232,120,159,104,289,121,0,154,220,41,93,218],
                     [221,251,424,137,307,301,95,190,353,169,117,354,150,169,125,92,228,181,69,197,236,213,162,154,0,352,147,247,350],
                     [169,105,116,242,57,118,437,245,72,200,359,169,208,327,280,277,358,292,283,172,110,371,345,220,352,0,265,178,39],
                     [108,191,337,165,220,188,190,43,266,161,216,241,104,246,177,55,299,233,121,189,149,290,80,41,147,265,0,124,263],
                     [45,139,273,228,121,60,314,81,132,189,308,112,158,335,266,155,380,314,213,182,97,379,189,93,247,178,124,0,199],
                     [167,79,77,205,97,185,435,243,111,163,322,238,206,288,243,275,319,253,281,135,108,332,342,218,350,39,263,199,0]])

class GA(object):
    def __init__(self,popsize,length,crossprob,mutationprob,dis_matrix):
        self.popsize=popsize
        self.length=length
        self.crossprob=crossprob
        self.mutationprob=mutationprob
        self.dis_matrix=dis_matrix
        
    #初始化种群，生成一个种群数量*染色体长度的np二维矩阵 ,其值为0或1  
    def pop_init(self):
        pop=np.zeros((self.popsize,self.length))
        for i in range(self.popsize):
            arr=np.arange(self.length,dtype=int)
            np.random.shuffle(arr)
            pop[i]=arr
        

        pop=pop.astype(int)
 
            
        return pop
    
    #交叉操作
    def crossover(self,pop):
        
        for i in range(pop.shape[0]):
            randnum=np.random.rand(1)
            if randnum<self.crossprob:
                crosspoint=np.random.randint(1,pop.shape[1],size=1)#随机生成一个交叉点
                crosspoint=crosspoint[0]
                crosschrom=np.random.randint(pop.shape[0],size=1)#随机指明要和它交叉的染色体
                crosschrom=crosschrom[0]
                
                #交叉操作
                a=pop[i,0:crosspoint]
                b=pop[crosschrom,crosspoint:-1]
                newchrom1=np.hstack(( pop[i,0:crosspoint],pop[crosschrom,crosspoint:]))
                newchrom2=np.hstack(( pop[crosschrom,0:crosspoint],pop[i,crosspoint:]))
                pop[i]=newchrom1
                pop[crosschrom]=newchrom2
                
                
                
                #由于是tsp问题，应该把染色体中的重复节点采用PMX映射
                
               
                    
                for j in range(crosspoint,self.length):
                    while pop[i,j] in pop[i,0:crosspoint]: 
                        for k in range(crosspoint):
                            
                            if pop[i,j]==pop[i,k]:
                                pop[i,j]=pop[crosschrom,k] 

                    
                    
                            
                            
               
                for j in range(crosspoint,self.length):
                    while pop[crosschrom,j] in pop[crosschrom,0:crosspoint]: 
                        for k in range(crosspoint):
                           if pop[crosschrom,j]==pop[crosschrom,k]:
                            pop[crosschrom,j]=pop[i,k]
                    
                
        return pop
    
    #变异操作
    def mutation(self,pop):
        for i in range(self.popsize):
            for j in range(self.length):
                random_num=np.random.rand(1)[0]
                if random_num<self.mutationprob:
                    mutation_gene=np.random.randint(self.length,size=1)[0]
                    buffer=pop[i,j]
                    pop[i,j]=pop[i,mutation_gene]
                    pop[i,mutation_gene]=buffer
                    
        
        return pop
    
    #选择操作,采用轮盘赌
    def selection(self,pop):
        
        #计算每条染色体的距离
        dist=np.zeros(self.popsize)
        for i in range(dist.shape[0]):
            for j in range(self.length):
                if j==self.length-1:
                    dist[i]=dist[i]+self.dis_matrix[pop[i,j],pop[i,0]]
                else:
                    dist[i]=dist[i]+self.dis_matrix[pop[i,j],pop[i,j+1]]
                
                
                
        
        #对于解码出的每个个体求适应度
        fitness=np.zeros(self.popsize)
        
        fitness=1/dist
        
        #储存最优值
        meanfitness=np.mean(fitness)
        bestfitness=np.max(fitness)
        best_y=1/bestfitness
        best_x=pop[np.argmax(fitness)]
        
        
        #通过轮盘赌进行选择
        
        #通过累加得到适应度表
        fitnesslist=fitness
        for i in range(1,self.popsize):
            fitnesslist[i]=fitnesslist[i]+fitnesslist[i-1]
            
            
        newpop=np.zeros_like(pop)   
        #通过适应度表进行选择操作
        for i in range(self.popsize):
            randnum=np.random.rand(1)*fitnesslist[-1]
            for j in range(self.popsize):
                if randnum<=fitnesslist[j]:
                    newpop[i]=pop[j]
                    break
        return newpop,bestfitness,meanfitness,best_y,best_x
                
if __name__=='__main__':
    popsize=500 #种群大小
    length=29 #染色体长度
    crossprob=0.6 #交叉概率
    mutationprob=0.001 #变异概率
    ga=GA(popsize,length,crossprob,mutationprob,dis_matrix)
    init_pop=ga.pop_init()

    iteration=1000
    Bestfitness=np.zeros(iteration)
    Meanfitness=np.zeros(iteration)
    Best_y=np.zeros(iteration)
    Best_x=np.zeros((iteration,length))
    
    for i in range(iteration):
     cross_pop=ga.crossover(init_pop)

     mut_pop=ga.mutation(cross_pop)

     init_pop,Bestfitness[i],Meanfitness[i],Best_y[i],Best_x[i]=ga.selection(mut_pop)
 
     print('第%d代'%(i))
     
    #绘图
    x=np.arange(iteration)
    plt.figure()
    plt.plot(x,Bestfitness)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('bestfitness')
    plt.show
    
    plt.figure()
    plt.plot(x,Meanfitness)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('meanfitness')
    plt.show
    
    plt.figure()
    plt.plot(x,Best_y)
    plt.xlabel('generation')
    plt.ylabel('y')
    plt.title('shortest_dis')
    plt.show
     
    the_best_one=np.argmax(Bestfitness)
    the_best_y=Best_y[the_best_one]
    the_best_x=Best_x[the_best_one]
    print('最短距离为 %f' % the_best_y)
    the_best_x=the_best_x.astype(int)
    print(the_best_x)
    
    #绘制路线图
    city_x=np.array([1150,	630	,40,	750	,750,	1030	,1650	,1490	,790,	710,	840,	1170,	970	,510	,750	,1280,	230	,460	,1040	,590	,830	,490	,1840,1260,	1280,	490,	1460,	1260	,360
])
    city_y=np.array([1760,	1660,	2090,	1100,	2030,	2070	,650,	1630	,2260	,1310	,550,	2300,	1340,	700,	900	,1200,	590	,860,	950,	1390,	1770,	500	,1240	,1500,	790	,2130	,1420	,1910	,1980
])

    for i in[0,250,500,750]:
        plt.figure()
        best_x=Best_x[i]
        best_x=best_x.astype(int)
        plt.plot(city_x[best_x],city_y[best_x],color='red')
        plt.scatter(city_x[best_x],city_y[best_x],color='blue')
        plt.plot(city_x[np.array([best_x[-1],best_x[0]])],city_y[np.array([best_x[-1],best_x[0]])],color='red')
        plt.title('%d generation best solution' % i)
        plt.show
        
    plt.figure()
    plt.plot(city_x[the_best_x],city_y[the_best_x],color='red')
    plt.scatter(city_x[the_best_x],city_y[the_best_x],color='blue')
    plt.plot(city_x[np.array([the_best_x[-1],the_best_x[0]])],city_y[np.array([the_best_x[-1],the_best_x[0]])],color='red')
    plt.title('best solution')
    plt.show