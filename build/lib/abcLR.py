
class ABC:
    def __init__(self, inputX, target, FVS, P, limit, lb, ub, MR, L2, parallelType):
        self.comp = parallelType
        self.X = inputX
        self.y = target
        self.P = P  # P is population size
        self.limit = limit
        # The number of parameters to be optimized (FVS: Feature Vector Size)
        self.D = FVS + 1
        self.lb = lb  # lower bound for parameters
        self.ub = ub  # upper bound for parameters
        self.MR = MR  # modification rate
        self.L2 = L2
        self.evaluationNumber = 0
        self.tmpID = [-1] * self.P
        self.Foods = self.lb + \
            self.comp.random.rand(self.P, self.D) * (self.ub - self.lb)
        # self.Foods = self.comp.random.uniform(self.lb, self.ub, size = (self.P, self.D))
        self.solution = self.comp.copy(self.Foods)
        self.f = self.calculateF(self.Foods)
        self.fitness = 1 / (1 + self.f)
        self.trial = self.comp.zeros(P)
        self.globalMin = self.f[0, 0]
        self.globalParams = self.comp.copy(self.Foods[0:1])  # 1st row
        self.scoutBeeCounts = 0

    def create_new(self, index):
        new_sol = self.lb + \
            self.comp.random.rand(1, self.D) * (self.ub - self.lb)
        # new_sol = self.comp.random.uniform(self.lb, self.ub, size = (1, self.D))
        self.Foods[index, :] = new_sol.flatten()
        self.solution[index, :] = self.comp.copy(new_sol.flatten())
        self.f[index] = self.calculateF(new_sol)[0]
        self.fitness[index] = 1 / (1 + self.f[index])
        self.trial[index] = 0
        self.scoutBeeCounts += 1

    def memorizeBestSource(self):
        index = self.comp.argmin(self.f)
        if self.f[index, 0] < self.globalMin:
            self.globalMin = self.f[index, 0]
            self.globalParams = self.comp.copy(self.Foods[index: index + 1])

    def calculateProbabilities(self):
        maxfit = self.comp.max(self.fitness)
        self.prob = (0.9 / maxfit * self.fitness) + 0.1

    def sendEmployedBees(self):
        for i in range(self.P):  # for each clone
            ar = self.comp.random.rand(self.D)
            param2change = self.comp.where(ar < self.MR)[0]

            neighbour = self.comp.random.randint(0, self.P)
            while neighbour == i:
                neighbour = self.comp.random.randint(0, self.P)

            self.solution[i, :] = self.comp.copy(self.Foods[i, :])

            # random number generation between -1 and 1 values
            r = -1 + (1 + 1) * self.comp.random.rand()
            self.solution[i, param2change] = self.Foods[i, param2change] + r * (
                self.Foods[i, param2change] - self.Foods[neighbour, param2change])  # self.comp.copy ?
            self.solution[i, param2change] = self.comp.where(
                self.solution[i, param2change] < self.lb, self.lb, self.solution[i, param2change])
            self.solution[i, param2change] = self.comp.where(
                self.solution[i, param2change] > self.ub, self.ub, self.solution[i, param2change])

    def sendOnLookerBees(self):
        i = 0
        t = 0
        while t < self.P:
            if self.comp.random.rand() < self.prob[i, 0]:
                ar = self.comp.random.rand(self.D)
                param2change = self.comp.where(ar < self.MR)[0]

                neighbour = self.comp.random.randint(self.P)
                while neighbour == i:
                    neighbour = self.comp.random.randint(self.P)

                self.solution[t, :] = self.comp.copy(self.Foods[i, :])
                # v_{ij} = x_{ij} + phi_{ij}*(x_{kj}-x_{ij})
                # random number generation between -1 and 1 values
                r = -1 + (1 + 1) * self.comp.random.rand()
                self.solution[t, param2change] = self.Foods[i, param2change] + r * (
                    self.Foods[i, param2change] - self.Foods[neighbour, param2change])  # self.comp.copy ?
                self.tmpID[t] = i

                self.solution[t, param2change] = self.comp.where(
                    self.solution[t, param2change] < self.lb, self.lb, self.solution[t, param2change])
                self.solution[t, param2change] = self.comp.where(
                    self.solution[t, param2change] > self.ub, self.ub, self.solution[t, param2change])
                t += 1
            i += 1
            if i >= self.P:
                i = 0

    def sendScoutBees(self):
        index = self.comp.argmax(self.trial)
        if self.trial[index] >= self.limit:
            self.create_new(index)

    def calculateF(self, foods):
        a1 = self.comp.append(self.comp.ones(
            (self.X.shape[0], 1)), self.X, axis=1)
        z2 = self.comp.dot(a1, foods.T)
        a2 = self.sig(z2)
        # f = self.comp.sum((a2 - self.y) ** 2, axis=0,
        #                   keepdims=True).T  # Transpose ??
        L2reg = self.L2 * self.comp.mean(foods**2, axis=1, keepdims=True)
        f = self.comp.mean((a2 - self.y) ** 2, axis=0, keepdims=True).T + L2reg
        # f = -1 * (self.comp.dot(self.y.T, self.comp.log(a2)) + self.comp.dot((1-self.y).T, self.comp.log(1-a2))).T + 0.001 * self.comp.sum(foods**2, axis=1, keepdims=True);
        self.evaluationNumber += len(f)
        # print(f"Eval Num: {self.evaluationNumber}")
        return f

    def sig(self, n):  # Sigmoid function
        return 1 / (1 + self.comp.exp(-n))


class LearnABC:
    def __init__(self, inputX, target, FVS, P, limit, lb, ub, MR, L2, parallelType, evaluationNumber):
        self.comp = parallelType
        self.abc = ABC(inputX, target, FVS, P, limit,
                       lb, ub, MR, L2, parallelType)
        self.total_numberof_evaluation = evaluationNumber

    def learn(self):
        self.f_values = []
        self.f_values.append(self.comp.min(self.abc.f))
        self.abc.memorizeBestSource()

        # sayac = 0
        while self.abc.evaluationNumber <= self.total_numberof_evaluation:
            self.abc.sendEmployedBees()
            objValSol = self.abc.calculateF(self.abc.solution)
            fitnessSol = 1 / (1 + objValSol)
            # a greedy selection is applied between the current solution i and its mutant
            # If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i

            ind = self.comp.where(fitnessSol > self.abc.fitness)[0]
            ind2 = self.comp.where(fitnessSol <= self.abc.fitness)[0]
            self.abc.trial[ind] = 0

            self.abc.Foods[ind, :] = self.abc.solution[ind, :]
            self.abc.f[ind] = objValSol[ind]
            self.abc.fitness[ind] = fitnessSol[ind]
            # if the solution i can not be improved, increase its trial counter
            self.abc.trial[ind2] += 1

            self.abc.calculateProbabilities()
            self.abc.sendOnLookerBees()

            objValSol = self.abc.calculateF(self.abc.solution)
            fitnessSol = 1 / (1 + objValSol)

            for i in range(self.abc.P):
                t = self.abc.tmpID[i]
                if fitnessSol[i] > self.abc.fitness[t]:
                    self.abc.trial[t] = 0
                    self.abc.Foods[t, :] = self.abc.solution[i, :]
                    self.abc.f[t] = objValSol[i]
                    self.abc.fitness[t] = fitnessSol[i]
                else:
                    self.abc.trial[t] += 1

            self.abc.memorizeBestSource()
            self.abc.sendScoutBees()

            self.f_values.append(self.comp.min(self.abc.f))
            # sayac += 1;
            # if sayac % 5000 == 0: print(f"Sayaç = {sayac}")

        self.net = self.abc.globalParams
        self.globalMin = self.abc.globalMin
        # print(f"Evaluation Number: {self.abc.evaluationNumber}")
        print(f"The number of scout bees: {self.abc.scoutBeeCounts}")


class ABC_LR_Model():
    def __init__(self, lb=-32, ub=32, evaluationNumber=60000, limit=50, P=40, MR=0.1, L2=0, parallelType=None):
        '''
        lb is lower bound for parameters to be learned
        ub is upper bound for parameters to be learned
        limit determines whether a scout bee can be created. If a solution cannot be improved up to the limit number, a scout bee is created instead of the solution.
        '''
        self.lb = lb
        self.ub = ub
        self.evaluationNumber = evaluationNumber
        self.limit = limit
        self.P = P
        self.MR = MR
        self.L2 = L2
        self.parallelType = parallelType

    def fit(self, trainX, trainY):
        self.FVS = trainX.shape[1]
        learn = LearnABC(trainX, trainY, self.FVS, self.P, self.limit, self.lb,
                         self.ub, self.MR, self.L2, self.parallelType, self.evaluationNumber)
        learn.learn()
        self.net = learn.net

    def logsig(self, x):
        return 1 / (1 + self.parallelType.exp(-x))

    def score(self, X, y):
        prediction = self.logsig(self.parallelType.dot(self.parallelType.append(
            self.parallelType.ones((X.shape[0], 1)), X, axis=1), self.net.T))
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        res = self.parallelType.average(y == prediction)
        return res
