import os
import NSGAII
import numpy as np
import random, copy, time
from func_timeout import func_set_timeout
from multiprocessing import Pool, Manager

# time budget
time_budget = 3200
# parallel (fully LS)
class paraTabu():

    def __init__(self, f, range, dir, stime, earlystop, max=25, EnsemblePool_Size=5, ESSelected_Size=3, ULPopSize=10):

        self.max = max
        self.upper = []
        self.lower = []
        for k, v in range.items():
            self.lower.append(v[0])
            self.upper.append(v[1])

        self.dir = dir
        self.objFunc = f
        self.stime = stime
        self.earlystop = earlystop
        self.ULPopSize = ULPopSize

        # tabuList
        self.tabuList = Manager().dict()
        self.processList = Manager().list()

        # EnsemblePool
        self.EnsemblePool = Manager().dict()
        self.EnsemblePool_Size = EnsemblePool_Size
        self.ESSelected_Size = ESSelected_Size
        self.Lock = Manager().Lock()


    def checkduplicate(self, x):

        """ Check for duplicate rows in x """

        seen = set()
        for i, row in enumerate(x):
            row_tuple = tuple(row)
            if row_tuple in seen:
                # Generate a new row if this row is a duplicate
                new_row = [random.randint(self.lower[i], self.upper[i]) for i in range(len(self.lower))]
                while tuple(new_row) in seen:
                    new_row = [random.randint(self.lower[i], self.upper[i]) for i in range(len(self.lower))]
                x[i] = new_row
                seen.add(tuple(new_row))
            else:
                seen.add(row_tuple)
        return x


    def Initialization(self):

        """ Initialization """
        archive = dict()

        # Generate the random matrix x
        x = [[random.randint(self.lower[i], self.upper[i]) for i in range(len(self.lower))]
             for j in range(self.ULPopSize)]
        x = self.checkduplicate(x)

        with open(self.dir, 'a+') as fdoc:
            fdoc.write('*******Initialization*******\n')

        for i in range(self.ULPopSize):

            archive[tuple(x[i])] = []
            print(x[i])

            [mul_res, best, MulObj_AUC_F1] = self.objFunc(x[i])

            with open(self.dir, 'a+') as fdoc:
                fdoc.write('location: {}\n'.format(x[i]))
                fdoc.write('MulObj: {}\n'.format(mul_res))
                fdoc.write('MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC: {}\n'.format(MulObj_AUC_F1))
                fdoc.write('Parameter: {}\n'.format(best))

            archive[tuple(x[i])] = [mul_res, best, MulObj_AUC_F1]

        self.tabuList.update(zip(map(tuple, x), [list(archive.values())[i] for i in range(len(archive))]))
        self.EnsemblePool.update(self.UpdateEnsemblePool(self.tabuList, self.EnsemblePool))

        tabu = dict(self.tabuList)
        EnsemblePool = dict(self.EnsemblePool)

        return


    def sbx_crossover(self, parent1, parent2):

        pc = 1     # Crossover Probability
        eta_c = 30   # Distribution Index of SBX

        # SBX Crossover
        child1, child2 = np.copy(parent1), np.copy(parent2)

        if np.random.rand() <= pc:
            for i in range(len(parent1)):
                if np.random.rand() <= 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:  # Avoid division by zero
                        mean = 0.5 * (parent1[i] + parent2[i])
                        spread = abs(parent1[i] - parent2[i])
                        beta = 1.0 + (2.0 * (min(parent1[i], parent2[i]) - mean) / spread)
                        alpha = 2.0 - beta ** -(eta_c + 1)
                        rand = np.random.rand()
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta_c + 1))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                        child1[i] = 0.5 * ((1 + beta_q) * parent1[i] + (1 - beta_q) * parent2[i])
                        child2[i] = 0.5 * ((1 - beta_q) * parent1[i] + (1 + beta_q) * parent2[i])

        return child1, child2


    def polynomial_mutation(self, child, N):

        pm = 1.0 / N  # Mutation Probability
        eta_m = 20    # Distribution Index of Polynomial Mutation

        # Polynomial Mutation
        for i in range(len(child)):
            if np.random.rand() < pm:
                delta = 1.0 - (child[i])
                if delta > 0:
                    mut_pow = 1.0 / (eta_m + 1.0)
                    delta_q = (1.0 - np.random.rand() ** mut_pow) ** (1.0 / mut_pow) - 1.0
                    child[i] += delta_q * delta

        return child


    def offspring(self, tabuList):

        x_select_population = []

        for _ in range(self.ULPopSize):
            local_dict = dict(tabuList)
            keys = list(local_dict.keys())

            if np.random.uniform(0, 1) > 1 / len(self.upper):
                # Sorting according to AUC value
                sorted_keys = sorted(local_dict, key=lambda k: local_dict[k][0][0])
                min_two_keys_arrays = [np.array(key) for key in sorted_keys[:2]]

                # SBX and Polynomial mutation
                child1, child2 = self.sbx_crossover(min_two_keys_arrays[0], min_two_keys_arrays[1])
                mutated_child1 = self.polynomial_mutation(child1, len(min_two_keys_arrays[0]))
                mutated_child2 = self.polynomial_mutation(child2, len(min_two_keys_arrays[0]))

                # Select one from the two generated offspring, and convert it to a discrete value
                x_select = np.round(mutated_child1 if np.random.rand() < 0.5 else mutated_child2).astype(int)
            else:
                x_select = np.random.randint(self.lower, self.upper + np.ones_like(self.upper))


            x_select = np.where(x_select > self.upper, self.upper, x_select)
            x_select = np.where(x_select < self.lower, self.lower, x_select)
            x_select_population.append(x_select.tolist())


        return x_select_population


    def UpdateEnsemblePool(self, new_dict, ens_dict):

        """ Update Ensemble Pool According to AUC value """

        if len(new_dict) != 0:
            AUC = [list(new_dict.values())[i][0][0] for i in range(len(new_dict))]
            keys = list(new_dict.keys())

            if len(ens_dict) != 0:
                Ensemble_keys = list(ens_dict.keys())

                for i in range(len(keys)):
                    if keys[i] in Ensemble_keys:
                        index_AUC = ens_dict[keys[i]][0][0]
                        if AUC[i] < index_AUC:
                            ens_dict[keys[i]] = new_dict[keys[i]]

                    else:
                        if new_dict[keys[i]][0][0] != 0:
                            ens_dict[keys[i]] = new_dict[keys[i]]

                sorted_pool = sorted(ens_dict.items(), key=lambda x: x[1][0][0])
                ens_dict = dict(sorted_pool[:self.EnsemblePool_Size])

            else:
                non_zero_items = {key: value for key, value in new_dict.items() if value[0][0] != 0}
                sorted_items = sorted(non_zero_items.items(), key=lambda x: x[1][0][0])
                ens_dict = dict(sorted_items[:self.EnsemblePool_Size])

        return ens_dict


    def tabuSearch(self, tabuList, EnsemblePool, processList, sn):

        # In order to release resources correctly, we record the ID of process.
        processList.append(os.getpid())
        ULGen = 0

        """ main loop """
        while 1:
            """ termination """
            if ULGen >= self.max * self.ULPopSize:
                print('stop!!!')
                break

            """ Crossover and Mutation """
            x_select_population = self.offspring(tabuList)
            x_select_population = self.checkduplicate(x_select_population)
            x_select = x_select_population[sn]


            """ find neighbors """
            neighbor = dict()
            tmp = copy.deepcopy(x_select)
            neighbor[str(tmp)] = tmp


            for i in range(len(self.lower)):
                if x_select[i] > self.lower[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x_select)
                        tmp[i] -= c
                        if tmp[i] < self.lower[i]:
                            break

                        if tuple(tmp) not in list(tabuList.keys()) and str(tmp) not in neighbor.keys() \
                                and str(x_select) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

                if x_select[i] < self.upper[i]:
                    c = 1
                    while 1:
                        tmp = copy.deepcopy(x_select)
                        tmp[i] += c
                        if tmp[i] > self.upper[i]:
                            break

                        if tuple(tmp) not in list(tabuList.keys()) and str(tmp) not in neighbor.keys() \
                                and str(x_select) != str(tmp):
                            neighbor[str(tmp)] = tmp
                            break
                        else:
                            c += 1
                            continue

            """ choose best from neighbors """
            if len(neighbor) != 0:
                for k, item in neighbor.items():
                    if tuple(item) not in list(tabuList.keys()):

                        print(item)

                        [mul_res, best, MulObj_AUC_F1] = self.objFunc(item)

                        with open(self.dir, 'a+') as fdoc:
                            fdoc.write('location: {}\n'.format(item))
                            fdoc.write('MulObj: {}\n'.format(mul_res))
                            fdoc.write('MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC: {}\n'.format(MulObj_AUC_F1))
                            fdoc.write('Parameter: {}\n'.format(best))

                        if mul_res[0] != 0:
                            with self.Lock:
                                # Check if item is in tabuList and update if new result is better
                                tabu_value = tabuList.get(tuple(item))
                                if tabu_value is not None and mul_res[0] < tabu_value[0][0]:
                                    # Update the tabuList for this item
                                    tabuList[tuple(item)] = [mul_res, best, MulObj_AUC_F1]
                                    EnsemblePool[tuple(item)] = [mul_res, best, MulObj_AUC_F1]
                                elif tabu_value is None:
                                    # Add new item to tabuList
                                    tabuList[tuple(item)] = [mul_res, best, MulObj_AUC_F1]
                                    EnsemblePool[tuple(item)] = [mul_res, best, MulObj_AUC_F1]


                    else:
                        continue


            """ Choose the nondominated solution in x """
            with self.Lock:
                allfunc = np.array(list([row[0] for row in tabuList.values()]))
                col1 = [row[0] for row in allfunc]
                col2 = [row[1] for row in allfunc]

                # Choose the minimum value in allfunc, the first output is the minimum nondominated value
                Nondominated = NSGAII.offspringselect(col1, col2, self.ULPopSize, FixedNum='True')
                select = Nondominated[0]
                new_dict = {list(tabuList.keys())[i]: list(tabuList.values())[i] for i in select}

                sorted_pool = sorted(EnsemblePool.items(), key=lambda x: x[1][0][0])
                ens = dict(sorted_pool[:self.EnsemblePool_Size])


                tabuList.clear()
                tabuList.update(new_dict)

                EnsemblePool.clear()
                EnsemblePool.update(ens)


            Select_x = [list(t) for t in list(tabuList.keys())]
            Select_allfunc, Select_parameter, Select_MulObj_AUC_F1 = zip(*tabuList.values())


            with open(self.dir, 'a+') as fdoc:
                fdoc.write('------------One Upper level Generation---------------\n')
                fdoc.write('location: {}\n'.format(Select_x))
                fdoc.write('MultiObj: {}\n'.format(Select_allfunc))
                fdoc.write('MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC: {}\n'.format(Select_MulObj_AUC_F1))
                fdoc.write('Parameter: {}\n\n'.format(Select_parameter))

            ULGen += 1


    @func_set_timeout(time_budget)
    def run(self):

        self.Initialization()

        p = Pool(self.ULPopSize)
        for sn in range(self.ULPopSize):
            p.apply_async(self.tabuSearch, args=(self.tabuList, self.EnsemblePool,
                                                 self.processList, sn))
        p.close()
        p.join()


        with open(self.dir, 'a+') as f:
            for k, v in self.tabuList.items():
                print(k, file=f)
                print(v, file=f)
            print('time:', time.time() - self.stime, file=f)

        res = np.asarray(list(self.tabuList.values()))
        location = np.asarray(list(self.tabuList.keys()))

        return res, location