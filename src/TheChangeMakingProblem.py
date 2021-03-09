'''

AA - Trabalho 1
Avaliacao de Desempenho Computacional de Algoritmos de Resolucao do "The Change-Making Problem" (tCMP)
Autor: Filipe Pires
NMEC: 85122
Data: 11/2019

'''

################################################################################ Required Libraries ##################################################################

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

################################################################################ Data ################################################################################

# simple example:
currency = [1,2,5,10,20,50,100,200]
amount = 187
# evolution study:
currencies = [[1], [1,2], [1,5], [1,2,5], [1,3,8,15,74,129], [1,2,5,10,20,50,100]]
amounts = [x for x in range(0,801)]
# global matrix (for memoization)
currency_used = [[None for x in range(0,amount+1)] for x in range(0,len(currency)+1)] # possible combinations of coins/bills that return the correct change
# study variables
studyN = 1 # number of times the simple study is repeated during the elaborate study (helps removing external factors that influence execution times)
smoothF = 1 # smoothing factor of the results (helps reducing irregularities and the size of the results)
logB = None # logarithmic base to transform execution times when plotting them (used to be able to see recursive side by side with the other algorithms)

################################################################################ Algorithms ##########################################################################

# Recursive Solution
def tCMP_recursive_getMinCurrency(i, currency, currentAmount):
    if currentAmount==0: # if there is no need for change
        return [0,[],0]
    elif i==-1 or currentAmount<0: # if recursion has ended 
        return [float("inf"),[],0] # float("inf") acts as an unbounded upper value for comparison
    else:
        basic_operations_counter = 0
        # process of looking for the best currency to return (and returning it)
        aux = tCMP_recursive_getMinCurrency(i-1, currency, currentAmount)
        alternative = aux[0], aux[1]
        basic_operations_counter += aux[2]

        if currentAmount-currency[i]>=0: # if index to be compared is inside of bounds
            aux = tCMP_recursive_getMinCurrency(i, currency, currentAmount-currency[i])
            best = aux[0], aux[1]
            basic_operations_counter += aux[2]

            basic_operations_counter += 1
            if alternative[0]<1+best[0]: # if alternative is better than the best 
                return [alternative[0], alternative[1], basic_operations_counter]
            return [1+best[0],[currency[i]]+best[1], basic_operations_counter]
        else:
            return [alternative[0], alternative[1], basic_operations_counter]

def tCMP_recursive(currency, amount):
    start = time.time()
    if len(currency)==0:
        print("Error: currency must contain at least 1 type of change to return.")
        return [0,[],0,time.time()-start]

    result = tCMP_recursive_getMinCurrency(len(currency)-1, currency, amount)
    end = time.time()
    result.append(round(end-start,5))
    return result

# Recursive Solution with use of "Memoization"
def tCMP_memoization_getMinCurrency(i, currency, currentAmount):
    global currency_used 

    if currentAmount==0: # if there is no need for change
        return [0,[],0]
    elif i==-1 or currentAmount<0: # if recursion has ended 
        return [float("inf"),[],0] # float("inf") acts as an unbounded upper value for comparison
    else:
        basic_operations_counter = 0
        # process of looking for the best currency to return
        if currency_used[i-1][currentAmount] != None: # if value has already been calculated
            alternative = currency_used[i-1][currentAmount]
        else:
            aux = tCMP_memoization_getMinCurrency(i-1, currency, currentAmount)
            alternative = aux[0],aux[1]
            currency_used[i-1][currentAmount] = alternative
            basic_operations_counter += aux[2]

        if currentAmount-currency[i]>=0: # if index to be compared is inside of bounds
            if currency_used[i][currentAmount-currency[i]] != None: # if value has already been calculated
                best = currency_used[i][currentAmount-currency[i]]
            else:
                aux = tCMP_memoization_getMinCurrency(i, currency, currentAmount-currency[i])
                best = aux[0],aux[1]
                currency_used[i][currentAmount-currency[i]] = best
                basic_operations_counter += aux[2]
        
            basic_operations_counter += 1
            # returning the best currency        
            if alternative[0]<1+best[0]: # if alternative is better than the best 
                return [alternative[0], alternative[1], basic_operations_counter]
            return [1+best[0],[currency[i]]+best[1], basic_operations_counter]
        else:
            return [alternative[0], alternative[1], basic_operations_counter]

def tCMP_memoization(currency, amount):
    global currency_used

    start = time.time()
    if len(currency)==0:
        print("Error: currency must contain at least 1 type of change to return.")
        return [0,[],0,time.time()-start]

    result = tCMP_memoization_getMinCurrency(len(currency)-1, currency, amount)
    end = time.time()
    result.append(round(end-start,5))
    return result

# Dynamic Programming Solution:
def tCMP_dynamic(currency, amount):
    start = time.time()

    min_currency = [0] # possible minimum number of coins/bills that permit the return of the correct change
    currency_used = [[]] # possible combinations of coins/bills that return the correct change

    if len(currency)==0:
        print("Error: currency must contain at least 1 type of change to return.")
        end = time.time()
        return [0,[],0,round(end-start,5)]

    if amount==0:
        end = time.time()
        return [0,[],0,round(end-start,5)]

    basic_operations_counter = 0
    for i in range(1, amount + 1):
        best = float("inf") # float("inf") acts as an unbounded upper value for comparison
        best_currency_used = [] # the actual coins/bills used to make the change (the best combination)

        for j in currency: # for each available coin/bill
            if i-j >= 0 and min_currency[i-j]+1 < best: # if coin/bill isn't too high and it offers a solution that requires less coins/bills
                best = min_currency[i-j] + 1
                best_currency_used = currency_used[i-j] + [j]
                basic_operations_counter += 1 # 1 for each iteration

        min_currency.append(best)
        currency_used.append(best_currency_used)

    end = time.time()
    return [best, best_currency_used, basic_operations_counter, round(end-start,5)]

################################################################################ Auxiliary Functions #################################################################

# converts currency arrays into strings
def currency_toString(currency):
    retval = "["
    for i in range(0,len(currency)):
        if i==len(currency)-1:
            retval += str(currency[i]) + "]"
        else:
            retval += str(currency[i]) + " "
    return retval

# smooths time results by calculating averages between value groups (hence, reducing the number of total values)
def smooth_results(amounts,times,factor=5):
    # select only relevant amounts
    new_amounts = []
    for a in amounts:
        if a%factor == 0:
            new_amounts.append(a)
    new_amounts = amounts
    
    # smooth times (to remove extraneous values)
    new_times = []
    for t in times:
        new_t = []
        new_t.append(t[0])
        smoothed_time = 0
        for i in range(1,len(t)):
            smoothed_time += t[i]
            if amounts[i]%factor == 0:
                smoothed_time = round(smoothed_time/factor,5)
                new_t.append(smoothed_time)
                smoothed_time = 0
        new_times.append(new_t)

    return new_amounts,new_times

# generates a plot with a study's results
def tCMP_generatePlot(amounts,times,title,smoothFactor=1,logBase=None):
    algorithms = ["recursive","recursive w/ memoization","dynamic programming",""]
    primary_colors = ["#ff0000","#00e532","#0a00ff","#8b00ff"]
    #secondary_colors = ["#b60000","#008a1e","#050075","#510094"]

    # smooth values (reduces total number of values)
    if smoothFactor > 1:
        amounts, times = smooth_results(amounts,times,smoothFactor)

    # convert time values to logarithmic scale
    if logBase and logBase>=2:
        new_times = []
        for time in times:
            new_time = []
            for t in time:
                new_t = t+1
                new_t = math.log(new_t,logBase)
                new_time.append(new_t)
            new_times.append(new_time)
        times = new_times

    ## first figure with recursion

    # prepare figure
    #plt.gcf().subplots_adjust(left=0.5)
    fig, ax = plt.subplots()
    ax.set(xlabel="amount (units)", ylabel="time (s)", title="Execution Time Evolution:")
    #ax.grid()

    # generate plot
    amounts = np.array(amounts)
    for i in range(0,len(times)):
        # original 
        times[i] = np.array(times[i])
        ax.plot(amounts, times[i], primary_colors[i],label=algorithms[i])
        # regression
        '''
        regr = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y)
        func = regr[0][0] * np.exp(regr[0][1]*x)
        '''
        '''
        regr = scipy.optimize.curve_fit(lambda x,a,b: a + b*x,  amounts,  times[i])
        print(regr)
        func = regr[0][0] + np.exp(regr[0][1]*amounts)
        print(func)
        ax.plot(amounts, func, secondary_colors[i],label=algorithms[i]+" regression")
        '''

    # save figure
    ax.legend(loc='upper left')
    fig.savefig("../results/"+title+".png",bbox_inches='tight')

    ## second figure without recursion

    # prepare figure
    fig2, ax2 = plt.subplots()
    ax2.set(xlabel="amount (units)", ylabel="time (s)", title="Execution Time Evolution:")

    # generate plot
    amounts = np.array(amounts)
    for i in range(1,len(times)):
        # original 
        times[i] = np.array(times[i])
        ax2.plot(amounts, times[i], primary_colors[i],label=algorithms[i])
        # regression
        '''
        regr = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y)
        func = regr[0][0] * np.exp(regr[0][1]*x)
        '''
        '''
        regr = scipy.optimize.curve_fit(lambda x,a,b: a + b*x,  amounts,  times[i])
        print(regr)
        func = regr[0][0] + np.exp(regr[0][1]*amounts)
        print(func)
        ax.plot(amounts, func, secondary_colors[i],label=algorithms[i]+" regression")
        '''

    # save figure
    ax2.legend(loc='upper left')
    fig2.savefig("../results/"+title+"_memory.png",bbox_inches='tight')

    return

################################################################################ Results Tables ######################################################################
def tCMP_simple_study(currencies, amounts, outputFile, smoothFactor=1,logBase=None):
    global currency_used

    # opening file for writting
    f = None
    try:
        f = open(outputFile + ".csv", "w")
    except:
        print("Error: Unable to open file for writting.")
        exit(1)

    # preparing headers for presenting results
    print("\nSimple Study of 'The Change-Making Problem'\n")
    print("Description: In this study the algorithms are run several times, each with a different amount, and the process is repeated for several currencies.\n")
    '''print(" %-45s | %-25s | %-25s | %-25s |" % (" ","Recursive","Recursive w/ Memoization","Dynamic Programming"))
    print("|----------------------------------------------+---------------------------+---------------------------+---------------------------|")
    print("| %-35s | %-6s | %-13s | %-9s | %-13s | %-9s | %-13s | %-9s |" % ("Currency","Amount","# of Basic Op","Exec Time","# of Basic Op","Exec Time","# of Basic Op","Exec Time"))
    print("|-------------------------------------+--------+---------------+-----------+---------------+-----------+---------------+-----------|")
    '''
    f.write("Currency, Amount, # of Basic Operations (Recursive), Execution Time (Recursive), # of Basic Operations (Recursive w/ Memoization), Execution Time (Recursive w/ Memoization), # of Basic Operations (Dynamic Programming), Execution Time (Dynamic Programming)\n")
    
    # initializing time lists
    times = []
    times_recursive = []
    times_memoization = []
    times_dynamic = []

    for c in currencies:
        for a in amounts:

            # calculating results for each algorithm
            recursive = tCMP_recursive(c, a) ############################## recursive is paused!!!
            currency_used = [[None for x in range(0,a+1)] for x in range(0,len(c)+1)] # reseting global memory for memoization
            memoization = tCMP_memoization(c, a)
            dynamic = tCMP_dynamic(c, a)
            
            # printing results in a raw format
            print(" > Amount: %d, Currency: %s" % (a,currency_toString(c)))
            print(recursive)
            print(memoization)
            print(dynamic)
            
            # printing results in a formatted table & writing them into a csv file
            #print("| %-35s | %-6s | %-13s | %-9s | %-13s | %-9s | %-13s | %-9s |" % (c,a,recursive[2],recursive[3],memoization[2],memoization[3],dynamic[2],dynamic[3]))
            f.write("%s, %s, %s, %s, %s, %s, %s, %s \n" % (currency_toString(c),a,recursive[2],recursive[3],memoization[2],memoization[3],dynamic[2],dynamic[3]))
            
            # preparing data for plotting
            times_recursive.append(recursive[3])
            times_memoization.append(memoization[3])
            times_dynamic.append(dynamic[3])

        # plotting results into a png file
        times.append(times_recursive)
        times.append(times_memoization)
        times.append(times_dynamic)
        tCMP_generatePlot(amounts,times,outputFile+"_plot_"+currency_toString(c),smoothFactor,logBase)

        # clearing time lists for next currency
        times_recursive.clear()
        times_memoization.clear()
        times_dynamic.clear()
        times.clear()

    print("+-------------------------------------+--------+---------------+-----------+---------------+-----------+---------------+-----------+")
    f.close()
    return

def tCMP_elaborate_study(currencies, amounts, outputFile, n=1, smoothFactor=1, logBase=None):
    global currency_used

    # opening file for writting
    f = None
    try:
        f = open("../results/" + outputFile + ".csv", "w")
    except:
        print("Error: Unable to open file for writting.")
        exit(1)

    # preparing headers for presenting results
    print("\nElaborate Study of 'The Change-Making Problem'\n")
    print("Description: In this study the simple study is run 'n' times and the execution time averages are calculated in order to remove ocasional time peaks from external reasons.\n")
    print(" %-45s | %-25s | %-25s | %-25s |" % (" ","Recursive","Recursive w/ Memoization","Dynamic Programming"))
    print("|----------------------------------------------+---------------------------+---------------------------+---------------------------|")
    print("| %-35s | %-6s | %-13s | %-9s | %-13s | %-9s | %-13s | %-9s |" % ("Currency","Amount","# of Basic Op","Exec Time","# of Basic Op","Exec Time","# of Basic Op","Exec Time"))
    print("|-------------------------------------+--------+---------------+-----------+---------------+-----------+---------------+-----------|")
    f.write("Currency, Amount, # of Basic Operations (Recursive), Execution Time (Recursive), # of Basic Operations (Recursive w/ Memoization), Execution Time (Recursive w/ Memoization), # of Basic Operations (Dynamic Programming), Execution Time (Dynamic Programming)\n")
    
    # initializing time lists
    average_times = []

    for c in currencies:
        # first iteration
        tmp_recursive = []
        tmp_memoization = []
        tmp_dynamic = []
        for i in range(0,len(amounts)):
            recursive = [0,[],0,0] #tCMP_recursive(c, amounts[i]) ############################## recursive is paused!!!
            currency_used = [[None for x in range(0,amounts[i]+1)] for x in range(0,len(c)+1)] # reseting global memory for memoization
            memoization = tCMP_memoization(c, amounts[i])
            dynamic = tCMP_dynamic(c, amounts[i])

            tmp_recursive.append(recursive)
            tmp_memoization.append(memoization)
            tmp_dynamic.append(dynamic)

        average_times.append(tmp_recursive)
        average_times.append(tmp_memoization)
        average_times.append(tmp_dynamic)

        # remaining iterations
        if n>1:
            for i in range(1,n):
                for j in range(0,len(amounts)):
                    # calculating results for each algorithm
                    recursive = [0,[],0,0] #tCMP_recursive(c, amounts[j]) ############################## recursive is paused!!!
                    currency_used = [[None for x in range(0,amounts[j]+1)] for x in range(0,len(c)+1)] # reseting global memory for memoization
                    memoization = tCMP_memoization(c, amounts[j])
                    dynamic = tCMP_dynamic(c, amounts[j])
                    
                    # adding times to average_times
                    average_times[0][j][3] += recursive[3]
                    average_times[1][j][3] += memoization[3]
                    average_times[2][j][3] += dynamic[3]

            # calculating average times
            for i in range(0,len(amounts)):
                average_times[0][i][3] = round(average_times[0][i][3]/n,5)
                average_times[1][i][3] = round(average_times[1][i][3]/n,5)
                average_times[2][i][3] = round(average_times[2][i][3]/n,5)

        # printing results in a formatted table & writing them into a csv file
        for i in range(0,len(amounts)):
            print("| %-35s | %-6s | %-13s | %-9s | %-13s | %-9s | %-13s | %-9s |" % (c,amounts[i],average_times[0][i][2],average_times[0][i][3],average_times[1][i][2],average_times[1][i][3],average_times[2][i][2],average_times[2][i][3]))
            f.write("%s, %s, %s, %s, %s, %s, %s, %s \n" % (currency_toString(c),amounts[i],average_times[0][i][2],average_times[0][i][3],average_times[1][i][2],average_times[1][i][3],average_times[2][i][2],average_times[2][i][3]))

        # preparing times for plotting
        for i in range(0,len(amounts)):
            average_times[0][i] = average_times[0][i][3]
            average_times[1][i] = average_times[1][i][3]
            average_times[2][i] = average_times[2][i][3]

        # generating plot
        tCMP_generatePlot(amounts,average_times,outputFile+"_plot_"+currency_toString(c),smoothFactor,logBase)
        average_times.clear()

    print("+-------------------------------------+--------+---------------+-----------+---------------+-----------+---------------+-----------+")
    f.close()
    return

################################################################################ Code Execution ######################################################################

# Running the Studies

#tCMP_simple_study(currencies,amounts,"tCMP_simple_results",smoothF,logB)
#tCMP_elaborate_study(currencies,amounts,"tCMP_elaborate_results",studyN,smoothF,logB)

# Testing the Algorithms

print("Amount: " + str(amount))

print("Recursive Solution:")
print(tCMP_recursive(currency, amount))

print("Recursive Solution w/ Memoization:")
print(tCMP_memoization(currency, amount))

print("Dynamic Programming Solution:")
print(tCMP_dynamic(currency, amount))
#print(tCMP_dynamic_new(currency, amount))

'''
def tCMP_dynamic_new(currency, amount):
    currency_used = [[None for x in range(0,amount+1)] for x in range(0,len(currency)+1)] # possible combinations of coins/bills that return the correct change

    start = time.time()

    if len(currency)==0:
        print("Error: currency must contain at least 1 type of change to return.")
        end = time.time()
        return [0,[],0,round(end-start,5)]

    if amount==0:
        end = time.time()
        return [0,[],0,round(end-start,5)]

    for i in range(1,amount+1):
        basic_operations_counter = 0
        for j in range(0,len(currency)):
            alternative = [0,[],0]
            best = [0,[],0]

            if currency_used[len(currency)-j-1][i] != None: # if value has already been calculated
                alternative = currency_used[len(currency)-j-1][i]
            else:
                aux = 0
                k = len(currency)-1
                while aux != i:
                    if (i-currency[k]-aux) >= 0:
                        aux += currency[k]
                        alternative = [alternative[0]+1,alternative[1]+[currency[k]],alternative[2]]
                    else:
                        assert(k > 0)
                        k = k-1
                    alternative = [alternative[0],alternative[1],alternative[2]+1]

                currency_used[len(currency)-j-1][i] = [alternative[0],alternative[1]]
                basic_operations_counter += alternative[2]

            
            if i-currency[len(currency)-j-1]>=0: # if index to be compared is inside of bounds
                if currency_used[len(currency)-j-1][i-currency[len(currency)-j-1]] != None: # if value has already been calculated
                    best = currency_used[len(currency)-j-1][i-currency[len(currency)-j-1]]
                else:
                    best = alternative
            
                basic_operations_counter += 1
                # returning the best currency        
                if alternative[0]<1+best[0]: # if alternative is better than the best 
                    currency_used[len(currency)-j-1][i] = [alternative[0], alternative[1], basic_operations_counter]
                currency_used[len(currency)-j-1][i] = [1+best[0],[currency[j]]+best[1], basic_operations_counter+best[2]+alternative[2]]
            else:
                currency_used[len(currency)-j-1][i] = [alternative[0], alternative[1], basic_operations_counter+alternative[2]+best[2]]

    best_currency_used = currency_used[0][amount]
    for i in range(1,len(currency)):
        if currency_used[i][amount][0] < best_currency_used[0]:
            best_currency_used = currency_used[i][amount]

    end = time.time()
    return [best_currency_used[0], best_currency_used[1], best_currency_used[2], round(end-start,5)]
'''