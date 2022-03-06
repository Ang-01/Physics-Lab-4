import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
def line(x,a,b):
    return a*x+b

lab4 = pd.read_csv("lab4.csv")

Voltages = {100:['indigo','blue'],200:['teal','c'],300:['m','r'],400:['olive','olive'],500:['fuchsia','pink']}
#i is in mA, need to convert to A
C=(4*math.pi*10**(-7)*132)/(0.1475*(1*1/4)**(3/2))
em_values=[]
em_values_error=[]
for x in Voltages:
    i = lab4['I'+str(int(x/100))]
    r = lab4['r'+str(int(x/100))]/200
    one_r = r**(-1)
    error_i = 10*[0.01]
    plt.errorbar(one_r, i, yerr=error_i,fmt='o',ecolor='black',color=Voltages[x][0],capsize=5, label=str(x)+' Volts')
    popt, pcov = curve_fit(line,one_r,i,sigma=error_i)
    #print(popt,pcov)
    print(f"A_",x," V =", popt[0], "+/-", pcov[0,0]**0.5)
    print("b_",x, " V =", popt[1], "+/-", pcov[1,1]**0.5)
    em_values+=[(2*x)/((C*popt[0])**2)]
    #em_values_error+=[(pcov[0,0]**0.5)/(popt[0]**2)]

    xfine = np.arange(15,41,1)
    plt.plot(xfine, line(xfine, popt[0], popt[1]), color=Voltages[x][1], label=str(x)+' Volts')
    plt.title("Least Squares Linear Fit of Required I With Respect to 1/r")
    plt.xlabel("1/r (1/m)")
    plt.ylabel("i (A)")
    plt.legend(bbox_to_anchor =(1, 0.75))
#    plt.savefig("Graph1.png")
plt.savefig("EM 1.png")
plt.show()
