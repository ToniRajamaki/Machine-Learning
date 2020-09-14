import matplotlib.pyplot as plt
import numpy as np

x_values = []
y_values = []

def my_linfit(x_list,y_list):

    N = len(x_list)

    x = np.array(x_list)
    y = np.array(y_list)


    b = (sum(y)*sum(x*x)-sum(x)*sum(x*y))/(N*sum(x*x)-sum(x)*sum(x))
    a = (N*sum(x*y)-sum(x)*sum(y))/(N*sum(x*x)-sum(x)*sum(x))
    return a,b


def onclick(event):
    global x_values
    global y_values

    if(event.button == 1):

        x_values.append(event.xdata)
        y_values.append(event.ydata)
        plt.scatter(event.xdata,event.ydata,c='blue')
        plt.show()
        return

    if(event.button == 3): #on right click plot fitted line to data points

        a, b = my_linfit(x_values, y_values)
        #a,b = np.polyfit(x_values,y_values,1)
        xp = np.arange(1, 100, 0.1)
        plt.plot(xp, a * xp + b, 'r-')
        plt.show()
        return

plt.show()
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.title('Left click to add datapoints, rightclick to fit line')
plt.axis([0,100,0,1000])
plt.grid(True)
plt.show()


