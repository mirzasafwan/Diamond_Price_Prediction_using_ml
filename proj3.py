import time
import numpy as np
import tkinter as tk
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler	
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import requests

from tkinter import Toplevel, messagebox
from tkinter.messagebox import *
from tkinter.ttk import Treeview

window = Tk()
window.geometry("1030x5000+250+0")
window.resizable(False, False)
window.title("Diamond Price Prediction")
from PIL import ImageTk, Image
img = ImageTk.PhotoImage(Image.open("diamond.png"))  
l=Label(image=img)
l.pack()
window.attributes('-alpha',0.8)

# clock
def tick():
    time_string = time.strftime("%H:%M:%S")  # strftime=places current time
    date_string = time.strftime("%d/%m/%Y")
    # clock.config(text='Date :'+date_string+"\n"+"Time :"+time_string)
    clock.config(text="Time : " + time_string + "\n" + 'Date : ' + date_string)
    clock.after(200, tick)

clock = Label(window, font=('arial', 13, 'bold'), relief=RIDGE, borderwidth=4, bg='white')
clock.place(x=0, y=0)
tick()

# tempreature
try:
    city_name = "Mumbai"
    a1 = "http://api.openweathermap.org/data/2.5/weather?units=metric"
    a2 = "&q=" + city_name
    a3 = "&appid=" + "c6e315d09197cec231495138183954bd"

    web_add = a1 + a2 + a3
    res = requests.get(web_add)
    # print(res)
    data = res.json()
    # print(data)

    m = data['main']
    t = m['temp']
    t = str(t)
    # print("t =", t)

except Exception as e:
    print("Issue ", e)

tempN = Label(window, font=('arial', 14, 'bold'), relief=RIDGE, borderwidth=6, bg='white')
tempN.place(x=833, y=0)
tempN.config(text="Temperature : " + t)

Label(window,text="Enter Carat",fg="black",font="arial 20 bold").place(x=410,y=100)
carat=StringVar()
Entry(window,text=carat,bd=3, font=('arial', 20, 'bold')).place(x=350,y=150)

Label(window,text="Enter Cut",fg="black",font="arial 20 bold").place(x=410,y=200)
cut=StringVar()
Entry(window,text=cut,bd=3, font=('arial', 20, 'bold')).place(x=350,y=250)


Label(window,text="Enter Color",fg="black",font="arial 20 bold").place(x=410,y=300)
color=StringVar()
Entry(window,text=color,bd=3, font=('arial', 20, 'bold')).place(x=350,y=350)

Label(window,text="Enter Clarity",fg="black",font="arial 20 bold").place(x=410,y=400)
clarity=StringVar()
Entry(window,text=clarity,bd=3, font=('arial', 20, 'bold')).place(x=350,y=450)


#delete.StringVar()

def submit():
    data=pd.read_csv("diamonds.csv")
    data=data.drop(['depth','table','x','y','z'],axis=1)
    data['price']=data.price.astype(float)
    
    le1=LabelEncoder()
    ct=le1.fit_transform(data['cut'])
    le1.classes_ 
    data['cut_label']=ct
    
    le2=LabelEncoder()
    clr=le2.fit_transform(data['color'])
    le2.classes_
    data['color_label']=clr
    #data['color'].isnull().sum
    
    le3=LabelEncoder()
    clt=le3.fit_transform(data['clarity'])
    le3.classes_
    data['clarity_label']=clt

    X=data.drop(['clarity','color','price','cut'],axis=1)
    Y=data['price']
    
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.2,random_state=42)
    
    ss=StandardScaler()
    X_test=ss.fit_transform(X_test)
    X_train=ss.fit_transform(X_train)
    
    lre=LinearRegression()  
    lre.fit(X_train,Y_train)
    pred=lre.predict(X_test)
    """plt.figure(figsize=(4,4))
    plt.scatter(Y_test, pred, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    p1 = max(max(pred), max(Y_test))
    p2 = min(min(pred), min(Y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title("Actual Vs Predicted value of Linear regressor  \n",fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Red'})
dd    plt.show()"""
    
    scr=r2_score(Y_test, pred)*100
    print(scr)
    
    dt=DecisionTreeRegressor()
    dt.fit(X_train,Y_train)
    pred1=dt.predict(X_test)
    t1=np.array_str(pred1)
    
    """plt.figure(figsize=(4,4))
    plt.scatter(Y_test, pred1, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    p1 = max(max(pred1), max(Y_test))
    p2 = min(min(pred1), min(Y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title("Actual Vs Predicted value of Decision tree regressor  \n",fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Red'})
    plt.show()"""
    
    scr1=r2_score(Y_test, pred1)*100
    print(scr1)
    
    X_test=([[carat.get(),cut.get(),color.get(),clarity.get()]])#[0]*0.1
    
    #print(X_test)
    
    price=dt.predict(X_test)[0]*0.1
    Y1="{:.1f}".format(price)
    
    return messagebox.showinfo('Prediction of diamond price',f'Price Prediction of Diamond is  Rs. {Y1}')
    
def graph():
    data=pd.read_csv("diamonds.csv")
    groupdata = data.groupby("cut").count()
    print(groupdata)
    x_vals = []
    y_vals = []
    for i in [0, 1, 4, 3, 2]: 
        x_vals.append(groupdata.index[i])
        y_vals.append(groupdata.iloc[i,0])
    plt.bar(x_vals, y_vals)
    explode = (0, 0, 0, 0, 0.2)  # only "explode" the 2nd slice (i.e. 'Very good')
    plt.figure(figsize = [5,5])
    plt.pie(y_vals, explode=explode, labels=x_vals, autopct='%2.1f%%', textprops={'fontsize': 12, 'fontweight' : 7, 'color' : 'Black'}, startangle=90)
    plt.title("Distribution based on quality of cut \n",fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Red'})
    plt.show()
    
    #plot using scatter plot method
  
    data=pd.read_csv("diamonds.csv")
    data=data.drop(['depth','table','x','y','z'],axis=1)
    data['price']=data.price.astype(float)
    
    le1=LabelEncoder()
    ct=le1.fit_transform(data['cut'])
    le1.classes_ 
    data['cut_label']=ct
    
    le2=LabelEncoder()
    clr=le2.fit_transform(data['color'])
    le2.classes_
    data['color_label']=clr
    #data['color'].isnull().sum
    
    le3=LabelEncoder()
    clt=le3.fit_transform(data['clarity'])
    le3.classes_
    data['clarity_label']=clt

    X=data.drop(['clarity','color','price','cut'],axis=1)
    Y=data['price']
    
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.2,random_state=42)
    
    ss=StandardScaler()
    X_test=ss.fit_transform(X_test)
    X_train=ss.fit_transform(X_train)
    
    lre=LinearRegression()  
    lre.fit(X_train,Y_train)
    pred=lre.predict(X_test)
    plt.figure(figsize=(4,4))
    plt.scatter(Y_test, pred, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    p1 = max(max(pred), max(Y_test))
    p2 = min(min(pred), min(Y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title("Actual Vs Predicted value of Linear regressor  \n",fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Red'})
    plt.show()
        
    dt=DecisionTreeRegressor()
    dt.fit(X_train,Y_train)
    pred1=dt.predict(X_test)
    plt.figure(figsize=(4,4))
    plt.scatter(Y_test, pred1, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    p1 = max(max(pred1), max(Y_test))
    p2 = min(min(pred1), min(Y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.title("Actual Vs Predicted value of Decision tree regressor  \n",fontdict={'fontsize': 20, 'fontweight' : 20, 'color' : 'Red'})
    plt.show()
    
    
    plt.xlabel('Carat', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    plt.scatter(data = data , x = 'carat', y = 'price')
    plt.show()
    plt.figure(figsize = [14,10])


    
def clear():
    carat.set("")
    cut.set("")
    color.set("")
    clarity.set("")
def exitbtn():
    res = messagebox.askyesno('Notification', 'Do you want to exit')
    if (res == True):
        window.destroy()

Button(window,text="Predict",command=submit,font="arial 20 bold",  width=7, relief=RIDGE, borderwidth=5).place(x=350,y=510)

Button(window,text="Graph" ,command=graph,font="arial 20 bold",  width=8, relief=RIDGE, borderwidth=5).place(x=510,y=510)

Button(window, text="Clear", command=clear, font=('arial', 20, 'bold'), width=7, relief=RIDGE, borderwidth=5).place(x=350,y=590)

Button(window, text="Exit", command=exitbtn, font=('arial', 20, 'bold'), width=8, relief=RIDGE, borderwidth=5).place(x=510,y=590)

window.mainloop()