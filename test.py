import  numpy as np
import torch
import  time
import  threading


condition1=threading.Condition(threading.Lock())
condition2=threading.Condition(threading.Lock())


def fun1():
    for i in range(10):
        print(f'condition1 {i}')
    with condition1:
        print('condtion1 acquire ')
        condition1.wait()
    for i in range(10):
        print(f'condition2 {i}')

def fun2():
    time.sleep(5)
    with condition1:
        print('condtion2 acquire ')
        condition1.notify()
    print('notify all')


a=threading.Thread(target=fun1)
b=threading.Thread(target=fun2)

a.start()
b.start()

