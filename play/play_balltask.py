# coding=utf-8
from tasks.balltask import Balltask

print("Welcome! We shall now play the balltask")
print("you will see a bunch of balls, one at a time")
print("the balls can be three different colors, red, green or blue")
print("your task is to predict the next color")

tsk = Balltask()

for state in tsk:
    print("trial #{}".format(tsk.trial))
    print("you see {}".format(state[0]))
    print("Please predict next color from these options {}".format(state[1]))
    action = int(raw_input("enter your choice: "))
    tsk.receiveAction(action)
    response = tsk.feedback()
    tsk.proceed()  # not actually needed but should be here for future reference

print("tasks complete!")
