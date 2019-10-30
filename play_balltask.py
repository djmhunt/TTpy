from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

from experiment.balltask import Balltask

print("Welcome! We shall now play the balltask")
print("you will see a bunch of balls, one at a time")
print("the balls can be three different colors, red, green or blue")
print("your task is to predict the next color")

exp = Balltask()

exp.reset()
for state in exp:
    print("trial #{}".format(exp.trial))
    print("you see {}".format(state[0]))
    print("Please predict next color from these options {}".format(state[1]))
    action = int(raw_input("enter your choice: "))
    exp.receiveAction(action)
    response = exp.feedback()
    exp.procede()  # not actually needed but should be here for future reference


print("experiment complete!")