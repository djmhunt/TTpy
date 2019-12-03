from __future__ import division, print_function
from __future__ import unicode_literals, absolute_import

# actually let's test the decks task, we know that works.
# let's see if we can play that ourselves.

from experiment.decks import Decks

print("Welcome! We shall now play the Worthy task")
print("you have two decks to pick cards from, you pick 80 cards")
print("your goal is to reach at least 450 points")

decky = Decks()

# decky.Name
total_points = 0
for state in decky:
    print("draw #{}".format(decky.t))
    print("Your total points are {}".format(total_points))
    print("Please pick an action between {}".format(state[1]))
    action = int(raw_input("enter your choice: "))
    decky.receiveAction(action)
    response = decky.feedback()
    print("you got {} points!".format(response))
    total_points += response
    decky.proceed()  # not actually needed because it's only used if


print("experiment complete! you raeched {} points!".format(total_points))

# counter
# decky.next()  # next will return (nextstim, nextvalidactions)
# # and SHOULD be what "state" is in above loop right?
# decky.t
