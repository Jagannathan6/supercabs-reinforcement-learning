# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
choices = [(1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)]
# {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
# {'MON':0, 'TUE':1, 'WED':2, 'THU':3, 'FRI':4, 'SAT':5, 'SUN':6}


class CabDriver():

    def __init__(self):

        self.overall_travel_hours = 0
        self.action_space = []
        self.state_space = []

        #Given any state, following are the list of actions that can be performed
        self.action_space = [(1,2), (2,1),
                            (3,4), (4,3),
                            (3,5), (5,3),
                            (1,5), (5,1),
                            (1,3), (3,1),
                            (1,4), (4,1),
                            (4,5), (5,4),
                            (2,3), (3,2),
                            (2,4), (4,2),
                            (2,5), (5,2),
                            (0,0)]

        for i in range(5):
            for j in range(24):
                for k in range(7):
                    self.state_space.append([i,j,k])

        # Choose the list of choices
        self.state_init = random.choice(choices)


        self.reset()

    def reset(self):
        self.overall_travel_hours = 0
        self.state_init = random.choice(choices)
        return self.action_space, self.state_space, self.state_init

    # Encoded state of the neural network input
    def state_encod_arch1(self, state):

        if not state:
            return

        state_encod = [0] * (m + t + d)

        # encode location
        state_encod[state[0] - 1] = 1

        # encode hour of the day
        state_encod[m + state[1]] = 1

        # encode day of the week
        state_encod[m + t + state[2]] = 1

        return state_encod



    def requests(self, state):


        # use poisson distribution for generating random # of requests based on average
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        pai = random.sample(range(1, (m-1)*m +1), requests)
        actions = []
        for i in pai:
            actions.append(self.action_space[i])

        if (0, 0) not in actions:
            actions.append((0,0))
            pai.append(20)

        return pai,actions



    # This function calculates the reward function based on the state and action.
    # It takes and Location and Time Factors and compute the reward.

    def reward_func(self, state, action, Time_matrix):
        current_location = state[0]
        start_location = action[0]
        end_location = action[1]
        time_of_day = state[1]
        day_of_week = state[2]

        #
        def get_new_time_day(time_of_day, day_of_week, total_time):
            """
            calculates new time and day
            """
            time_of_day = time_of_day + total_time % (t - 1)
            day_of_week = day_of_week + (total_time // (t - 1))

            if time_of_day > (t-1):
                day_of_week = day_of_week + (time_of_day // (t - 1))
                time_of_day = time_of_day % (t - 1)
                if day_of_week > (d - 1):
                    day_of_week = day_of_week % (d - 1)

            return time_of_day, day_of_week

        # Find the total travel time based on the positon and time factors.
        def get_total_travel_time(current_location, start_location, end_location, time_of_day, day_of_week):

            if not start_location and not end_location:
                return 0, 1

            t1 = 0

            if start_location and current_location != start_location:
                #Find the time taken to move from current location to start location
                t1 = int(Time_matrix[current_location-1][start_location-1][time_of_day][day_of_week])
                #Find the updated time and day in the week.
                time_of_day, day_of_week = get_new_time_day(time_of_day, day_of_week, t1)

            t2 = int(Time_matrix[start_location-1][end_location-1][time_of_day][day_of_week])

            return t1, t2

        # Cacluate the total travel time from starting to ending location.
        t1, t2 = get_total_travel_time(current_location, start_location, end_location, time_of_day, day_of_week)
        # Reward = Revenue calculated from pickup to Drop. - (Fuel used from moving from pickup to Drop + Fuel used from current position to Pickup)

        if not start_location and not end_location:
            reward = -C
        else:
            #Compute the actual reward.
            reward = R * t2 - C * (t1 + t2)

        return reward



    def next_state_func(self, state, action, Time_matrix):
        """
        Takes state and action as input and returns next state
        """
        current_location = state[0]
        start_location = action[0]
        end_location = action[1]
        time_of_day = state[1]
        day_of_week = state[2]

        #-----------------
        def get_total_travel_time(current_location, start_location, end_location, time_of_day, day_of_week):
            """
            calculates the total time of trave based on
            """
            if not start_location and not end_location:
                return 1

            t1 = 0
            if start_location and current_location != start_location:
                # Time taken from moving from pickup point to start locaiton of customer.
                t1 = int(Time_matrix[current_location-1][start_location-1][time_of_day][day_of_week])

                # compute new time_of_day and day_of_week after travel t1
                time_of_day, day_of_week = get_new_time_day(time_of_day, day_of_week, t1)

            # Time taken for the actual customer journey
            t2 = int(Time_matrix[start_location-1][end_location-1][time_of_day][day_of_week])
            return t1 + t2


        def get_new_time_day(time_of_day, day_of_week, total_time):
            """
            calculates new time and day
            """
            time_of_day = time_of_day + total_time % (t - 1)
            day_of_week = day_of_week + (total_time // (t - 1))

            if time_of_day > (t-1):
                day_of_week = day_of_week + (time_of_day // (t - 1))
                time_of_day = time_of_day % (t - 1)
                if day_of_week > (d - 1):
                    day_of_week = day_of_week % (d - 1)

            return time_of_day, day_of_week

        #Find the total time spent in travel
        total_travel_time_taken = get_total_travel_time(current_location, start_location, end_location, time_of_day, day_of_week)
        #Find the time spend in travel overall
        self.overall_travel_hours = self.overall_travel_hours + total_travel_time_taken
        #Update the day and time.
        new_time_of_day, new_day_of_week = get_new_time_day(time_of_day, day_of_week, total_travel_time_taken)

        if not start_location and not end_location:
            new_location = state[0]
        else:
            new_location = action[1]

        return (new_location, new_time_of_day, new_day_of_week)
