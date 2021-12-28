from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User
import matplotlib.pyplot as plt

if __name__ == "__main__":
     f = open("ve.txt",'w') # mở file mode ‘w’ để ghi
     M = [1,2,3,4,5,6,7,8,9,10]
     N = [1,3,5,6,7,8,9,5,2,5]
     Z = [3,2,3,4,5,2,5,8,2,9]
     for i in M :
         f.write("{}\n".format(i))    
     #plt.plot(M,N)
     #plt.plot(M,Z)
     #plt.show()
    
     
     f.close()