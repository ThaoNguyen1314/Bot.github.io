from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User
import numpy as np

#import class "UserSimulator" từ user_simulator.py
#import class "ErrorModelController" từ error_model_controller.py
#import class "DQNAgent" từ dqn_agent.py
#import class "StateTracker" từ state_tracker.py
#import class "User" từ user.py
if __name__ == "__main__":
    #Khi một trình thông dịch Python đọc một tệp Python, trước tiên nó sẽ đặt một vài biến đặc biệt. Sau đó, nó thực thi mã từ tệp.
    #Một trong những biến đó được gọi là __name__.
    #Các tệp .py được gọi là module. Một module có thể xác định các hàm, class và biến.
    #Vì vậy, khi trình thông dịch chạy một module, biến __name__ sẽ được đặt là __main__ nếu module đang được chạy là chương trình chính.
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser() #tạo tham số trong môi trường tương tác với máy tính bằng cmd
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'#Có thể load lên file json khác tại đây
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    # Load file path constants
    # Load lên đường dẫn file constants
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    WARMUP_MEM = run_dict['warmup_mem']
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    MAX_ROUND_NUM = run_dict['max_round_num']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load movie DB 
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')
    #module 'pickle' dùng để save và load object (gọi hàm open và load) 
    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')
    #module 'pickle' dùng để save và load object (gọi hàm open và load) 
    # Load goal File
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects
    if USE_USERSIM: # Nếu khởi chạy USE_USERSIM (run_dict['usersim'])
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def run_round(state, warmup=False): # Xem ảnh model trong file tutorial để hiểu rõ
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=warmup)
    #1) 'Agent' nhận được state dựa trên sự miêu tả của 'state tracker' về cuộc đối thoại và đưa ra action từ 'get action'

    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent(agent_action)
    #2) action từ 'Agent' được gửi đến 'Update w/Agent' của 'state tracker' và được update ở đó

    # 3) User takes action given agent action
    user_action, reward, done, success = user.step(agent_action)
     # 3) 'action updated' dưới dạng đầu vào được gửi đến 'Step' method của 'User Sim'. Trong 'Step', 'User Sim' tạo response dựa trên rule-based của chính nó và xuất ra
    # thông tin reward,success,user_action,done.

    if not done: # Nếu không done
        # 4) Infuse error into semantic frame level of user action
        emc.infuse_error(user_action)
        #4)'user_action' được đưa tới 'infuse error' của 'EMC' và được thêm lỗi cho nó

    # 5) Update state tracker with user action
    state_tracker.update_state_user(user_action)
    #5)Chuyển 'user_action' with error tới 'Update w/ User' của 'State Tracker' và update state user mới

    # 6) Get next state and add experience
    next_state = state_tracker.get_state(done) #get next state từ 'Get State' của 'State Tracker'
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)#'add_experience' từ 'dqn_agent'
    #6) Get next state và add experience

    return next_state, reward, done, success
    #Trả về next_state, reward, done, success  


def warmup_run(): # Hàm chạy warmup
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
    Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

    """

    print('Warmup Started...')
    total_step = 0
    while total_step != WARMUP_MEM and not dqn_agent.is_memory_full():
    #xác định vòng lặp chỉ chạy cho đến khi bộ nhớ của Agent được lấp đầy vào WARMUP_MEM(total_step = WARMUP_MEM) hoặc dqn_agent.is_memory_full()=false.
        
        episode_reset()# Reset episode (Thiết lập lại episode)
        done = False #thiết lập done = false
        state = state_tracker.get_state()# Get initial state from state tracker (Lấy lại state ban đầu bằng state_tracker.get_state() )
        
        while not done: #Chạy vòng lặp cho đến khi done=TRUE (episode kết thúc)
            next_state, _, done, _ = run_round(state, warmup=True)
            total_step += 1
            state = next_state

    print('...Warmup Ended') # in '...Warmup Ended' khi kết thúc.

suc = open('success2.txt','w')
avg = open('avg2','w')

def train_run():#Hàm chạy train
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """
    # Chạy vòng lặp trains Agent
    # Train Agent về nhiệm vụ định hướng goal. 
    # Việc train Neural network của Agent xảy ra trên mọi episode mà TRAIN_FREQ là bội số của nó. Kết thúc khi số episode đạt đến NUM_EP_TRAIN.
    print('Training Started...')
    #set các giá trị ban đầu =0
    episode = 0 
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    
    while episode < NUM_EP_TRAIN:
    #Chạy vòng lặp cho tới khi episode đạt đến NUM_EP_TRAIN.
        episode_reset() # thiết lập lại episode sau mỗi tập
        episode += 1 #tăng bộ đếm episode
        print(episode)
        done = False #thiết lập done = false
        state = state_tracker.get_state() #Lấy lại state ban đầu bằng state_tracker.get_state()
        while not done:#Chạy vòng lặp cho đến khi done=TRUE (episode kết thúc)
            next_state, reward, done, success = run_round(state) 
            #trả về next_state, reward, done, success bằng hàm run_round với đầu vào state  
            period_reward_total += reward #Tăng period_reward_total 
            state = next_state

        period_success_total += success # Tăng  period_success_total 

        # Train
        if episode % TRAIN_FREQ == 0: #Nếu số episde là bội số của số lần TRAIN_FREQ

            # Check success rate (kiểm tra tỉ lệ success)
            success_rate = period_success_total / TRAIN_FREQ # success_rate = period_success_total chia cho số lần TRAIN_FREQ
            avg_reward = period_reward_total / TRAIN_FREQ 
            suc.write("{}\n".format(success_rate)) 
            avg.write("{}\n".format(avg_reward)) 
            
            # Flush
            if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
            # Nếu success_rate >= success_rate_best hoặc success_rate >= SUCCESS_RATE_THRESHOLD(ngưỡng SUCCESS_RATE)
                dqn_agent.empty_memory()#Hàm làm trống bộ nhớ và đặt lại index cho bộ nhớ
            
            # Update current best success rate 
            if success_rate > success_rate_best: #basic
                print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}' .format(episode, success_rate, avg_reward))
                success_rate_best = success_rate
                dqn_agent.save_weights()#Lưu lại trọng số của cả 2 model behavior và target vào 2 files h5 bằng hàm save_weights
            period_success_total = 0
            period_reward_total = 0
            # Copy
            dqn_agent.copy()#Hàm sao chép trọng số của behavior model vào trọng số của target model
            # Train
            dqn_agent.train()
            #Hàm train agent bằng cách cải thiện behavior model với bộ nhớ. Lấy các batches bộ nhớ từ vùng memory và xử lý chúng.
            #Là quá trình lấy các tuples và xếp chúng ở định dạng chính xác cho Neural Network và tính toán phương trình Bellman cho Q-learning.

    print('...Training Ended')            

def episode_reset(): #Hàm thiết lập lại episode cho quá trình warmup và training
    """
    Resets the episode/conversation in the warmup and training loops.

    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

    """

    # First reset the state tracker
    state_tracker.reset() #THiết lập lại 'state tracker'
    # Then pick an init user action
    user_action = user.reset() #thiết lập lại 'user_action'
    # Infuse with error
    emc.infuse_error(user_action) #infuse error cho 'user_action' ở 'EMC' để đưa tới 'state tracker' update state user
    # And update state tracker
    state_tracker.update_state_user(user_action) #update 'state tracker' với đầu vào là 'user_action' with error từ 'EMC'
    # Finally, reset agent
    dqn_agent.reset()#reset slot index hiện tại, đặt lại các biến dựa trên rule-based

#Chạy warmup và train
warmup_run()
train_run()
suc.close()
avg.close()
