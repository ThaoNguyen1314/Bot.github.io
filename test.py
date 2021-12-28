from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json
from user import User
from utils import remove_empty_slots


if __name__ == "__main__":
    '''
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python test.py --constants_path "constants.json"
        khi thực thi file test.py thì sẽ thực thi file constant.json và truyền vào nó dưới dạng hằng số.
    # 2) Run this file as is
    '''
    #thư viện argparse dùng để xử lý các tham số được truyền vào từ command - line
    parser = argparse.ArgumentParser()   #tạo một instance argparse
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='') #xác định cách tham số được xử lý
    args = parser.parse_args() #chuyển các tham số của cm-line thành obj
    params = vars(args) #chuyển obj thành kiểu Dictionary 

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:   #nếu tham số được truyền vào từ command-line
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH  #nếu ko đọc file bằng luôn đường dẫn

    with open(constants_file) as f:
        constants = json.load(f)

    # Load file path constants 
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    NUM_EP_TEST = run_dict['num_ep_run']
    MAX_ROUND_NUM = run_dict['max_round_num']

    #load các file pickle
    # Load movie DB
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    # Load goal file
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects _ Khởi tạo 
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)


def test_run():
    """
    Runs the loop that tests the agent.

    Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
    reaches NUM_EP_TEST. 
    Kiểm tra tác nhân về nhiệm vụ chatbot định hướng mục tiêu. Chỉ để đánh giá một tác tử đã được đào tạo. Kết thúc khi tập
    đạt đến NUM_EP_TEST.

    """

    print('Testing Started...')
    episode = 0
    # while episode < NUM_EP_TEST:
    while episode < 2:
        episode_reset()  #Reset các giá trị như hàm ở dưới
        episode += 1
        ep_reward = 0
        done = False
        # Get initial state from state tracker
        state = state_tracker.get_state()
        print("Agent: State: ", state)
        while not done:
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = dqn_agent.get_action(state) #action tác tử
            print("Agent: Action", agent_action, "\n")
            # Update state tracker with the agent's action
            state_tracker.update_state_agent(agent_action) #state tác tử
            # User takes action given agent action
            user_action, reward, done, success = user.step(agent_action) #tính toán thông tin của user sim

            print("User sim: Action", user_action, "\n================")
            ep_reward += reward
            if not done:
                # Infuse error into semantic frame level of user action
                emc.infuse_error(user_action)
            # Update state tracker with user action
            state_tracker.update_state_user(user_action)
            # Grab "next state" as state
            state = state_tracker.get_state(done)
        print('Episode: {} Success: {} Reward: {}'.format(episode, success, ep_reward))
    print('...Testing Ended')


def episode_reset():
    """Resets the episode/conversation in the testing loop."""

    # First reset the state tracker
    state_tracker.reset()
    # Then pick an init user action
    user_action = user.reset()
    # Infuse with error
    emc.infuse_error(user_action)
    # And update state tracker
    state_tracker.update_state_user(user_action)
    # Finally, reset agent
    dqn_agent.reset()


test_run()
