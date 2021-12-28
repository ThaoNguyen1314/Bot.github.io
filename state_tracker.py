from db_query import DBQuery
import numpy as np
from utils import convert_list_to_dict
from dialogue_config import all_intents, all_slots, usersim_default_key
import copy


class StateTracker:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent.
    Theo dõi state của tập / hội thoại và chuẩn bị biểu diễn trạng thái cho agent."""

    def __init__(self, database, constants):
        """
        The constructor of StateTracker.
        Hàm khởi tạo của StateTracker.

        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.
        Hàm tạo của StateTracker tạo đối tượng truy vấn DB, tạo dict đại diện trạng thái cần thiết, v.v. và đặt lại cuộc gọi.

        Parameters:
            database (dict): The database with format dict(long: dict)
            constants (dict): Loaded constants in dict

        """

        self.db_helper = DBQuery(database)
        self.match_key = usersim_default_key
        self.intents_dict = convert_list_to_dict(all_intents)
        self.num_intents = len(all_intents)
        self.slots_dict = convert_list_to_dict(all_slots)
        self.num_slots = len(all_slots)
        self.max_round_num = constants['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())
        self.reset()

    def get_state_size(self):
        """Returns the state size of the state representation used by the agent."""
        # Trả về kích thước trạng thái của biểu diễn trạng thái được agent sử dụng.

        return 2 * self.num_intents + 7 * self.num_slots + 3 + self.max_round_num

    def reset(self):
        """Resets current_informs, history and round_num."""
        # Đặt lại current_informs, history và round_num

        self.current_informs = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        # Danh sách các cuộc đối thoại (phân đoạn) của agent và user cho đến nay trong cuộc trò chuyện
        self.history = []
        self.round_num = 0

    def print_history(self):
        """Helper function if you want to see the current history action by action."""
        """Hàm trợ giúp nếu bạn muốn xem lịch sử hiện tại của từng hành động."""
        for action in self.history:
            print(action)

    def get_state(self, done=False):
        """
        Returns the state representation as a numpy array which is fed into the agent's neural network.
        Trả về biểu diễn trạng thái dưới dạng một mảng numpy được đưa vào mạng nơ-ron của agent.

        The state representation contains useful information for the agent about the current state of the conversation.
        Biểu diễn trạng thái chứa thông tin hữu ích cho agent về trạng thái hiện tại của cuộc hội thoại.
        Processes by the agent to be fed into the neural network. Ripe for experimentation and optimization.
        Các tiến trình của agent sẽ được đưa vào mạng nơ-ron. Đã chín muồi (đã đc đào tạo hoàn thiện) để thử nghiệm và tối ưu hóa.

        Parameters:
            done (bool): Indicates whether this is the last dialogue in the episode/conversation. Default: False
                         Cho biết đây có phải là đoạn hội thoại cuối cùng trong tập / cuộc hội thoại hay không. Mặc định: Sai
                         
        Returns:
            numpy.array: A numpy array of shape (state size,)
                         

        """

        # If done then fill state with zeros
        # Nếu done = True thì điền trạng thái bằng không
        if done:
            return self.none_state

        user_action = self.history[-1] # user action là hành động cuối cùng trong lịch sử
        # Nhận thông tin database hữu ích cho agent
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs)
        # hành động cuối của agent là hành động thứ 2 tính từ cuối list lịch sử nếu list dài hơn 1
        last_agent_action = self.history[-2] if len(self.history) > 1 else None


        # Create one-hot of intents to represent the current user action
        # Tạo một nhóm các ý định để biểu diễn hành động hiện tại của người dùng
        user_act_rep = np.zeros((self.num_intents,)) # self.num_intents là tổng số tất cả các ý định khác nhau được xác định trong config.py
        # self.intents_dict là một dict có các key của các ý định và giá trị của chỉ mục của chúng trong danh sách các ý định
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0 
        # giá trị tại chỉ số của ý định của user action trong mảng biểu diễn user action = 1

        # Create bag of inform slots representation to represent the current user action
        # Tạo túi biểu diễn inform slot để biểu diễn hành động người dùng hiện tại 
        user_inform_slots_rep = np.zeros((self.num_slots,)) 
        for key in user_action['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0 # self.slots_dict giống như self.intents_dict ngoại trừ việc có slot
            # giá trị tại chỉ số của key của hành động inform của user trong mảng biểu diễn user inform = 1
            # vd: key = city => value của 'city' trong slots_dict = 3(tra file config.py) => user_inform_slots_rep[3] = 1.0

        # Create bag of request slots representation to represent the current user action
         # Tạo túi biểu diễn request slot để biểu diễn hành động người dùng hiện tại 
        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['request_slots'].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0
            # giá trị tại chỉ số của key của hành động request của user trong mảng biểu diễn user request = 1


        # Create bag of filled_in slots based on the current_slots
        # Tạo túi của filled_in slots dựa trên current_slots
        current_slots_rep = np.zeros((self.num_slots,))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0
            # giá trị tại chỉ số của key của current inform trong mảng biểu diễn current slot = 1


        # Encode last agent intent
        # Mã hóa ý định agent cuối cùng
        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action['intent']]] = 1.0

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['inform_slots'].keys():
                agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['request_slots'].keys():
                agent_request_slots_rep[self.slots_dict[key]] = 1.0

        # Value representation of the round num
        # Biểu diễn giá trị của số vòng
        turn_rep = np.zeros((1,)) + self.round_num / 5.

        # One-hot representation of the round num
        # https://machinelearningcoban.com/tabml_book/ch_data_processing/onehot.html (định nghĩa one-hot)
        turn_onehot_rep = np.zeros((self.max_round_num,))
        turn_onehot_rep[self.round_num - 1] = 1.0

        # Representation of DB query results (scaled counts)
        # Biểu diễn kết quả truy vấn DB (số lượng theo tỷ lệ)
        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict['matching_all_constraints'] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        # Representation of DB query results (binary)
        # Biểu diễn kết quả truy vấn DB (nhị phân)
        kb_binary_rep = np.zeros((self.num_slots + 1,)) + np.sum(db_results_dict['matching_all_constraints'] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = np.sum(db_results_dict[key] > 0.)

        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep]).flatten()  #gộp các phần tử cùng chiều và ghép các mảng vs nhau

        return state_representation

    def update_state_agent(self, agent_action):
        """
        Updates the dialogue history with the agent's action and augments the agent's action.
        Cập nhật lịch sử đối thoại với hành động của agent và tăng cường hành động của agent.

        Takes an agent action and updates the history. Also augments the agent_action param with query information and
        any other necessary information.
        Thực hiện agent action và cập nhật lịch sử. Đồng thời tăng cường thông số agent_action với thông tin truy vấn và 
        bất kỳ thông tin cần thiết nào khác.

        Parameters:
            agent_action (dict): The agent action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'Agent')

        """

        # Xử lý inform
        if agent_action['intent'] == 'inform':
        # Điền vào inform slot (ban đầu có giá trị là ‘PLACEHOLDER’) của action bằng cách truy vấn cơ sở dữ liệu sử dụng current inform làm ràng buộc
        # db_helper là một đối tượng của lớp DBQuery
            assert agent_action['inform_slots'] # nếu value của khoá inform_slot là rỗng, ý định là thông báo nhưng lại k có dữ liệu thì báo lỗi.
            inform_slots = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs) #tìm kiêm thông tin phù hợp
            agent_action['inform_slots'] = inform_slots 
            assert agent_action['inform_slots']
            key, value = list(agent_action['inform_slots'].items())[0]  # Only one
            assert key != 'match_found'  
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key) # nếu giá trị k đc cập nhật thì báo lỗi
            self.current_informs[key] = value # Cập nhật current inform với inform slot đã điền
        # If intent is match_found then fill the action informs with the matches informs (if there is a match)
        # Nếu ý định là match_found thì điền vào action informs với thông tin về các match (nếu có sự trùng khớp)
        elif agent_action['intent'] == 'match_found':
            # Nhận danh sách các vé từ database trong đó các slot của mỗi vé khớp với các slot (cả khóa VÀ giá trị) của inform ràng buộc hiện tại
            assert not agent_action['inform_slots'], 'Cannot inform and have intent of match found!'
            db_results = self.db_helper.get_db_results(self.current_informs)
            # Nếu có các vé phù hợp thì hãy đặt các inform slot của agent action thành các slot của một vé từ danh sách db_result. 
            # Tạo và đặt giá trị của self.match_key trong các inform slot về action agent thành ID của vé này
            if db_results:
                # Arbitrarily pick the first value of the dict
                # Tùy ý chọn giá trị đầu tiên của dict
                key, value = list(db_results.items())[0]
                agent_action['inform_slots'] = copy.deepcopy(value)
                agent_action['inform_slots'][self.match_key] = str(key)
            else:
                agent_action['inform_slots'][self.match_key] = 'no match available' # Nếu ko có vé thì self.match_key = "no match available"
            # Cập nhật giá trị của self.match_key trong current inform với giá trị mới được tìm thấy ở trên
            self.current_informs[self.match_key] = agent_action['inform_slots'][self.match_key] 
        agent_action.update({'round': self.round_num, 'speaker': 'Agent'}) # Thêm số vòng
        self.history.append(agent_action) # Thêm lịch sử

    def update_state_user(self, user_action):
        """
        Updates the dialogue history with the user's action and augments the user's action.
        Cập nhật lịch sử đối thoại với hành động của người dùng và tăng cường hành động của người dùng.

        Takes a user action and updates the history. Also augments the user_action param with necessary information.
        Thực hiện user action và cập nhật lịch sử. Đồng thời bổ sung thêm thông tin cần thiết cho tham số user_action.

        Parameters:
            user_action (dict): The user action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'User')

        """
        # Cập nhật thông tin hiện tại với bất kỳ inform slot nào trong action
        for key, value in user_action['inform_slots'].items():
            self.current_informs[key] = value
        user_action.update({'round': self.round_num, 'speaker': 'User'}) # cập nhật thêm thông tin vào dict user_action 
        # Thêm action vào lịch sử
        self.history.append(user_action)
        # Tăng dần số vòng (end of current round)
        self.round_num += 1
