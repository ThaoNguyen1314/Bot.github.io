from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions #import từ dialogue_config.py
import re


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either


class DQNAgent:
    """The DQN agent that interacts with the user.""" #Tác nhân DQN tương tác với người dùng

    def __init__(self, state_size, constants):
        """
        The constructor of DQNAgent. #Hàm cấu tạo DQNAgent.

        The constructor of DQNAgent which saves constants, sets up neural network graphs, etc.
        #Hàm cấu tạo DQNAgent giúp lưu các hằng số, thiết lập đồ thị mạng nơ-ron, v.v.

        Parameters: #thông số
            state_size (int): The state representation size or length of numpy array
            #Kích thước biểu diễn trạng thái hoặc chiều dài của mảng numpy
            constants (dict): Loaded constants in dict
            #Hằng số đã tải trong dict

        """

        self.C = constants['agent']# gán self.C cho constants với file 'agent' trong constants.json
        self.memory = []#Khởi tạo mảng rỗng cho memory
        self.memory_index = 0 #set memory_index=0
        self.max_memory_size = self.C['max_mem_size']#Gán 'max_mem_size' trong 'agent' cho self.max_memory_size
        self.eps = self.C['epsilon_init']#Gán 'epsilon_init' trong 'agent' cho self.eps 
        self.vanilla = self.C['vanilla']#Gán 'vanilla' trong 'agent' cho self.vanilla
        self.lr = self.C['learning_rate']#Gán 'learning_rate' trong 'agent' cho self.lr 
        self.gamma = self.C['gamma']#tương tự 
        self.batch_size = self.C['batch_size']#tương tự 
        self.hidden_size = self.C['dqn_hidden_size']#tương tự 

        self.load_weights_file_path = self.C['load_weights_file_path']#tương tự 
        self.save_weights_file_path = self.C['save_weights_file_path']#tương tự 

        if self.max_memory_size < self.batch_size: #Xử lý ngoại lệ
            raise ValueError('Max memory size must be at least as great as batch size!') #Max_memory_size ít nhất phải lớn bằng batch_size

        self.state_size = state_size# khởi tạo self.state_size
        self.possible_actions = agent_actions
        #khởi tạo self.possible_actions-là các actions possible lấy từ agent_actions của dialogue_config.py:
        # " {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
        #   {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}"
        self.num_actions = len(self.possible_actions)#khởi tạo  self.num_action bằng số lượng kí tự của chuỗi self.possible_actions 

        self.rule_request_set = rule_requests
        #khởi tạo self.rule_request_set-là các request follow theo policy lấy từ rule_requests của dialogue_config.py:
        #['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

        self.beh_model = self._build_model()
        #khởi tạo self.beh_model (behavior model) với self._build_model()- Là hàm builds và returns Neural network cho DQNmodel.
        self.tar_model = self._build_model()
        #khởi tạo self.tar_model (target model) với self._build_model()- Là hàm builds và returns Neural network cho DQNmodel.
        self._load_weights()
        #Gọi hàm self._load_weights() có tác dụng tải load weights của cả 2 model từ 2 files .h5
        self.reset()
        #Gọi hàm self.reset() có tác dụng reset slot index hiện tại, đặt lại các biến dựa trên quy tắc
    def _build_model(self):# hàm builds và returns Neural network cho DQNmodel
        """Builds and returns model/graph of neural network."""

        model = Sequential()#Tạo sequential model
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        #Dense ( ): Layer này cũng như một layer neural network bình thường, với các tham số sau:
        #self.hidden_size: số neurons ở first hidden layer
        #input_dim(số tham số đầu vào)= self.state_size
        #hàm activation : chọn activation như linear, softmax, relu, tanh, sigmoid. Đặc điểm mỗi hàm có thể search thêm để biết cụ thể nó ntn
        #'relu' là một hàm kích hoạt phi tuyến tính được sử dụng trong mạng nơ-ron nhiều lớp hoặc mạng nơ-ron sâu.
        model.add(Dense(self.num_actions, activation='linear'))
        #self.num_actions là số neurols ở lớp đầu ra
        # activation 'linear' chỉ được sử dụng ở một nơi, tức là lớp đầu ra. (kích hoạt tuyến tính ở lớp đầu ra, neural network phải có hàm 'relu' tại các lớp ẩn)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        #Hàm compile: Ở hàm này chúng ta sử dụng để training models như thuật toán train qua optimizer như Adam, SGD, RMSprop,..
        #hàm loss có nhiều cách chọn như:mean_squared_eror: thường dùng trong regression tính theo eculid; 
        #                                mean_absolute_error : để tính giá trị tuyệt đối
        #                                binary_crossentropy : dùng cho classifier 2 class
        #                                categorical_crossentropy : dùng classifier nhiều class
        #learning_rate : dạng float , tốc độc học, chọn phù hợp để hàm số hội tụ nhanh.
        return model

    def reset(self): #hàm reset
        """Resets the rule-based variables."""
        #reset slot index hiện tại, đặt lại các biến dựa trên quy tắc
        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False):#Hàm get_action
        """
        Returns the action of the agent given a state.
        #Trả về action của agent có được trong một state.
        Gets the action of the agent given the current state. Either the rule-based policy or the neural networks are
        used to respond.
        # Nhận action của agent với state hiện tại. Phản hồi dựa trên quy tắc của policy hoặc neural networks được sử dụng.
        Parameters:
            state (numpy.array): The database with format dict(long: dict)
            #state (kiểu numpy.array): Tạo mảng với kiểu dữ liệu dict

            use_rule (bool): Indicates whether or not to use the rule-based policy, which depends on if this was called
                             in warmup or training. Default: False
            #ues_rule( kiểu bool):Cho biết có nên sử dụng chính sách dựa trên quy tắc hay không, 
            tùy thuộc vào việc điều này được gọi trong quá trình khởi động hay đào tạo. Mặc định: False

        Returns:#Trả về
            int: The index of the action in the possible actions 
            #Kiểu int: index của action trong possible actions.
            dict: The action/response itself
            #Kiểu dict: response chính action đó

        """

        if self.eps > random.random():#Nếu self.eps(ban đầu đc khởi tạo=0)>[0,1]
            index = random.randint(0, self.num_actions - 1)#index= random[0,self.num_actions - 1] (kiểu int)
            action = self._map_index_to_action(index)# Gán index cho một action trong possible actions
            return index, action
        else:
            if use_rule: #Nếu use_rule = TRUE
                return self._rule_action() # Trả về index của action(int) và action/response đó(dict) dựa trên quy tắc của policy
            else:
                return self._dqn_action(state) # Trả về index của action(int) và action/response đó(dict) dựa trên model neural networks được sử dụng

    def _rule_action(self):#Hàm rule_action
        #Trả về index của action(int) và action/response đó(dict) dựa trên quy tắc của policy
        """
        
        Returns a rule-based policy action.

        Selects the next action of a simple rule-based policy.

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """

        if self.rule_current_slot_index < len(self.rule_request_set):
        #Nếu index của rule_current_slot (index của slot hiện tại trong dãy theo quy tắc của policy) < kích thước thiết lập của dãy request theo policy
            slot = self.rule_request_set[self.rule_current_slot_index] # slot sẽ được gán = value trong dãy request có index = self.rule_current_slot_index
            self.rule_current_slot_index += 1# Tăng cho đến khi được gán hết các value trong self.rule_request_set (dãy request theo policy)
            rule_response = {'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}}#gán dãy rule_respone( phản hồi theo policy)
        elif self.rule_phase == 'not done': #Nếu self.rule_phase (giai đoạn của policy) = 'not done'
            rule_response = {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}#gán dãy rule_respone( phản hồi theo policy)
            self.rule_phase = 'done'
        elif self.rule_phase == 'done':#Nếu self.rule_phase (giai đoạn của policy) = 'done'
            rule_response = {'intent': 'done', 'inform_slots': {}, 'request_slots': {}}
        else:
            raise Exception('Should not have reached this clause')# Xử lý ngoại lệ: print ('Should not have reached this clause')

        index = self._map_action_to_index(rule_response)#Lấy index từ action trong possible actions đưa vào
        return index, rule_response

    def _map_action_to_index(self, response):
        #Lấy index từ action trong possible actions đưa vào
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions): #Hàm enumerate thêm vào self.possible_actions 1 bộ đếm, không thêm star thì mặc định từ 0)
            #chạy vòng for i từ 0
            if response == action: #Nếu response là action
                return i#Trả về index
        raise ValueError('Response: {} not found in possible actions'.format(response)) 
        #Xử lý ngoại lệ: không tìm thấy trong possile action, print: ...

    def _dqn_action(self, state):#Hàm _dqn_action
        # Trả về index của action(int) và action/response đó(dict) dựa trên model neural networks được sử dụng (target model hoặc behavior model)
        """
        Returns a behavior model output given a state.#

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """

        index = np.argmax(self._dqn_predict_one(state)) 
        # sử dụng np.argmax Gán index cho giá trị lớn nhất trong mảng các giá trị tỉ lệ của state đã được reshape bằng hàm _dqn_predict_one 
        action = self._map_index_to_action(index) # lấy action từ index bằng hàm _map_index_to_action (action có tỉ lệ cao nhất)
        return index, action

    def _map_index_to_action(self, index):
        #Trả về action(dict) trong possible actions từ index.
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):#Chạy vòng for sử dụng bộ đếm enumerate cho self.possible_actions
            if index == i: # Nếu index =i
                return copy.deepcopy(action)
                #Sử dụng module "copy" để dùng hàm deepcopy cho việc trả về một action được deepcopy. Điều này có nghĩa action bản sao có thể thay đổi thỏai mái
                #nhưng không ảnh hưởng tới action original.
        raise ValueError('Index: {} not in range of possible actions'.format(index))
        #Xử lý trường hợp ngoại lệ không có index
    def _dqn_predict_one(self, state, target=False):
        #Trả về mảng các giá trị tỉ lệ của state đã được reshape lại.
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()#dùng hàm reshape mảng các giá trị của state [1,self.state_size]

    def _dqn_predict(self, states, target=False):
        #Trả về mảng các giá trị tỉ lệ từ [0-1] của state, với target mặc định là False
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target: #nếu target = True
            return self.tar_model.predict(states) #Trả về mảng các giá trị của state trong model target với hàm .predict() sử dụng để kiểm tra mô hình
        else:
            return self.beh_model.predict(states)#Trả về mảng các giá trị của state trong model bahavior với hàm .predict() sử dụng để kiểm tra mô hình

    def add_experience(self, state, action, reward, next_state, done):
        #Thêm experience được tạo từ state, action, reward, next_state, done(parameters) vào bộ nhớ
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            state (numpy.array)
            action (int)
            reward (int)
            next_state (numpy.array)
            done (bool)

        """

        if len(self.memory) < self.max_memory_size: # Nếu kích thước của self.memory < self.max_memory_size
            self.memory.append(None)# Thêm None vào cuối self.memory
        self.memory[self.memory_index] = (state, action, reward, next_state, done)#thêm experience
        self.memory_index = (self.memory_index + 1) % self.max_memory_size 
        #Gán self.memory_index bằng cách tăng dần self.memory_index thêm 1 sau mỗi lần thêm experience và thực hiện chia lấy dư cho self.max_memory_size
        #để lặp lại việc thêm add experience khi self.memory_index vượt quá self.max_memory_size 

    def empty_memory(self):
        #Hàm làm trống bộ nhớ và đặt lại index cho bộ nhớ
        """Empties the memory and resets the memory index."""

        self.memory = []#basic
        self.memory_index = 0#basic

    def is_memory_full(self):
        #Hàm kiểm tra kích thước của bộ nhớ, trả về true khi bộ nhớ đầy
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size #basic

    def train(self):
        #Hàm train agent bằng cách cải thiện behavior model với bộ nhớ.
        # Lấy các batches bộ nhớ từ vùng bộ nhớ và xử lý chúng.
        # Quá trình lấy các tuples và xếp chúng ở định dạng chính xác cho Neural Network và tính toán phương trình Bellman cho Q-learning.
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        # Calc. num of batches to run
        #Tính toán số batches để chạy
        num_batches = len(self.memory) // self.batch_size 
        #gán num_batches bằng kích thước của (self.memory) chia làm tròn xuống ("//") cho self.batch_size
        for b in range(num_batches): #Chạy vòng for với b trong khoảng num_batches
            batch = random.sample(self.memory, self.batch_size)
            #Sử dụng hàm sample() của module "random" gán batch thành 1 list chứa "self.batch_size" phần tử ngẫu nhiên 
            # trong số phần tử của list self.memory
            states = np.array([sample[0] for sample in batch])
            #Dùng hàm array của module "numpy" để tạo mảng cho state bằng vòng for với sample (chỉ lấy đến sample[0]) trong list batch.
            #Mảng của state lúc đó là phần tử ứng với sample[0] trong list batch
            next_states = np.array([sample[3] for sample in batch])
            #Tương tự với state, Mảng của next_states lúc đó là phần tử ứng với sample[3] trong list batch
            assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            assert next_states.shape == states.shape #
            #Hàm assert sử dụng để thêm điều kiện chạy chương trình, nếu điều kiện TRUE thì chương trình được cho phép.
            #Nếu điều kiện FALSE chương trình sẽ dừng.
            beh_state_preds = self._dqn_predict(states)  # For leveling error
            #Gán beh_state_preds là mảng các giá trị tỉ lệ từ [0-1] của state trong model behavior
            if not self.vanilla:#Nếu vanilla != TRUE (kích hoạt vanilla method hay không)
                beh_next_states_preds = self._dqn_predict(next_states)  # For indexing for DDQN
                #Gán beh_next_states_preds là mảng các giá trị tỉ lệ từ [0-1] của next_states trong model behavior
            tar_next_state_preds = self._dqn_predict(next_states, target=True)  # For target value for DQN (& DDQN)
            #Gán tar_next_state_preds là mảng các giá trị tỉ lệ từ [0-1] của next_states trong model target (vì target=TRUE)
            inputs = np.zeros((self.batch_size, self.state_size))
            #Tạo ma trận inputs [self.batch_size x self.state_size]
            targets = np.zeros((self.batch_size, self.num_actions))
            #Tạo ma trận targets [self.batch_size x self.num_actions]

            for i, (s, a, r, s_, d) in enumerate(batch):
            #Chạy vòng for với i và các (s, a, r, s_, d) ánh xạ bởi i trong list batch với bộ đếm enumerate.
                t = beh_state_preds[i] #Gán t cho beh_state_preds[i] (beh_state_preds đã được định nghĩa ở trên)
                if not self.vanilla: #Nếu self.vanilla != True (kích hoạt vanilla method hay không)
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                    #cập nhật giá trị t theo a (action) được ánh xạ từ i
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)
                    #cập nhật giá trị t theo a (action) được ánh xạ từ i

                inputs[i] = s #inputs là mảng i và s (ánh xạ s trong batch từ i)
                targets[i] = t #targets là mảng i và a (ánh xạ a trong batch từ i)

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)
            #Fit là hàm train beh_model với số lượng epochs cố định (các lần lặp lại trên một tập data (inputs, targets, epochs=1, verbose=0))

    def copy(self):
        #Hàm sao chép trọng số của behavior model vào trọng số của target model
        """Copies the behavior model's weights into the target model's weights."""

        self.tar_model.set_weights(self.beh_model.get_weights())
        #Dùng hàm get_weights và set_weights của module "Keras" cho việc lấy trọng số của behavior model cho target model

    def save_weights(self):
        #Chỉ đơn giản là hàm lưu trọng số của cả 2 models vào 2 files h5
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        #Chỉ đơn giản là load trọng số của hai models từ 2 files h5
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
