from dialogue_config import usersim_default_key, FAIL, NO_OUTCOME, SUCCESS, usersim_required_init_inform_keys, \
    no_query_keys
from utils import reward_function
import random, copy


class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, goal_list, constants, database):
        """
        Hàm khởi tạo giá trị ban đầu cho một user sim

        Parameters:
            goal_list (list): User goals loaded from file, mục tiêu người dùng
            constants (dict): Dict of constants loaded from file
            database (dict): The database in the format dict(long: dict)
        """

        self.goal_list = goal_list
        self.max_round = constants['run']['max_round_num']
        self.default_key = usersim_default_key # từ khóa mặc đinh là 'ticket'
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = usersim_required_init_inform_keys  #init_Inform = 'moviename', khởi tạo mặc định thông tin 'moviename'
        self.no_query = no_query_keys

        # TEMP ----
        self.database = database
        # ---------
        
        #print("Init ====> Goal_list: ", self.goal_list, "\n Init Inform", self.init_informs)

    def reset(self):
        """
        Chọn một mục tiêu ngẫu nhiên
        Add default slot (“ticket”) 
        Xóa tất cả các khía cạnh trạng thái
        Thêm tất cả các request slots và inform vào rest_slots
        Khởi tạo kiểm tra ràng buộc thất bại (được thảo luận bên dưới các ràng buộc thành công)
        Trả về  hành động ban đầu của tập

        """

        self.goal = random.choice(self.goal_list) #chọn ngẫu nhiên trong database
        # Add default slot to requests of goal
        self.goal['request_slots'][self.default_key] = 'UNK' #khởi tạo thông tin 'ticket' = 'UNK'
        
        #______________________ trạng thái nội bộ của user sim, khởi tạo các giá trị rỗng
        self.state = {} 
        # Add all inform slots informed by agent or user sim to this dict 

        
        
        self.state['history_slots'] = {}
        # Any inform slots for the current user sim action, empty at start of turn
        self.state['inform_slots'] = {}
        # Current request slots the user sim wants to request
        self.state['request_slots'] = {}

        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state['rest_slots'] = {}
        self.state['rest_slots'].update(self.goal['inform_slots'])
        self.state['rest_slots'].update(self.goal['request_slots'])
        self.state['intent'] = ''
        ###############################


        # False for failure, true for success, init. to failure
        self.constraint_check = FAIL
        
        #print("Reset", "\n State_Reset: ",self.state, "\n Goal_reset", self.goal)
        return self._return_init_action()

    def _return_init_action(self):
        """
        
        Trả về hành động khởi tạo cho tập
        
        Init Action chứa: 'intent': 'request', một 'request_slots' có gt là 'UNK' và giá trị cung cấp trc 'inform_slots'.
        VD: Uses response:  {'intent': 'request', 'request_slots': {'theater': 'UNK'}, 'inform_slots': {'moviename': 'first'}}
        Returns:
            dict: Initial user response
        """

        # Always request
        self.state['intent'] = 'request'

        if self.goal['inform_slots']:
            # Pick all the required init. informs, and add if they exist in goal inform slots
            for inform_key in self.init_informs:
                if inform_key in self.goal['inform_slots']:
                    self.state['inform_slots'][inform_key] = self.goal['inform_slots'][inform_key]
                    self.state['rest_slots'].pop(inform_key)
                    self.state['history_slots'][inform_key] = self.goal['inform_slots'][inform_key]
            # If nothing was added then pick a random one to add
            if not self.state['inform_slots']:
                key, value = random.choice(list(self.goal['inform_slots'].items()))
                self.state['inform_slots'][key] = value
                self.state['rest_slots'].pop(key)
                self.state['history_slots'][key] = value

        # Now add a request, do a random one if something other than def. available
        self.goal['request_slots'].pop(self.default_key)
        if self.goal['request_slots']:
            req_key = random.choice(list(self.goal['request_slots'].keys()))
        else:
            req_key = self.default_key
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state['request_slots'][req_key] = 'UNK'

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])

        #print("Uses response", user_response)
        return user_response

    def step(self, agent_action):
        """
        1.Xóa dữ liệu khi chuyển sang round khác
        2.Nếu quá max round thì gán intent = “done” và success = FAIL
        3.Nếu không thì tạo ra một hành động dựa trên hành động của tác tử. Nếu intent là “done” thì tính toán xem tập có được tính toán là thành công hay ko và trả là là “done”
        4.phản hồi theo intent, request, inform của tác tử
        5.tính phần thưởng cho step đó
        6.Trả lại phản hồi, phần thưởng, bool đã hoàn thành và bool cho biết thành công
        success = “NO_OUTCOME” là ko đc hoàn thành

        """

        # Assertions -----
        # No UNK in agent action informs
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            assert value != 'PLACEHOLDER'
        # No PLACEHOLDER in agent at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'
        # ----------------

        self.state['inform_slots'].clear()
        self.state['intent'] = ''

        done = False
        success = NO_OUTCOME
        # First check round num, if equal to max then fail
        if agent_action['round'] == self.max_round:
            done = True
            success = FAIL
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            agent_intent = agent_action['intent']
            if agent_intent == 'request':
                self._response_to_request(agent_action)
            elif agent_intent == 'inform':
                self._response_to_inform(agent_action)
            elif agent_intent == 'match_found':
                self._response_to_match_found(agent_action)
            elif agent_intent == 'done':
                success = self._response_to_done()
                self.state['intent'] = 'done'
                self.state['request_slots'].clear()
                done = True

        # Assumptions -------
        # If request intent, then make sure request slots
        if self.state['intent'] == 'request':
            assert self.state['request_slots']
        # If inform intent, then make sure inform slots and NO request slots
        if self.state['intent'] == 'inform':
            assert self.state['inform_slots']
            assert not self.state['request_slots']
        assert 'UNK' not in self.state['inform_slots'].values()
        assert 'PLACEHOLDER' not in self.state['request_slots'].values()
        # No overlap between rest and hist
        for key in self.state['rest_slots']:
            assert key not in self.state['history_slots']
        for key in self.state['history_slots']:
            assert key not in self.state['rest_slots']
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal['inform_slots']:
            assert self.state['history_slots'].get(inf_key, False) or self.state['rest_slots'].get(inf_key, False)
        for req_key in self.goal['request_slots']:
            assert self.state['history_slots'].get(req_key, False) or self.state['rest_slots'].get(req_key,
                                                                                                   False), req_key
        # Anything in the rest should be in the goal
        for key in self.state['rest_slots']:
            assert self.goal['inform_slots'].get(key, False) or self.goal['request_slots'].get(key, False)
        assert self.state['intent'] != ''
        # -----------------------

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])

        reward = reward_function(success, self.max_round)
        # print("Step \n User Respones", user_response, "Reward", reward)
        return user_response, reward, done, True if success is 1 else False

    def _response_to_request(self, agent_action):
        """
        
        Phản hồi với intent của tác tử là 'request'
        There are 4 main cases for responding.

        Parameters:
            agent_action (dict): Intent of request with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_request_key = list(agent_action['request_slots'].keys())[0]
        # First Case: if agent requests for something that is in the user sims goal inform slots, then inform it
        if agent_request_key in self.goal['inform_slots']:
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
            self.state['request_slots'].clear()
            self.state['rest_slots'].pop(agent_request_key, None)
            self.state['history_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
        # Second Case: if the agent requests for something in user sims goal request slots and it has already been
        # informed, then inform it
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['history_slots']:
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.state['history_slots'][agent_request_key]
            self.state['request_slots'].clear()
            assert agent_request_key not in self.state['rest_slots']
        # Third Case: if the agent requests for something in the user sims goal request slots and it HASN'T been
        # informed, then request it with a random inform
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['rest_slots']:
            self.state['request_slots'].clear()
            self.state['intent'] = 'request'
            self.state['request_slots'][agent_request_key] = 'UNK'
            rest_informs = {}
            for key, value in list(self.state['rest_slots'].items()):
                if value != 'UNK':
                    rest_informs[key] = value
            if rest_informs:
                key_choice, value_choice = random.choice(list(rest_informs.items()))
                self.state['inform_slots'][key_choice] = value_choice
                self.state['rest_slots'].pop(key_choice)
                self.state['history_slots'][key_choice] = value_choice
        # Fourth and Final Case: otherwise the user sim does not care about the slot being requested, then inform
        # 'anything' as the value of the requested slot
        else:
            assert agent_request_key not in self.state['rest_slots']
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = 'anything'
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_request_key] = 'anything'

        # print("_response_to_request, state: ", self.state)
    def _response_to_inform(self, agent_action):
        """
        Phản hồi với intent của tác tử là 'inform'
        There are 2 main cases for responding. Add the agent inform slots to history slots,
        and remove the agent inform slots from the rest and request slots.

        Parameters:
            agent_action (dict): Intent of inform with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_inform_key = list(agent_action['inform_slots'].keys())[0]
        agent_inform_value = agent_action['inform_slots'][agent_inform_key]

        assert agent_inform_key != self.default_key

        # Add all informs (by agent too) to hist slots
        self.state['history_slots'][agent_inform_key] = agent_inform_value
        # Remove from rest slots if in it
        self.state['rest_slots'].pop(agent_inform_key, None)
        # Remove from request slots if in it
        self.state['request_slots'].pop(agent_inform_key, None)

        # First Case: If agent informs something that is in goal informs and the value it informed doesnt match,
        # then inform the correct value
        if agent_inform_value != self.goal['inform_slots'].get(agent_inform_key, agent_inform_value):
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
        # Second Case: Otherwise pick a random action to take
        else:
            # - If anything in state requests then request it
            if self.state['request_slots']:
                self.state['intent'] = 'request'
            # - Else if something to say in rest slots, pick something
            elif self.state['rest_slots']:
                def_in = self.state['rest_slots'].pop(self.default_key, False)
                if self.state['rest_slots']:
                    key, value = random.choice(list(self.state['rest_slots'].items()))
                    if value != 'UNK':
                        self.state['intent'] = 'inform'
                        self.state['inform_slots'][key] = value
                        self.state['rest_slots'].pop(key)
                        self.state['history_slots'][key] = value
                    else:
                        self.state['intent'] = 'request'
                        self.state['request_slots'][key] = 'UNK'
                else:
                    self.state['intent'] = 'request'
                    self.state['request_slots'][self.default_key] = 'UNK'
                if def_in == 'UNK':
                    self.state['rest_slots'][self.default_key] = 'UNK'
            # - Otherwise respond with 'nothing to say' intent
            else:
                self.state['intent'] = 'thanks'
        # print("_response_to_inf, state: ", self.state)
    def _response_to_match_found(self, agent_action):
        """
        Phản hồi với intent của tác tử là  'match_found'.

        Check if there is a match in the agent action that works with the current goal.

        Parameters:
            agent_action (dict): Intent of match_found with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_informs = agent_action['inform_slots']

        self.state['intent'] = 'thanks'
        self.constraint_check = SUCCESS

        assert self.default_key in agent_informs
        self.state['rest_slots'].pop(self.default_key, None)
        self.state['history_slots'][self.default_key] = str(agent_informs[self.default_key])
        self.state['request_slots'].pop(self.default_key, None)

        if agent_informs[self.default_key] == 'no match available':
            self.constraint_check = FAIL

        # Check to see if all goal informs are in the agent informs, and that the values match
        for key, value in self.goal['inform_slots'].items():
            assert value != None
            # For items that cannot be in the queries don't check to see if they are in the agent informs here
            if key in self.no_query:
                continue
            # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
            if value != agent_informs.get(key, None):
                self.constraint_check = FAIL
                break

        if self.constraint_check == FAIL:
            self.state['intent'] = 'reject'
            self.state['request_slots'].clear()
        # print("_response_to_match_found, state: ", self.state)
    def _response_to_done(self):
        """
        Phản hồi với intent của tác tử là 'done'.

        If the constraint_check is SUCCESS and both the rest and request slots of the state are empty for the agent
        to succeed in this episode/conversation.

        Returns:
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        if self.constraint_check == FAIL:
            return FAIL

        if not self.state['rest_slots']:
            assert not self.state['request_slots']
        if self.state['rest_slots']:
            return FAIL

        # TEMP: ----
        assert self.state['history_slots'][self.default_key] != 'no match available'

        match = copy.deepcopy(self.database[int(self.state['history_slots'][self.default_key])])

        for key, value in self.goal['inform_slots'].items():
            assert value != None
            if key in self.no_query:
                continue
            if value != match.get(key, None):
                assert True is False, 'match: {}\ngoal: {}'.format(match, self.goal)
                break
        # ----------
        # print("_response_to_done, state: ", self.state)
        return SUCCESS
