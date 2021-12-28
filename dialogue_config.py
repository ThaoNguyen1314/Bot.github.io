# Special slot values (for reference) #Các values đặc biệt trong dãy thông báo
'PLACEHOLDER'  # For informs 
'UNK'  # For requests
'anything'  # means any value works for the slot with this value
'no match available'  # When the intent of the agent is match_found yet no db match fits current constraints

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ['inform', 'request', 'thanks', 'reject', 'done']#Tập các tham số intents sử dụng trong E_M_C.py và cả user.py


# The goal of the agent is to inform a match for this key
usersim_default_key = 'ticket'# value này thông báo trong agent_inform_slots có match được agent_request_slots hay không

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['moviename'] #value đầu tiên yêu cầu trong dãy thông báo của usersim nếu nó tồn tại trong dãy thông báo

#######################################
#Cấu hình đối thoại cho tác nhân
#Dưới đây là các constants cấu hình đối thoại trong file dialogue_config.py được tác nhân sử dụng:
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating',
                      'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor',
                      'description', 'other', 'numberofkids', usersim_default_key] # agent_inform_slots là tất cả các key values có thể có cho các thông tin của tác nhân.
agent_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip',
                       'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price',
                       'actor', 'description', 'other', 'numberofkids'] #agent_request_slots là tất cả các key values có thể có cho các yêu cầu của tác nhân 


# Possible actions for agent
agent_actions = [ #các tham số intents khi xuất hiện thì kích hoạt kết thúc trò chuyện
    {'intent': 'done', 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': 'match_found', 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots: # kiểm tra các vị trí trong dãy thông báo
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key: # nếu một vị trí là 'ticket' tức là đã match inform với request
        continue
    agent_actions.append({'intent': 'inform', 'inform_slots': {slot: 'PLACEHOLDER'}, 'request_slots': {}})#Thêm 'placeholder' (giữ chỗ) vào agent_actions
for slot in agent_request_slots:
    agent_actions.append({'intent': 'request', 'inform_slots': {}, 'request_slots': {slot: 'UNK'}})

# Rule-based policy request list # Quy tắc của request follow theo policy sau:
rule_requests = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

# These are possible inform slot keys that cannot be used to query
no_query_keys = ['numberofpeople', usersim_default_key]# từ khóa trong file db_query.py

#######################################
# Global config
#######################################

# These are used for both constraint check AND success check in usersim (sử dụng cho kiểm tra ràng buộc và tính thành công của usersim)
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

# All possible intents (for one-hot conversion in ST.get_state()) (toàn bộ các tham số intent có thể xuất hiện trong 1 cuộc trò chuyện)
all_intents = ['inform', 'request', 'done', 'match_found', 'thanks', 'reject']

# All possible slots (for one-hot conversion in ST.get_state()) (toàn bộ các vị trí có thể có trong dãy thông báo của một cuộc trò chuyện)
all_slots = ['actor', 'actress', 'city', 'critic_rating', 'date', 'description', 'distanceconstraints',
             'genre', 'greeting', 'implicit_value', 'movie_series', 'moviename', 'mpaa_rating',
             'numberofpeople', 'numberofkids', 'other', 'price', 'seating', 'starttime', 'state',
             'theater', 'theater_chain', 'video_format', 'zip', 'result', usersim_default_key, 'mc_list']
