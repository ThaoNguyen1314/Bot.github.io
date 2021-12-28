import random
from dialogue_config import usersim_intents


class ErrorModelController:
    """
    Adds error to the user action.
    Thêm lỗi vào hành động người dùng.
    """

    def __init__(self, db_dict, constants):
        """
        The constructor for ErrorModelController.
        Hàm khởi tạo cho Bộ điều khiển mô hình lỗi

        Saves items in constants, etc.
        Lưu các mục trong file hằng số

        Parameters:
        Thông số
            db_dict (dict): The database dict with format dict(string: list) where each key is the slot name and
                            the list is of possible values
                            Cơ sở dữ liệu từ điển với định dạng dict(key là dạng xâu kí tự, value là dạng list), mỗi key là một slot name và 
                            danh sách là các giá trị có thể có
            constants (dict): Loaded constants in dict
                              Hằng số đã tải trong dict
        """
        '''
        Các loại lỗi cấp độ slot cho mỗi inform slot trong hành động:
            Thay thế giá trị bằng một giá trị ngẫu nhiên cho khóa đó
            Thay thế toàn bộ slot: khóa ngẫu nhiên và giá trị ngẫu nhiên cho khóa đó
            Xóa slot
            Xác suất bằng nhau cho cả 3 loại lỗi ở trên cho một slot
        Lỗi mức mục tiêu:
            Thay thế mục tiêu bằng một mục tiêu ngẫu nhiên
        '''
        self.movie_dict = db_dict 
        self.slot_error_prob = constants['emc']['slot_error_prob']  # Xác suất để mỗi slot bị thay đổi
        self.slot_error_mode = constants['emc']['slot_error_mode']  # [0, 3] Một trong 4 chế độ cho lỗi cấp độ slot được liệt kê ở trên
        self.intent_error_prob = constants['emc']['intent_error_prob'] # Xác suất để chọn ngẫu nhiên mục tiêu mới
        self.intents = usersim_intents # Danh sách tất cả các mục tiêu mô phỏng người dùng có thể có, từ config_dia.py

    def infuse_error(self, frame):  # hàm truyền lỗi
        """
        Takes a semantic frame/action as a dict and adds 'error'.
        Lấy khung / hành động ngữ nghĩa làm dict và thêm 'lỗi'.

        Given a dict/frame it adds error based on specifications in constants. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.
        Với một dict/frame, nó sẽ thêm lỗi dựa trên các thông số kỹ thuật trong constants. Nó có thể thay thế các giá trị slot, 
        thay thế slot và các giá trị của nó, xóa một vị trí hoặc thực hiện cả ba. Nó cũng có thể ngẫu nhiên hóa mục đích.

        Parameters:
            frame (dict): format dict('intent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
            VD về nội dung của 1 khung từ điển
        """

        informs_dict = frame['inform_slots']  # gán dict thông báo = key và các value của 'inform_slots trong frame
        for key in list(frame['inform_slots'].keys()):
            assert key in self.movie_dict # nếu key ko có trong self.movie_dict thì dừng chương trình chạy và báp lỗi
            if random.random() < self.slot_error_prob: # chọn xác suất ngẫu nhiên 
                if self.slot_error_mode == 0:  # replace the slot_value only
                    self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1:  # replace slot and its values
                    self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    self._slot_remove(key, informs_dict)
                else:  # Combine all three (Kết hợp cả 3)
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self._slot_noise(key, informs_dict)
                    else:
                        self._slot_remove(key, informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level (thêm nhiễu cho cấp độ mục tiêu)
            frame['intent'] = random.choice(self.intents) # chọn ngẫu nhiên trong list

    def _slot_value_noise(self, key, informs_dict):
        """
        Selects a new value for the slot given a key and the dict to change.
        Chọn một giá trị mới cho vị trí được cung cấp một khóa và từ điển để thay đổi.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict[key] = random.choice(self.movie_dict[key])

    def _slot_noise(self, key, informs_dict):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for this new slot.
        Thay thế vị trí hiện tại được cung cấp một khóa trong dict thông báo bằng một vị trí mới và chọn một giá trị ngẫu nhiên cho vị trí mới này.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
        random_slot = random.choice(list(self.movie_dict.keys()))
        informs_dict[random_slot] = random.choice(self.movie_dict[random_slot])

    def _slot_remove(self, key, informs_dict):
        """
        Removes the slot given the key from the informs dict.
        Loại bỏ vị trí được cung cấp khóa khỏi dict thông báo.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
