from dialogue_config import FAIL, SUCCESS


def convert_list_to_dict(lst):
    """
    Convert list to dict where the keys are the list elements, and the values are the indices of the elements in the list.
    Chuyển đổi danh sách nhập vào thành các từ có trong tệp từ điển,
    các khoá là phần tử trong danh sách  và các giá trị là chỉ số của các phần tử trong danh sách
    
    Parameters:
        lst (list)
    Thông số là danh sách lst
  
    Returns:
        dict
    trả về từ điển (các phần tử được đánh số lần lượt)
    """

    if len(lst) > len(set(lst)):  # kiểm tra nếu danh sách bị trùng
        raise ValueError('List must be unique!')  # dừng chương trình và hiển thị lỗi "danh sách phải là duy nhất"
    return {k: v for v, k in enumerate(lst)}   # trả về các phần tử đc đánh số {A:0, B:1, C:2,..}


def remove_empty_slots(dic):
    """
    Removes all items with values of '' (ie values of empty string).
    Xoá tất cả chỗ trống có giá trị ''(vd giá trị của xâu ký tự rỗng)

    Parameters:
        dic (dict)
    Thông số là từ điển
    """

    for id in list(dic.keys()):  #chạy id trong mảng dic.keys
        for key in list(dic[id].keys()): #chay key trong mảng thứ id của dic.keys
            if dic[id][key] == '': # nếu phần tử thứ id,key của mảng dic là rỗng
                dic[id].pop(key) # xoá phần tử thứ id,key


def reward_function(success, max_round):
    """
    Return the reward given the success.
    Trả lại phần thưởng cho thành công

    Return -1 + -max_round if success is FAIL, -1 + 2 * max_round if success is SUCCESS and -1 otherwise.
    Trả về -1 + -max_round nếu success là FAIL, 
             -1 + 2 * max_round nếu success là SUCCESS
             -1 trong trường hợp khác
    max_round đc khai báo trong file hằng số constants

    success có thể là NO_OUTCOME nếu tập không được hoàn thành, FAIL nếu tập được hoàn thành và FAIL hoặc SUCCESS nếu tập được hoàn thành và thành công
    Parameters:
        success (int)

    Returns:
        int: Reward

    Chức năng phần thưởng này giúp tác tử học cách thành công bằng cách trao cho nó một phần thưởng lớn cho việc thành công. 
    Và cho phép nó học cách tránh thất bại bằng cách cho nó một hình phạt lớn nếu thất bại nhưng không lớn bằng phần thưởng cho thành công. 
    Điều này giúp tác tử không quá sợ hãi khi chấp nhận rủi ro để có được phần thưởng lớn khi thành công, 
    nếu không, nó có thể kết thúc sớm tập phim để giảm bớt phần thưởng tiêu cực.

    """

    reward = -1
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward

