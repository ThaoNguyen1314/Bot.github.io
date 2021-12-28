from collections import defaultdict
from dialogue_config import no_query_keys, usersim_default_key
import copy


class DBQuery:
    """
    Queries the database for the state tracker.
    Truy vấn cơ sở dữ liệu cho trình theo dõi trạng thái.
    """
    def __init__(self, database):
        """
        The constructor for DBQuery.
        Hàm tạo cho DBQuery.

        Parameters:
            database (dict): The database in the format dict(long: dict)
                            Cơ sở dữ liệu ở định dạng dict (key là số nguyên : value là kiểu dict)
        """

        self.database = database
        # {frozenset: {string: int}} A dict of dicts
        # kiểu dữ liệu dict có key là mảng bất biến với các chỉ mục ko đổi, ko chứa giá trị trùng lặp và value là kiểu dict 
        self.cached_db_slot = defaultdict(dict) # mặc định mảng cached có kiểu dữ liệu dict
        # {frozenset: {'#': {'slot': 'value'}}} A dict of dicts of dicts, a dict of DB sub-dicts
        # frozenset là mảng bất biến với các chỉ mục ko đổi, ko chứa giá trị trùng lặp 
        self.cached_db = defaultdict(dict)
        self.no_query = no_query_keys # các khoá inform có thể ko cần sử dụng để truy vấn
        self.match_key = usersim_default_key # thông báo kết quả ghép inform và request khớp = "ticket"

    def fill_inform_slot(self, inform_slot_to_fill, current_inform_slots):
        """
        Given the current informs/constraints fill the informs that need to be filled with values from the database.
        Dùng các thông tin / ràng buộc hiện tại, điền vào các thông báo cần được điền với các giá trị từ cơ sở dữ liệu.

        Searches through the database to fill the inform slots with PLACEHOLDER with values that work given the current
        constraints of the current episode.
        Tìm kiếm thông qua cơ sở dữ liệu để điền các inform slot bằng PLACEHOLDER với các giá trị mà phù hợp với 
        các ràng buộc hiện tại của tập hiện tại.

        Parameters:
            inform_slot_to_fill (dict): Inform slots to fill with values 
                                        Mảng các Inform slot để điền giá trị
            current_inform_slots (dict): Current inform slots with values from the StateTracker
                                        Các inform slot hiện tại với giá trị từ StateTracker

        Returns:
            dict: inform_slot_to_fill filled with values
                  inform_slot_to_fill được điền hết các giá trị
        """

        # For this simple system only one inform slot should ever passed in
        # Đối với hệ thống đơn giản này, chỉ có 1 inform slot sẽ được nhập vào
        assert len(inform_slot_to_fill) == 1

        key = list(inform_slot_to_fill.keys())[0] # key = giá trị tại vị trí 0 của mảng gồm các khoá trong inform_slot_to_fil

        # This removes the inform we want to fill from the current informs if it is present in the current informs
        # so it can be re-queried
        # Điều này sẽ xóa thông tin chúng tôi muốn điền từ thông báo hiện tại nếu nó đã có trong thông báo hiện tại để nó được truy vấn lại

        current_informs = copy.deepcopy(current_inform_slots) # khi thay đổi phần tử trong current inform thì current_inform_slots vẫn giữ nguyên
        current_informs.pop(key, None) #nếu ko có phần tử = key trả về none

        # db_results is a dict of dict in the same exact format as the db, it is just a subset of the db
        # db_results là một từ điển của dict có cùng định dạng chính xác hệt như db, nó chỉ là một tập con của db
        db_results = self.get_db_results(current_informs)

        filled_inform = {}
        values_dict = self._count_slot_values(key, db_results)
        if values_dict:
            # Get key with max value (ie slot value with highest count of available results)
            # Nhận key với giá trị max (giá trị slot với số lượng kết quả có sẵn cao nhất)
            filled_inform[key] = max(values_dict, key=values_dict.get)
        else:
            filled_inform[key] = 'no match available'

        return filled_inform

    def _count_slot_values(self, key, db_subdict):
        """
        Return a dict of the different values and occurrences of each, given a key, from a sub-dict of database
        Trả về dict của các value khác nhau và lần xuất hiện của mỗi value, được cung cấp một key, từ một dict con của cơ sở dữ liệu

        Parameters:
            key (string): The key to be counted
                          Khoá được đếm (định dạng xâu kí tự)
            db_subdict (dict): A sub-dict of the database

        Returns:
            dict: The values and their occurrences given the key
                  Các giá trị và lần xuất hiện của chúng được cung cấp cho khóa
        """

        slot_values = defaultdict(int)  # init to 0 (khởi tạo mảng kiểu biến mặc định là int, bắt đầu từ 0)
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            # If there is a match (nếu có kết nối)
            if key in current_option_dict.keys():
                slot_value = current_option_dict[key]
                # This will add 1 to 0 if this is the first time this value has been encountered, or it will add 1
                # to whatever was already in there
                # Nó sẽ thêm 1 vào 0 nếu đây là lần đầu giá trị này bị gặp phải, hoặc nó sẽ + 1 vào giá trị đã có
               
                slot_values[slot_value] += 1
        return slot_values

    def get_db_results(self, constraints):
        """
        Get all items in the database that fit the current constraints.
        Nhận tất cả các mục trong cơ sở dữ liệu phù hợp với các ràng buộc hiện tại.

        Looks at each item in the database and if its slots contain all constraints and their values match then the item
        is added to the return dict.
        Xem xét từng mục trong cơ sở dữ liệu và nếu các slot của nó chứa tất cả các ràng buộc và giá trị của chúng khớp nhau 
        thì mục đó được thêm vào dict trả về.

        Parameters:
            constraints (dict): The current informs
                                Thông báo hiện tại

        Returns: 
            dict: The available items in the database (các mục có sẵn trong cơ sở dữ liệu)
        """

        # Filter non-queryable items and keys with the value 'anything' since those are inconsequential to the constraints
        # Lọc các mục và khóa không thể truy vấn có giá trị 'anything' vì chúng không quan trọng đối với các ràng buộc
        new_constraints = {k: v for k, v in constraints.items() if k not in self.no_query and v != 'anything'}

        inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]

        if cache_return == None:
            # If it is none then no matches fit with the constraints so return an empty dict
            # Nếu không có thì không có kết quả nào phù hợp với các ràng buộc vì vậy trả về một dict trống
            return {}
        # if it isnt empty then return what it is
        # nếu nó không trống thì trả về chính nó
        if cache_return:
            return cache_return
        # else continue on (nếu khác thì tiếp tục)

        available_options = {}
        for id in self.database.keys():
            current_option_dict = self.database[id]
            # First check if that database item actually contains the inform keys
            # Đầu tiên hãy kiểm tra xem mục cơ sở dữ liệu đó có thực sự chứa các khóa thông tin hay không
            # Note: this assumes that if a constraint is not found in the db item then that item is not a match
            # Lưu ý: điều này giả định rằng nếu không tìm thấy ràng buộc trong mục db thì mục đó không phải là đối sánh
            if len(set(new_constraints.keys()) - set(self.database[id].keys())) == 0:
                match = True
                # Now check all the constraint values against the db values and if there is a mismatch don't store
                # Bây giờ kiểm tra tất cả các giá trị ràng buộc so với các giá trị db và nếu có cái k khớp thì ko lưu trữ
                for k, v in new_constraints.items():
                    if str(v).lower() != str(current_option_dict[k]).lower():
                        match = False
                if match:
                    # Update cache
                    # cập nhật bộ nhớ cache
                    self.cached_db[inform_items].update({id: current_option_dict})
                    available_options.update({id: current_option_dict})

        # if nothing available then set the set of constraint items to none in cache
        # nếu không có gì thì hãy đặt tập hợp các mục ràng buộc thành không có trong bộ nhớ cache
        if not available_options:
            self.cached_db[inform_items] = None

        return available_options

    def get_db_results_for_slots(self, current_informs):
        """
        Counts occurrences of each current inform slot (key and value) in the database items.
        Đếm số lần xuất hiện của mỗi inform slot hiện tại (khóa và giá trị) trong các mục cơ sở dữ liệu.

        For each item in the database and each current inform slot if that slot is in the database item (matches key
        and value) then increment the count for that key by 1.
        Đối với mỗi mục trong cơ sở dữ liệu và mỗi inform slot hiện tại nếu slot đó nằm trong mục cơ sở dữ liệu (khớp với khóa và giá trị) 
        thì hãy tăng số lượng cho khóa đó lên 1.

        Parameters:
            current_informs (dict): The current informs/constraints

        Returns:
            dict: Each key in current_informs with the count of the number of matches for that key
                  Mỗi khóa trong current_informs với số lượng khớp cho khóa đó
        """

        # The items (key, value) of the current informs are used as a key to the cached_db_slot
        # Các mục (khóa, giá trị) của thông tin hiện tại được sử dụng làm khóa cho cache_db_slot
        inform_items = frozenset(current_informs.items())
        # A dict of the inform keys and their counts as stored (or not stored) in the cached_db_slot
        # Một dict của các khóa thông báo và số lượng của chúng như được lưu trữ (hoặc không được lưu trữ) trong cache_db_slot
        cache_return = self.cached_db_slot[inform_items]

        if cache_return:
            return cache_return

        # If it made it down here then a new query was made and it must add it to cached_db_slot and return it
        # Nếu chạy đến đây thì một truy vấn mới đã được thực hiện và nó phải thêm nó vào cache_db_slot và trả lại nó
        # Init all key values with 0
        # Nhập tất cả các giá trị khóa bằng 0
        db_results = {key: 0 for key in current_informs.keys()}
        db_results['matching_all_constraints'] = 0

        for id in self.database.keys():
            all_slots_match = True
            for CI_key, CI_value in current_informs.items():
                # Skip if a no query item and all_slots_match stays true
                # Bỏ qua nếu không có mục truy vấn nào và all_slots_match vẫn đúng
                if CI_key in self.no_query:
                    continue
                # If anything all_slots_match stays true AND the specific key slot gets a +1
                # Nếu bất kỳ điều gì all_slots_match vẫn đúng VÀ vị trí khóa cụ thể nhận được +1
                if CI_value == 'anything':
                    db_results[CI_key] += 1
                    continue
                if CI_key in self.database[id].keys():
                    if CI_value.lower() == self.database[id][CI_key].lower():
                        db_results[CI_key] += 1
                    else:
                        all_slots_match = False
                else:
                    all_slots_match = False
            if all_slots_match: db_results['matching_all_constraints'] += 1

        # update cache (set the empty dict)
        # cập nhật bộ nhớ cache (đặt chính sách trống)
        self.cached_db_slot[inform_items].update(db_results)
        assert self.cached_db_slot[inform_items] == db_results
        return db_results
