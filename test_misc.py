
from sc3s._misc import _generate_num_range

def test_num_range():
    assert _generate_num_range(5,2,1) == [3,4,5,6,7]
