import example as eg

class TestOperations:
    def test_addition(self):
        assert eg.addx(7,8) == 15

    def test_subtraction(self):
        assert eg.subx(10,5) == 5

def test_multiplication():
    assert eg.sd.multx(2,9) == 18

def test_division():
    assert eg.sd.divx(8,3) == 2

def test_division_is_integer():
    assert isinstance(eg.sd.divx(10,9), int)