import unittest
from roost.cudatools import *


class ToolsTest(unittest.TestCase):

    def test_get_next_power_of_two(self):
        assert get_next_power_of_two(7) == 8
        assert get_next_power_of_two(8) == 8
        assert get_next_power_of_two(11) == 16
        assert get_next_power_of_two(15) == 16
        assert get_next_power_of_two(16) == 16
        assert get_next_power_of_two(17) == 32
        assert get_next_power_of_two(0) == 1
        assert get_next_power_of_two(1) == 1
        assert get_next_power_of_two(2) == 2
        assert get_next_power_of_two(3) == 4
        assert get_next_power_of_two(4) == 4
        assert get_next_power_of_two(5) == 8


if __name__ == '__main__':
    unittest.main()

