import unittest
from main import *

class TestFunctions(unittest.TestCase):
	def test1(self):
		self.assertEqual(2/2, 1)

	# def test_Menu(self):
	# 	ret = menu()
	# 	print('ret:  ', ret)
	# 	# self.assertNotEqual(ret,'Jay')

if __name__ == '__main__':
    unittest.main()