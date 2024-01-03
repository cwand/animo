import unittest
import animo


class TestTester(unittest.TestCase):

	def test_add(self):
		t = animo.add(2, 3)
		self.assertEqual(t, 5)
