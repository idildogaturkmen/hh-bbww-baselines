import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analysis.hh4b_expected_limit import expected_upper_limit,qmu_asimov
class Tests(unittest.TestCase):
 def test_large_count(self): self.assertTrue(math.isclose(expected_upper_limit([100],[1e6]),19.6,rel_tol=2e-3))
 def test_monotonic(self):
  x=expected_upper_limit([10],[1000]); self.assertLess(expected_upper_limit([20],[1000]),x); self.assertLess(expected_upper_limit([10],[500]),x)
 def test_split_consistency(self): self.assertTrue(math.isclose(expected_upper_limit([10],[1000]),expected_upper_limit([4,6],[400,600]),rel_tol=1e-12))
 def test_categories(self): self.assertLessEqual(expected_upper_limit([10,2],[1000,100]),expected_upper_limit([12],[1100]))
 def test_finite_mc(self): self.assertGreater(expected_upper_limit([10],[1000],[10000]),expected_upper_limit([10],[1000]))
if __name__=='__main__': unittest.main()
