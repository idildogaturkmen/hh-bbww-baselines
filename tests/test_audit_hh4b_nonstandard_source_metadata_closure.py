#!/usr/bin/env python3
import importlib.util,sys,tempfile,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
SCRIPT=REPO/"scripts/analysis/audit_hh4b_nonstandard_source_metadata_closure.py"
spec=importlib.util.spec_from_file_location("m",SCRIPT);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class T(unittest.TestCase):
 def test_qcd(self):
  x=m.parse_tag("qcd_hardqcd_x_bin03_pthat200to300_shard1033_seed1201033_bundle.tar.gz","qcd_hardqcd")
  self.assertEqual((x["bin_index"],x["pthat_min_GeV"],x["pthat_max_GeV"],x["shard_id"],x["seed"]),(3,200,300,1033,1201033))
 def test_qcd_open_ended_upper_bound(self):
  x=m.parse_tag("qcd_hardqcd_importance_adaptive1500k_phys_wave3_20260718_bin07_pthat1000toInf_shard1182_seed1271182_bundle.tar.gz","qcd_hardqcd")
  self.assertEqual((x["bin_index"],x["pthat_min_GeV"],x["pthat_max_GeV"],x["shard_id"],x["seed"]),(7,1000,"Inf",1182,1271182))
  self.assertTrue(m.pthat_interval_valid(1000,"Inf"))
 def test_qcd_open_ended_lowercase_is_canonicalized(self):
  x=m.parse_tag("qcd_hardqcd_x_bin07_pthat1000toinf_shard1182_seed1271182_bundle.tar.gz","qcd_hardqcd")
  self.assertEqual(x["pthat_max_GeV"],"Inf")
 def test_qcd_group_sort_supports_finite_and_open_bins(self):
  groups={
   ("campaign",6,700,1000):[],
   ("campaign",7,1000,"Inf"):[],
  }
  ordered=[item[0][1] for item in sorted(groups.items(),key=m.qcd_group_sort_key)]
  self.assertEqual(ordered,[6,7])
 def test_ggf(self):
  x=m.parse_tag("HH4b_ggf_campaign_shard000_seed86000_bundle.tar.gz","ggf_hh4b");self.assertEqual((x["shard_id"],x["seed"]),(0,86000))
 def test_payload_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   for n in ("a.root","a.parquet","a.hepmc","a.lhe","a.tar.gz"):
    p=Path(d)/n;p.write_bytes(b"x");self.assertFalse(m.safe_text(p))
 def test_keys(self):self.assertIsNotNone(m.KEY_RE.search("generator.sum_event_weights"))
 def test_no_payload_reader(self):
  s=SCRIPT.read_text()
  for token in ("uproot","pyhepmc","read_parquet","pyarrow"):self.assertNotIn(token,s)
if __name__=="__main__":unittest.main()
