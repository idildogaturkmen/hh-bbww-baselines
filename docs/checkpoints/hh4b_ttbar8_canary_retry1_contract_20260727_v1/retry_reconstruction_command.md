| field | frozen_value | verification |
| --- | --- | --- |
| exact_command | env PYTHONPATH=/srv/payload/bootstrap OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=60s 45m python3 /srv/payload/repo/scripts/delphes/reconstruct_hh4b_candidates_v2.py --input /srv/ttbar_100k_shard003_pythia8_delphes.root --out /srv/output/parquet/ttbar_100k_shard003_pythia8_delphes_canonical72.parquet --sample ttbar_100k_shard003_pythia8_delphes --target-mass 125.0 --jet-pt-min 30.0 --jet-eta-max 2.5 --btag-min 0.0 --max-bjets-for-pairing 8 --higgs-ordering pt | protected builder plus explicit frozen arguments |
| input | /srv/ttbar_100k_shard003_pythia8_delphes.root | 127c87510087d164038b7629ea9d5210008cc4cb1473aa9b069a5cd4b859629e |
| policy | /srv/payload/repo/configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml | 4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68 |
| serial_executor | sitecustomize_sets_MultithreadedFileSource_num_workers_1 | successful retained-ROOT mechanism |
| isolated_writer | /srv/payload/repo/scripts/delphes/write_parquet_from_pickle.py | abef7e4f5d1b82fe72837834b0b0794b59bb31ff16032b1b5a7f1519a6edbcfb |
