| requirement | frozen_value | verification |
| --- | --- | --- |
| top_level_output_created_before_checks | True | worker mkdir precedes payload/input checks |
| shell_failure_mode | set -Eeuo pipefail | worker line 3 |
| final_status_always_written | True | EXIT trap |
| failure_receipt_always_written | True | EXIT trap |
| transfer_existing_directory | output | submit transfer_output_files |
| mandatory_individual_parquet_transfer | False | directory transfer |
| placeholder_parquet | forbidden | wrapper never creates placeholder |
| wrapper_nonzero_without_valid_parquet | True | EXIT trap and schema audit |
| LHE_HepMC_new_ROOT_transfer | forbidden | output registry |
