# HH→4b validation in-place transport recovery

The accepted 116-job cluster was repaired in place after a pre-execution client-local `/tmp` transfer failure. Only `Cmd` and `TransferInput` were changed to byte-identical frozen shared copies; no new submission occurred.
