| split   | key              | shape         | dtype   |
|:--------|:-----------------|:--------------|:--------|
| all     | X_jets           | (20664, 8, 5) | float32 |
| all     | assignment       | (20664, 2, 2) | int64   |
| all     | assignment_mask  | (20664,)      | int64   |
| all     | event_id         | (20664,)      | int64   |
| all     | feature_names    | (5,)          | <U4     |
| all     | jet_mask         | (20664, 8)    | bool    |
| all     | n_selected_btags | (20664,)      | int64   |
| all     | n_selected_jets  | (20664,)      | int64   |
| all     | sample_groups    | (8,)          | <U10    |
| all     | sample_id        | (20664,)      | int64   |
| all     | sample_names     | (8,)          | <U33    |
| all     | weight_pb        | (20664,)      | float64 |
| all     | y                | (20664,)      | int64   |
| train   | X_jets           | (14462, 8, 5) | float32 |
| train   | assignment       | (14462, 2, 2) | int64   |
| train   | assignment_mask  | (14462,)      | int64   |
| train   | event_id         | (14462,)      | int64   |
| train   | feature_names    | (5,)          | <U4     |
| train   | jet_mask         | (14462, 8)    | bool    |
| train   | n_selected_btags | (14462,)      | int64   |
| train   | n_selected_jets  | (14462,)      | int64   |
| train   | sample_groups    | (8,)          | <U10    |
| train   | sample_id        | (14462,)      | int64   |
| train   | sample_names     | (8,)          | <U33    |
| train   | weight_pb        | (14462,)      | float64 |
| train   | y                | (14462,)      | int64   |
| val     | X_jets           | (3095, 8, 5)  | float32 |
| val     | assignment       | (3095, 2, 2)  | int64   |
| val     | assignment_mask  | (3095,)       | int64   |
| val     | event_id         | (3095,)       | int64   |
| val     | feature_names    | (5,)          | <U4     |
| val     | jet_mask         | (3095, 8)     | bool    |
| val     | n_selected_btags | (3095,)       | int64   |
| val     | n_selected_jets  | (3095,)       | int64   |
| val     | sample_groups    | (8,)          | <U10    |
| val     | sample_id        | (3095,)       | int64   |
| val     | sample_names     | (8,)          | <U33    |
| val     | weight_pb        | (3095,)       | float64 |
| val     | y                | (3095,)       | int64   |
| test    | X_jets           | (3107, 8, 5)  | float32 |
| test    | assignment       | (3107, 2, 2)  | int64   |
| test    | assignment_mask  | (3107,)       | int64   |
| test    | event_id         | (3107,)       | int64   |
| test    | feature_names    | (5,)          | <U4     |
| test    | jet_mask         | (3107, 8)     | bool    |
| test    | n_selected_btags | (3107,)       | int64   |
| test    | n_selected_jets  | (3107,)       | int64   |
| test    | sample_groups    | (8,)          | <U10    |
| test    | sample_id        | (3107,)       | int64   |
| test    | sample_names     | (8,)          | <U33    |
| test    | weight_pb        | (3107,)       | float64 |
| test    | y                | (3107,)       | int64   |
