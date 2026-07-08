# PHAT-JeT relevance to HH4b project

Paper: Patch Hierarchical Attention Transformer for Efficient Particle Jet Tagging (PHAT-JeT), arXiv:2605.21789.

Main idea:
- PHAT-JeT is a trigger-oriented particle-cloud jet tagger.
- It reduces the cost of full self-attention by computing exact attention within small particle patches.
- It restores global communication through patch-level tokens.
- It adds geometric message passing in the detector eta-phi plane.

Relevance to this project:
- My current HH→4b baseline uses jet-level features and candidate-level variables.
- Similar performance among BDT/DNN/LBN baselines suggests that future gains may require richer inputs.
- PHAT-JeT motivates a particle-aware extension where each AK4/AK8 jet is represented by learned constituent-level embeddings.
- This is especially relevant for boosted or high-mHH HH→4b regions, where H→bb substructure becomes important.

Practical caveat:
- The current storage-safe parquet files are analysis-level summaries, not particle-cloud datasets.
- To test PHAT/ParT-like embeddings, I need to save slim constituent-level parquet files from Delphes ROOT before deleting the ROOT shards, or keep a small ROOT validation sample.
