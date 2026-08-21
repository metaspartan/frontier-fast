# qwen3.8 R9700: tree verify + mmvf-16 (the 13-node tree shape candidate)

Base: qwen38-tree-verify branch (0001-0008: the tree-speculative engine extension
- tree-aware recurrent scan, one weight pass for the whole tree, DFS emission,
conv snapshots, dead-slot cap, default-ON) + 0009 mmvf-16-column-float-matvec.

Tree shape '2,2,1,1,0,0,1': 7 nodes, depth 4, 3 leaves, 8 verify positions.
Measured E=3.527 accepted/pass vs chain 2.947 (+19.7%), decode ratio +2.38%
over 36 paired ABBA measurements (tree-verify-single-pass finding).

The prior tree attempts (wider 13-node + rs-decouple, 1787118714) scored
1.399 vs frontier 1.493 - they LOST to the chain. The deltas: this series has
NO rs-decouple (that crashed on find_slot) and keeps the same 7-node default
shape that verified locally; mmvf-16 is added to remove the width-9 rocBLAS
cliff for the tree's 8 verify positions.

Not expected to beat the 1.4947 chain frontier on its own; this is the
composition probe to see if tree+mmvf-16 lands above it. If it does, the next
step is widening the tree (13 nodes, E~4) with the find_slot fix.