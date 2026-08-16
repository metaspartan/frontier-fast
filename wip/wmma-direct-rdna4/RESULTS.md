# Direct global->VGPR WMMA feasibility slices (RDNA4, Q4_K, width 16)

Standalone prototypes for the "MMQ without LDS staging" kernel. Compile:
  hipcc -O3 --offload-arch=gfx1201 sliceN.hip -o sliceN

Verdicts (R9700, ffn_gate shape 17408x5120, cold-stream = 8 x 56MB copies):
- slice1 (single 16-row stripe/wave): bit-exact vs dp4a reference
  (0/278528), 122 VGPR 0 spills, ~371 GB/s qs cold.
- slice2 (two stripes/wave): bit-exact but 192 VGPR with 94 spilled
  registers -> 151 GB/s. Stripe-doubling is dead at this budget.
- slice3 (transposed half2 scales, one 32B scale load per lane/sub):
  bit-exact, cold 374 qs / 468 total-stream GB/s. Warm (Infinity-Cache
  resident) 915 GB/s-equivalent.

Established: the WMMA operand layout is solved (tile<16,8,int>: row=lane%16,
k-dword=4*(lane/16)+l; C is TRANSPOSED on RDNA4 - row scales vary per acc
element, col scale fixed per lane). One qs dword feeds TWO sub-blocks via
low/high nibble masks. Exactness by construction.

Open (the resume point): cold DRAM is the limiter, at 73% vs MMQ's 79% and
MMVQ's 84%, because per-lane fragment loads make every wave instruction a
16-cache-line gather (row stride 2880B). Warm 915 proves ~2x issue headroom.
Next iteration: coalesced row-loads (one row's 128B superblock per wave
instruction) redistributed into fragment layout with lane permutes
(v_permlane16/32), priced ~16 VALU per lane per sub-block against the idle
VALU budget. If that lands >550 GB/s the full kernel is worth building;
below MMQ's 507 the family closes.
