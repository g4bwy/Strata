# Unsloth Q8_0 (every expert on the GPU)

**Experimental, and not measured on hardware yet.** Strata can be configured for
[unsloth/Qwen3.8-Flash-Next-GGUF, `Q8_0`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): the 8-bit file,
whose experts stay on the graphics cards for the whole run. Nothing in this page is a measurement. Every number
below comes from the model's geometry and the engine's own arithmetic, and each one names the formula it came
from. A number with no measurement behind it is written as a computation, not as a result.

The mode is `--all-experts-vram`. It is for a PC with more VRAM than the model's 8-bit experts
(128 GB, which wants ~185 GB of VRAM between at least three cards once each card's own weights and buffers are counted) and little use for a RAM tier: four cards of 64 GB, for example. Setup writes it
for you when the cards can hold the model, and refuses when they cannot.

## Why a card with a lot of VRAM needs this mode

Strata's normal design spreads the experts over three tiers, because a consumer PC has 12-24 GB of VRAM and the
experts are 34-128 GB:

```
  normal (one 12 GB card)                 --all-experts-vram (4 x 64 GB)
  +---------------------+                 +---------------------+
  | VRAM  hottest part  |                 | VRAM  ALL 24,576    |
  +---------------------+                 +---------------------+
  | RAM   every expert  |  <- 128 GB      | (no expert tier)    |
  +---------------------+     of RAM      +---------------------+
  | SSD   the file      |                 | RAM  the n-gram     |
  +---------------------+                 |      table only     |
                                          +---------------------+
```

Two numbers explain the change. One 8-bit expert is 5.22 MB, so it crosses a Gen1 x4 link (the electrical width
of a CMP 170HX: about 1 GB/s) in 5.2 ms. The engine already refuses to send misses over a link slower than
about 4 GB/s for that reason (`--pcie-frac`, generate.cpp), and a 24 GB card never gets that close: its experts
are 2.625 bits and cross in 1.4 ms. And the usual RAM tier means a 128 GB host copy of bytes the model's own
file already holds. So on a card like this one, every tier below the GPU is a loss, and the mode removes it:

- No host expert arena. The experts are read from the model's GGUF straight into the cache slots once, at start,
  and the mapped pages are left behind. Setup still writes `--mmap-experts` so that an older engine refuses the
  run instead of starting a 128 GB copy.
- `--pcie-frac 0`: no expert is ever read over the link mid-token, because no expert is missing.
- Adaptive swaps off: the tier is already complete, so there is nothing to swap in.
- The prompt path keeps its own buffers instead of borrowing cache slots, because a borrowed slot has to come back
  from the file when the prompt ends.
- The start refuses if the cards cannot hold all `layers x experts` pairs, and names the GiB that is missing.

## What it needs

| | |
| --- | --- |
| GPU | ~185 GB of VRAM between at least three cards: 128.35 GB of experts plus 5.5 GB of dense weights, 2 GB of prompt buffers and 1 GB of reserve on every card, plus the draft layer on the last. Three 64 GB cards is the smallest rig that fits a 1M fp16 cache (183.9 GB of the 206 GB they have); two do not. Four leave 17-22 GB free on every card. NVIDIA only so far: the mode has not been run on AMD. |
| RAM | 64 GB is enough to read the table through the OS file cache. 96 GB or more lets setup lock the whole n-gram table in RAM (`--ple-io ram`, Linux only), which is the fast choice. |
| Disk | ~188 GB for the six shards, 1.4 GB for the pack, ~6 GB for the MTP draft layer. No `experts.bin`: the engine reads the experts from the GGUF files in place. |
| Engine | 0.1.35 or newer (`--all-experts-vram`, and the 8-bit n-gram table). Setup checks this before it downloads anything. |

The card's own limits do not matter to Strata: no NVLink, no peer access, and no display output. A layer split
hands a token from one card to the next once per verify window through ordinary RAM
([MULTI_GPU.md](MULTI_GPU.md)), and a resizable BAR of 64 MiB does not affect it.

## Setup

```sh
START-HERE.bat --setup --family unsloth --model Q8_0 --context 1048576 --gpus 0,1,2,3
```

Linux: `./setup.sh --setup` with the same flags. What setup decides:

- **The file list.** Unsloth splits a release by file size, so the number of shards belongs to the quantization
  and not to the model. Setup reads it from the repository's own listing at the pinned revision and takes the
  count from the `-0000i-of-0000N` names. `--gguf-dir` with files you downloaded yourself works too and reads
  the same names. A release with a pinned size and SHA-256 table (UD-Q4_K_XL) still uses that table.
- **The context.** The recommendation is the longest context whose plan fits the cards, which on 4 x 64 GB is
  1,048,576. Past the trained 262,144 positions setup adds yarn with the factor that covers it (1048576 /
  262144 = 4), as it does for every other model.
- **The KV cache.** `--kv fp16`, the unquantized cache, is the default for this model and the only value that
  keeps K and V exactly as the projections computed them. `--kv int8` still works and takes half the space.
- **No KV streaming.** `--kv-resident` is not written. Streaming exists to free VRAM for experts, and here the
  experts already have all the room they need.
- **The n-gram table.** Setup reads the table's size from the GGUF (it is 54.4 GB as Q8_0 rows, 28.8 GB as
  IQ4_NL) and picks the tier the RAM can afford: locked in RAM, then the OS file cache, then direct SSD reads.
  See [The table stays off the GPU](#the-table-stays-off-the-gpu).

## The numbers for 4 x 64 GB, a 1M context, fp16

Computed from `src/core/layer.cpp` (the cache's bytes per position) and the geometry in
`include/strata/core/layout.hpp`, with the model's expert count and the block size of Q8_0 (34 bytes per 32
values). A card reports 64 GiB, which is 68.7 GB, and everything below is in decimal GB.

Per card, with the 48 layers split 12 / 12 / 12 / 12:

| | GB |
| --- | ---: |
| its 12 layers' experts (6,144 x 5.22 MB) | 32.08 |
| the whole 1M context's KV for its 3 QSA layers (3 x 1,048,576 x 2,048 B) | 6.44 |
| its indexer keys and (on the last card) the rope table | 0.10 (+0.27) |
| its copy of the dense weights, the head and the embedding | 5.50 |
| its prompt buffers | 2.00 |
| the reserve (`--vram-reserve-mib 700` and the verify windows) | 1.00 |
| **a card's total** | **47.12** |
| **of the 68.72 GB it has** | 21.60 free |

The last card runs the draft layer too, so it holds a fourth KV pool (2.15 GB) and the draft's weights (1.5 GB):
**51.07 GB, with 17.65 GB free**. That is the card the plan is judged by, and `setup --check` prints the same
figures before it downloads anything.

Whole model:

| | GB |
| --- | ---: |
| experts, 24,576 x 5.22 MB | 128.35 |
| KV cache, 1M positions, 13 pools x 2,048 B (12 QSA layers + the draft layer) | 27.9 |
| indexer's pooled keys, 13 x 32 B a token | 0.44 |
| rope table, 256 B a position, one per card | 0.27 |
| dense weights and the head, per card | 5.50 x 4 |
| draft layer's weights, one card | ~1.5 |
| **VRAM, all four cards** | **~192 of 275** |
| n-gram table in host RAM (a tier of its own, never in VRAM) | 54.40 |

The experts at 8 bits are 128.35 GB because one is gate + up + down = 3 x 640 x 2,560 = 4,915,200 values, and
Q8_0 stores 32 values in 34 bytes: 5,222,400 B an expert. The same model at Q2_0 is 34 GB
([DETAILS.md](DETAILS.md#which-model)).

At start the cards fill at the same time. Each takes its own 32 GB across its own link, so the wait is one
card's share, not four: about 30 s each on a 1 GB/s link, about 1 s on a 26 GB/s one. The engine prints the
measured rate and what is left, per card.

## The table stays off the GPU

The n-gram table (`per_layer_token_embd.weight`, 320,001,536 rows of 160 values) is the one large thing this
mode keeps in host RAM. It is never copied to the GPU: a token needs 16 of its rows, and the engine reads those
and uploads 2,560 floats
([DETAILS.md](DETAILS.md#how-it-works)). Three tiers, chosen by `--ple-io`:

| | needs | what it costs |
| --- | --- | --- |
| `ram` | the table's size plus 12 GB | locked in RAM at start with `mlock`. The fastest, and Linux only |
| `mmap` | the table's size plus 10 GB | the OS file cache: read once, then kept as long as the OS can |
| `direct` | nothing | unbuffered SSD reads, about 4 KiB a row. The table never enters RAM |

At 8 bits the table is 54.4 GB. That is a Q8_0 row of 170 bytes (five 32-value blocks at 34 bytes), which
engine 0.1.35 reads next to the IQ4_NL, Q5_0 and FP8 rows it already knew. A PC with 64 GB of RAM gets `mmap` or
`direct` from setup, and that is the right answer: 54 GB locked in RAM leaves nothing for the file cache the
model's own experts are read through at start.

## What is not measured

Written as of engine 0.1.35, before any run on hardware of this size:

- Speed. Nothing here is a tokens/s figure, and the published tables
  ([DETAILS.md](DETAILS.md#speed-measured)) are one-card 12-24 GB numbers. The 8-bit experts also move about 2.5x
  the bytes through the GPU's memory that the 3-bit ones do, so expect decode to be bound by the card's own
  bandwidth (1.49 TB/s on a CMP 170HX) rather than by anything Strata changes.
- Quality at 8 bits. The experts, the head and the attention projections are read as the file stores them, the
  same way the 4-bit Unsloth file's are ([UNSLOTH_Q4.md](UNSLOTH_Q4.md#the-numbers-what-is-native-and-what-is-converted)).
- Perplexity and needle tests at 1M with yarn 4. The rope extension past the trained range is experimental for
  every model, measured by contributors at 293K-714K on one quant
  ([DETAILS.md](DETAILS.md#context-extension-past-262k-rope-scaling-experimental)).
- AMD. The mode is NVIDIA-only until somebody runs it there.

If you run it, `--stats` and the engine log line (`strata serve: ... MiB of VRAM free with everything loaded`,
and `all 24576 experts are in VRAM`) are what a report needs.

## By hand

The engine flags, without setup:

```
--pack packs/unsloth-q8_0
--native Qwen3.8-Flash-Next-Q8_0-00001-of-0000N.gguf
--expert-profile data/expert-profile.bin
--expert-cache auto
--all-experts-vram --mmap-experts
--max-context 1048576 --kv fp16
--rope-scaling yarn --rope-scale 4
--ple-io ram
--prefill auto --spec 4 --spec-min-p 0.5 --mtp mtp/rt
--layer-split auto --split-device 1,2,3
```

`--expert-profile` must rank every pair: `data/expert-profile.bin` holds all 48 x 512. A profile that names
fewer experts cannot complete the tier, and the start stops with the number that is missing. Build the pack with
`tools/iq_pack.py --gguf <shard1> --out packs/unsloth-q8_0 --compat-bf16`, as for the 4-bit file.
