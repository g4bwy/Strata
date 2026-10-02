"""Tests for setup.py's "everything on the GPU" mode: Unsloth's Q8_0 file, whose 128 GB of experts go on the
cards or nowhere (docs/UNSLOTH_Q8.md).  The per-format KV cache sizes, the per-card plan, the context it can
afford, the tier the n-gram table lands in, the shard list read from the repository, and the config setup writes
for a four-card PC.  Mocked - no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_vram
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

CTX1M = 1048576
# The rig this was configured for: four 64 GB cards (nvidia-smi reports 65536 MiB, so setup sees 64.0 "GB")
CMP = [64.0, 64.0, 64.0, 64.0]


def cards(n, gib=64.0):
    return [gib] * n


class KvSizes(unittest.TestCase):
    """The bytes a context token costs, per format, from src/core/layer.cpp's kv_pool_bytes()."""

    def test_every_format_setup_can_ask_for(self):
        self.assertEqual(setup.KV_BYTES_PER_CELL, {"fp16": 2048, "int8": 1056, "k8v4": 816, "q4_0": 576})
        self.assertEqual(setup.KV_POOLS, 13)          # the 12 QSA layers + the draft layer's own

    def test_the_published_figures_are_kept(self):
        # docs/DETAILS.md: ~13.7 KB of RAM per context token at 8-bit, 7.5 KB at 4-bit - including the indexer's
        # pooled keys, which the per-cell figure alone would miss
        self.assertEqual(setup.kv_token_bytes("int8"), 13 * 1056)
        self.assertEqual(setup.kv_token_bytes("q4_0"), 13 * 576)
        # k8v4 used to be charged as int8 by these estimates: it is 816 B a cell, not 1,056
        self.assertEqual(setup.kv_token_bytes("k8v4"), 13 * 816)
        # fp16 is NOT 2x int8: int8 carries 32 B of per-group scales that fp16 has no equivalent of, so the
        # difference over 13 pools is exactly 13 x 64 B
        self.assertEqual(setup.kv_token_bytes("int8") * 2 - setup.kv_token_bytes("fp16"), 13 * 64)

    def test_the_command_line_offers_all_four(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--kv", choices=["fp16", "int8", "q4_0", "k8v4"]', src)


class Plan(unittest.TestCase):
    def test_the_layer_split_of_four_equal_cards_is_even(self):
        self.assertEqual(setup.layer_ranges(48, CMP), [(0, 12), (12, 24), (24, 36), (36, 48)])
        self.assertEqual(setup.qsa_layers(0, 12), 3)      # layers 3, 7, 11

    def test_1m_fp16_kv_over_four_64gb_cards_fits(self):
        p = setup.vram_resident_plan("Q8_0", CMP, CTX1M, "fp16")
        self.assertTrue(p["fits"], json.dumps(p["cards"], indent=1))
        self.assertAlmostEqual(p["expert_gb"], 128.35, places=1)
        self.assertAlmostEqual(p["kv_total_gb"], 28.6, delta=1.0)
        for c in p["cards"]:
            self.assertLess(c["need_gb"], c["room_gb"])
        # the card that carries the draft layer and the head is the fullest one, and it still fits
        self.assertTrue(p["cards"][-1]["need_gb"] == max(c["need_gb"] for c in p["cards"]))

    def test_too_few_cards_cannot_hold_it(self):
        self.assertFalse(setup.vram_resident_plan("Q8_0", cards(1), 8192)["fits"])   # 128 GB of experts, one card
        self.assertFalse(setup.vram_resident_plan("Q8_0", cards(2), CTX1M)["fits"])
        self.assertFalse(setup.vram_resident_ctx("Q8_0", cards(2), "fp16"))
        # three 64 GB cards do hold it (42.8 GB of experts + 8.6 GB of cache + 8.5 GB of their own per card), so
        # the fourth is headroom rather than a requirement: the plan has to say so, not assume four
        self.assertTrue(setup.vram_resident_plan("Q8_0", cards(3), CTX1M, "fp16")["fits"])

    def test_a_bigger_kv_format_costs_what_the_arithmetic_says(self):
        a = setup.vram_resident_plan("Q8_0", CMP, CTX1M, "int8")
        b = setup.vram_resident_plan("Q8_0", CMP, CTX1M, "fp16")
        self.assertLess(a["kv_total_gb"], b["kv_total_gb"])
        self.assertAlmostEqual(b["kv_total_gb"] / a["kv_total_gb"], 2048 / 1056, places=1)

    def test_the_recommended_context_grows_with_the_cards(self):
        self.assertEqual(setup.vram_resident_ctx("Q8_0", CMP, "fp16"), CTX1M)
        self.assertLess(setup.vram_resident_ctx("Q8_0", cards(4, 24.0), "fp16"), CTX1M)
        # two 64 GB cards cannot even hold the experts: no context is recommended
        self.assertEqual(setup.vram_resident_ctx("Q8_0", cards(2), "fp16"), 0)


class PleTier(unittest.TestCase):
    """The n-gram table is the one thing this mode keeps off the GPUs, and where it goes depends on its size and
    the RAM beside it.  54.4 GB (Q8_0's own rows) and 28.8 GB (the IQ4_NL table) are different answers."""

    def test_locked_in_ram_only_when_there_is_ram_to_lock(self):
        with mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.ple_tier(54.4, 128.0), "ram")      # 54 of it, 74 left: lockable
            self.assertEqual(setup.ple_tier(28.8, 64.0), "ram")       # the IQ4_NL table on the same PC
            self.assertEqual(setup.ple_tier(54.4, 66.0), "mmap")      # fits as cache, not enough beside it to lock
            self.assertEqual(setup.ple_tier(54.4, 64.0), "direct")   # 64 GB of RAM cannot hold 54 of table
            self.assertEqual(setup.ple_tier(54.4, 65.0), "mmap")

    def test_windows_never_locks(self):
        with mock.patch.object(setup, "WIN", True):
            self.assertEqual(setup.ple_tier(28.8, 256.0), "mmap")


class ShardList(unittest.TestCase):
    """Unsloth splits by file size, so setup cannot know an unpinned quantization's file count: `"shards": 0`
    reads the repository's own listing, and the count comes from the -0000i-of-0000N names."""

    def test_a_pinned_family_still_uses_its_template(self):
        fam = setup.FAMILIES["unsloth"]
        self.assertEqual(setup.model_shard_names(fam, "UD-Q4_K_XL"),
                         [fam["file"].format(q="UD-Q4_K_XL", i=i) for i in range(1, 5)])

    def test_the_count_comes_from_the_listing(self):
        listing = {f"Qwen3.8-Flash-Next-Q8_0-0000{i}-of-00006.gguf": 32_000_000_000 for i in range(1, 7)}
        with mock.patch.object(setup, "hf_tree_files", lambda url, sub: dict(listing)):
            names = setup.model_shard_names(setup.FAMILIES["unsloth"], "Q8_0")
        self.assertEqual(len(names), 6)
        self.assertEqual(names[0], "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf")

    def test_a_partial_listing_is_refused_rather_than_guessed(self):
        listing = {"Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf": 1}
        with mock.patch.object(setup, "hf_tree_files", lambda url, sub: dict(listing)):
            self.assertIsNone(setup.model_shard_names(setup.FAMILIES["unsloth"], "Q8_0"))

    def test_no_listing_is_no_answer(self):
        with mock.patch.object(setup, "hf_tree_files", lambda url, sub: None):
            self.assertIsNone(setup.model_shard_names(setup.FAMILIES["unsloth"], "Q8_0"))

    def test_the_repo_and_revision_are_read_back_out_of_a_download_url(self):
        got = setup.hf_repo_revision(setup.FAMILIES["unsloth"]["hf"].format(q="Q8_0"))
        self.assertEqual(got, ("unsloth/Qwen3.8-Flash-Next-GGUF", setup.HF_REVISIONS["unsloth/Qwen3.8-Flash-Next-GGUF"]))


class Q8Model(unittest.TestCase):
    def test_the_expert_total_is_the_geometry_not_a_remembered_number(self):
        # 512 experts x 48 layers, each gate/up/down = 640x2560 values, Q8_0 = 34 bytes per 32 values
        blob = 3 * 640 * 2560 * 34 // 32
        self.assertEqual(blob, setup.MODELS["Q8_0"]["expert_bytes"])
        self.assertAlmostEqual(setup.MODELS["Q8_0"]["arena_gb"], blob * 512 * 48 / 1e9, places=1)

    def test_the_table_row_is_five_q8_0_blocks(self):
        self.assertAlmostEqual(setup.MODELS["Q8_0"]["ple_bytes"] / 320001536, 170.0, places=3)

    def test_it_is_offered_for_no_other_family(self):
        self.assertEqual(setup.MODELS["Q8_0"]["families"], ("unsloth",))
        self.assertTrue(setup.MODELS["Q8_0"]["vram_only"])
        self.assertNotIn("Q8_0", [m for m in setup.MODELS if "qwen" in setup.MODELS[m].get("families", ())])

    def test_1m_is_a_context_the_setup_knows(self):
        self.assertEqual(setup.CONTEXTS[-1], CTX1M)
        # and the RAM rule for a streamed cache still stops at 512K: 1M is the cards' decision, not RAM's
        self.assertLessEqual(setup.RAM_CTX_MAX, 524288)


class Config(unittest.TestCase):
    """One mocked install on the four-card PC: the config setup actually writes."""

    def found(self):
        return [{"index": i, "name": "NVIDIA CMP 170HX", "vram_gb": 64.0, "arch": "80", "driver": "580.0"}
                for i in range(4)]

    def test_the_args_of_a_four_card_q8_0_setup(self):
        import test_setup_golden as g
        code, out, cfg, asked = g.install(128.0, self.found(),
                                         ["--family", "unsloth", "--model", "Q8_0", "--context", str(CTX1M),
                                          "--gpus", "0,1,2,3", "--no-start"], engine="0.1.35")
        self.assertEqual(code, 0, out[-3000:])
        a = cfg["args"]
        self.assertIn("--all-experts-vram", a)
        self.assertIn("--mmap-experts", a)                    # so an older engine fails instead of copying 128 GB
        self.assertEqual(a[a.index("--kv") + 1], "fp16")
        self.assertEqual(a[a.index("--max-context") + 1], str(CTX1M))
        self.assertNotIn("--kv-resident", a)                  # the cache is not streamed: it is the point
        self.assertNotIn("--resident-budget-gib", a)
        self.assertEqual(a.count("--mmap-experts"), 1)      # written once, not implied twice
        self.assertEqual(cfg["gpu"], [0, 1, 2, 3])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertIn("--rope-scaling", a)                    # 1M is past the trained 262,144
        self.assertEqual(a[a.index("--rope-scaling") + 1], "yarn")
        self.assertEqual(a[a.index("--rope-scale") + 1], "4")
        self.assertIn("--ple-io", a)
        self.assertNotIn("--vision", a)                       # the encoder is not wired to this file

    def test_a_single_card_is_refused_before_anything_is_downloaded(self):
        import test_setup_golden as g
        code, out, cfg, asked = g.install(128.0, self.found()[:1],
                                         ["--family", "unsloth", "--model", "Q8_0", "--no-start"], engine="0.1.35")
        self.assertNotEqual(code, 0)
        self.assertIn("cannot hold them", out)

    def test_an_engine_without_the_flag_is_refused_before_the_download(self):
        import test_setup_golden as g
        code, out, cfg, asked = g.install(128.0, self.found(),
                                         ["--family", "unsloth", "--model", "Q8_0", "--gpus", "0,1,2,3",
                                          "--no-start"], engine="0.1.34")
        self.assertNotEqual(code, 0)
        self.assertIn("--all-experts-vram", out)

    def test_the_plan_is_printed_for_the_user(self):
        import test_setup_golden as g
        code, out, cfg, asked = g.install(128.0, self.found(),
                                         ["--family", "unsloth", "--model", "Q8_0", "--context", str(CTX1M),
                                          "--gpus", "0,1,2,3", "--no-start"], engine="0.1.35")
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("every expert fits", out)
        self.assertIn("KV streaming off", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
