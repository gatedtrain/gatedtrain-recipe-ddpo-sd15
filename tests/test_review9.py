"""Review round 9 cases of the `ddpo_sd15` adapter (EXT-154, EXT-156, EXT-158, EXT-159), moved from gatedtrain's tests with the
recipe: unreadable run and data files refuse (EXT-154), the evaluation scripts' inputs and the script itself are bound
at launch (EXT-156), a step log longer than the run refuses (EXT-158), the area's construct-alignment messages are
unchanged by the disclosure table (EXT-159).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from gatedtrain.testing.support import EXIT_REFUSED, describe, parse_errors

import _ddpo as D


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def _root_can_read() -> bool:
    return os.geteuid() == 0


class Case(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-review9-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def path(self, name: str) -> Path:
        type(self).n += 1
        return self.tmp / f"{name}-{type(self).n}"

    def unreadable(self, p: Path) -> None:
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)

    def refused(self, proc, rule: str, field_prefix: str | None = None, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix} {text!r}; got {errs}\n{describe(proc)}")
        return hits


class TestUnreadableDdpo(Case):

    def sc(self):
        return D.DdpoScenario(self.path("d"), n_items=12)

    def test_shard(self):
        sc = self.sc()
        self.unreadable(sc.data / "train.jsonl")
        self.refused(sc.collect(self.path("runs")), "SPEC-REF", "artifacts.prompts.uri")

    def test_last_cut_file(self):
        sc = self.sc()
        self.unreadable(sc.final_cut / "trainer_state.json")
        self.refused(sc.collect(self.path("runs")), "EXT-STAGE", "--stage-out", "cannot be read")

    def test_freeze(self):
        sc = self.sc()
        self.unreadable(sc.stage_out / "env-freeze.txt")
        self.refused(sc.collect(self.path("runs")), "EXT-FREEZE", "--stage-out")

    def test_checks_on_an_unreadable_shard(self):
        sc = self.sc()
        self.unreadable(sc.data / "train.jsonl")
        launch = json.loads(sc.launch.read_text())
        probs = registry().adapter(D.ADAPTER).checks(last_cut=sc.final_cut, base=sc.base, launch=launch,
                                                     base_field="b", stage_out=sc.stage_out, data=sc.data,
                                                     data_field="d")
        self.assertIn("EXT-DATA", [p.rule for p in probs])


class TestEvalArgv(unittest.TestCase):

    A3 = "/srv/recipes/ddpo-sd15/runtime"

    def problems(self, adapter: str, argv: list, profile: str | None = None) -> list[str]:
        return registry().adapter(adapter).argv_problems(argv, profile=profile)

    def cases(self):
        return (("ddpo_sd15", self.A3, "eval_images.py", "train_ddpo.py", "11,12"),)

    def test_items_default_only(self):
        for adapter, rt, ev, _, seeds in self.cases():
            with self.subTest(adapter=adapter):
                base = [f"{rt}/.venv/bin/python", f"{rt}/{ev}", "--base-only", "--out", "/out", "--eval-seeds", seeds]
                self.assertEqual(self.problems(adapter, base), [])
                self.assertEqual(self.problems(adapter, base + ["--items", f"{rt}/items"]), [])
                for bad in (["--items", "/inputs/items-copy"], ["--items=/inputs/items-copy"],
                            ["--items", f"{rt}/items/../items-copy"]):
                    self.assertTrue(self.problems(adapter, base + bad), bad)

    def test_script_pinned(self):
        """argv[1] is one of the adapter's scripts in the runtime directory argv[0] (the pinned interpreter) names:
        a byte-identical copy elsewhere (another input root) brings its own base and items."""
        for adapter, rt, ev, tr, _ in self.cases():
            with self.subTest(adapter=adapter):
                py = f"{rt}/.venv/bin/python"
                self.assertEqual(self.problems(adapter, [py, f"{rt}/{tr}", "--profile", "smoke", "--out", "/out"]), [])
                for bad in (f"/srv/outputs/ddpo-sd15/x/{ev}", f"{rt}/make_data.py", f"{rt}/sub/{ev}"):
                    self.assertTrue(self.problems(adapter, [py, bad, "--out", "/out"]), bad)

    def test_limit_refused_for_full(self):
        for adapter, rt, ev, _, seeds in self.cases():
            with self.subTest(adapter=adapter):
                argv = [f"{rt}/.venv/bin/python", f"{rt}/{ev}", "--base-only", "--out", "/out", "--limit", "20"]
                self.assertEqual(self.problems(adapter, argv, "smoke"), [])
                self.assertTrue(self.problems(adapter, argv, "full"))
                self.assertTrue(self.problems(adapter, argv[:-2] + ["--limit=20"], "full"))
                self.assertEqual(self.problems(adapter, argv[:-2], "full"), [])


class TestLogBeyondTheRun(Case):


    def test_ddpo_extra_epoch(self):
        """The ddpo analogue: steps of an epoch after settings.epochs (and their samples) refuse."""
        sc = D.DdpoScenario(self.path("d"), n_items=12)
        steps = D.read_jsonl(sc.stage_out / "train_steps.jsonl")
        steps.append(dict(steps[-1], step=steps[-1]["step"] + 1, epoch=4))
        D.write_jsonl(sc.stage_out / "train_steps.jsonl", steps)
        self.refused(sc.collect(self.path("runs")), "EXT-CONSUMPTION", "--stage-out", "settings.epochs")


class TestDisclosureTable(unittest.TestCase):
    def ddpo_reg(self) -> dict:
        text = (D.DDPO / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(D.SAMPLER))
        return yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def problems(self, reg, evaluator):
        from gatedtrain.registration import check_contents
        return [(p.field, p.message, p.fix) for p in check_contents(reg, evaluator_id=evaluator)
                if p.field.endswith(("construct_alignment", "shared_annotation"))]

    def test_area3_messages_unchanged(self):
        """The messages area 3 had before the table (review 9 L3 a)."""
        stmt = registry().EVALUATOR_ROWS[D.EVALUATOR_ID]["construct_aligned"]["statement"]
        reg = self.ddpo_reg()
        del reg["construct_alignment"]
        self.assertEqual(self.problems(reg, D.EVALUATOR_ID), [(
            "registration.construct_alignment",
            "the deciding evaluator t2i.sd15_webp_items@0.1 is construct-aligned with the training reward "
            "jpeg_compressibility (05 section 6.1, R15-1), but the registration does not state what a pass shows",
            f"add `construct_alignment:` stating, for example: {stmt}")])
        reg["construct_alignment"] = "  "
        self.assertEqual(self.problems(reg, D.EVALUATOR_ID), [(
            "registration.construct_alignment", "construct_alignment is not a non-empty text",
            "state what a pass of the construct-aligned metric shows (05 section 6.1, R15-1)")])
        reg = self.ddpo_reg()
        reg["evaluators"] = [{"id": D.SUPPORT_ID, "manifest_hash": "sha256:" + "c" * 64}]
        reg["clause"]["metric"] = D.SUPPORT_METRIC
        self.assertEqual(self.problems(reg, D.SUPPORT_ID), [(
            "registration.construct_alignment",
            "the deciding evaluator t2i.sd15_jpeg_items@0.1 is not construct-aligned with a training reward (its "
            "registry row declares none), so the statement would claim what the gate does not show (05 section "
            "6.1, R15-1)", "remove `construct_alignment`")])


if __name__ == "__main__":
    unittest.main()
