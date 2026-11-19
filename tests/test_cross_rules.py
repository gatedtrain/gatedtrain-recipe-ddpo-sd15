"""gatedtrain's generic rules exercised through `ddpo_sd15`, moved from gatedtrain's SD3.5, CAD and sampler-rule tests with the
recipe: a boolean step number refuses (review 10 L3), a mixed-type `deployable` list is an incomplete cut (L1), a
base with a `.cache/` directory refuses (the exclusion is the Flow-GRPO adapters' only), and a `shared_annotation`
statement is refused for a deciding evaluator whose row declares none (05 section 6.1, R16-1; EXT-147)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import _ddpo as D
from gatedtrain.testing.support import EXIT_REFUSED, describe, parse_errors


class TestCrossRules(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ddpo-cross-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_boolean_step_refused(self):
        sc = D.DdpoScenario(self.tmp / "d", n_items=12)
        steps = D.read_jsonl(sc.stage_out / "train_steps.jsonl")
        steps[0]["step"] = True
        D.write_jsonl(sc.stage_out / "train_steps.jsonl", steps)
        proc = sc.collect(self.tmp / "runs-d")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-CONSUMPTION"
                         and "consecutive" in e["message"]], proc.stderr)

    def test_mixed_type_deployable(self):
        from gatedtrain.ext import registry
        other = D.DdpoScenario(self.tmp / "l1")
        d = other.final_cut
        doc = json.loads((d / "cut.json").read_text())
        doc["deployable"] = [1, "x"]
        (d / "cut.json").write_text(json.dumps(doc))
        step = int(d.name.split("-")[1])
        self.assertIn("deployable", registry.adapter(D.ADAPTER).cut_problem(d, step))

    def test_base_cache_refused(self):
        """The `.cache/` exclusion is the Flow-GRPO adapters' (decision); this base with a `.cache/` still refuses."""
        sc = D.DdpoScenario(self.tmp / "c", n_items=12)
        (sc.base / ".cache").mkdir()
        proc = sc.collect(self.tmp / "runs-c")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-BASE" and ".cache" in e["message"]])


class TestSharedAnnotation(unittest.TestCase):
    """A `shared_annotation` statement is refused for a deciding evaluator whose row declares none (05 section 6.1,
    R16-1; EXT-147)."""

    def fields(self, reg: dict, evaluator: str) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)]

    def test_shared_annotation_refused_where_not_declared(self):
        """The statement is refused for the WebP evaluator (its row declares no shared annotation); its
        construct_alignment rule is unchanged."""
        text = (D.DDPO / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(D.SAMPLER))
        reg = yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))
        self.assertEqual(self.fields(reg, D.EVALUATOR_ID), [])
        reg["shared_annotation"] = "The references share the training labels' annotation."
        self.assertEqual(self.fields(reg, D.EVALUATOR_ID), ["registration.shared_annotation"])
        del reg["shared_annotation"], reg["construct_alignment"]
        self.assertEqual(self.fields(reg, D.EVALUATOR_ID), ["registration.construct_alignment"])


if __name__ == "__main__":
    unittest.main()
