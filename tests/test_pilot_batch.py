"""Checks the pilot annotation batch against the raw Amazon file. Skipped where the local data files are absent (data/ is git-ignored)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_pilot_batch as mp

pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in (mp.RAW, mp.BATCH, mp.MANIFEST, mp.TEMPLATE)),
                                reason="local data or pilot batch not present")


def test_pilot_batch_is_genuine_blank_and_leaves_the_template_alone(capsys):
    assert mp.verify() == []                  # product, timestamp, text, order, reviewers, blanks, overlap, template hash
    capsys.readouterr()


def test_pilot_batch_refuses_to_overwrite_an_existing_batch():
    with pytest.raises(SystemExit):
        mp.build()
