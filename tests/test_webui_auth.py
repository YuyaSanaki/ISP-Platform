"""Optional Web UI login (ISP_WEBUI_USERS_FILE) and the approver it records."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import webui_auth as auth  # noqa: E402


class TestPasswordHashes(unittest.TestCase):
    def test_round_trip(self):
        encoded = auth.hash_password("s3cret", iterations=1000)
        self.assertTrue(encoded.startswith("pbkdf2_sha256$1000$"))
        self.assertTrue(auth.verify_password("s3cret", encoded))
        self.assertFalse(auth.verify_password("wrong", encoded))

    def test_malformed_hash_never_matches(self):
        for bad in ("", "plain", "md5$1$00$00", "pbkdf2_sha256$x$zz$00"):
            self.assertFalse(auth.verify_password("x", bad))

    def test_check_login_unknown_user(self):
        users = {"alice": auth.hash_password("pw", iterations=1000)}
        self.assertTrue(auth.check_login(users, "alice", "pw"))
        self.assertFalse(auth.check_login(users, "alice", "nope"))
        self.assertFalse(auth.check_login(users, "mallory", "pw"))

    def test_users_file_must_list_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "users.yaml"
            path.write_text("users: {}\n", encoding="utf-8")
            with self.assertRaises(auth.AuthConfigError):
                auth.load_users(path)
            path.write_text(yaml.safe_dump({"users": {"bob": "h"}}), encoding="utf-8")
            self.assertEqual(auth.load_users(path), {"bob": "h"})


class TestApprovalRecordsLoggedInUser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_ortholog_loss_gate import gate

        cls.gate = gate
        blocked_run = {"input_genes_sha256": "0" * 64, "species": {"model_organism": "human"}}
        cls.request = {
            "status": "pending",
            "blocked_run": blocked_run,
            "request_id": gate.mint_request_id(blocked_run),
            "blocked_genes": ["POU5F1"],
        }

    def test_records_authenticated_user(self):
        record = self.gate.build_approved_record(
            self.request, decision="stop", approved_by="alice", reason="r",
            authenticated_user="alice",
        )
        self.assertEqual(record["authenticated_user"], "alice")

    def test_approver_must_match_login(self):
        with self.assertRaises(self.gate.OrthologApprovalError):
            self.gate.build_approved_record(
                self.request, decision="reject", approved_by="bob", reason="r",
                authenticated_user="alice",
            )

    def test_without_login_no_authenticated_user(self):
        record = self.gate.build_approved_record(
            self.request, decision="stop", approved_by="anyone", reason="r",
        )
        self.assertNotIn("authenticated_user", record)


if __name__ == "__main__":
    unittest.main()
