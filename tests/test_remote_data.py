"""Tests for URL → study import (streamlit_remote_data)."""
from __future__ import annotations

import sys

import http.server
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "core", ROOT / "contracts", ROOT / "webui"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)



class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return


def _serve_directory(directory: Path) -> tuple[str, threading.Thread, http.server.HTTPServer]:
    directory = directory.resolve()

    class Handler(_QuietHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", thread, server


class TestRemoteDataImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = (
            ROOT
            / "data"
            / "streamlit_workspace"
            / "_fixtures"
            / "smoke_mouse_url_test.zip"
        )
        if not cls.fixture.is_file():
            raise unittest.SkipTest(f"Missing fixture zip: {cls.fixture}")

    def test_validate_rejects_non_http(self) -> None:
        from streamlit_remote_data import RemoteDataError, validate_data_url

        with self.assertRaises(RemoteDataError):
            validate_data_url("ftp://example.com/a.zip")
        with self.assertRaises(RemoteDataError):
            validate_data_url("")
        self.assertTrue(
            validate_data_url("https://example.com/a.zip").startswith("https://")
        )

    def test_rewrite_google_drive_and_dropbox(self) -> None:
        from streamlit_remote_data import rewrite_share_url

        drive = rewrite_share_url(
            "https://drive.google.com/file/d/abc123XYZ/view?usp=sharing"
        )
        self.assertIn("uc?export=download", drive)
        self.assertIn("id=abc123XYZ", drive)

        drive2 = rewrite_share_url(
            "https://drive.google.com/open?id=fileid999"
        )
        self.assertIn("id=fileid999", drive2)

        drop = rewrite_share_url(
            "https://www.dropbox.com/s/abcdef/study.zip?dl=0"
        )
        self.assertIn("dl=1", drop)

    def test_soft_link_geo_guidance(self) -> None:
        from streamlit_remote_data import (
            RemoteDataError,
            geo_series_ftp_prefix,
            pick_geo_supplementary_file,
            resolve_download_url,
        )

        self.assertEqual(geo_series_ftp_prefix("GSE12345"), "GSE12nnn")
        self.assertEqual(geo_series_ftp_prefix("GSE1"), "GSEnnn")
        self.assertEqual(
            pick_geo_supplementary_file(
                ["filelist.txt", "GSE1_series_matrix.txt.gz", "GSE1_RAW.tar.gz"]
            ),
            "GSE1_RAW.tar.gz",
        )
        self.assertEqual(
            pick_geo_supplementary_file(["notes.txt", "counts.zip"]),
            "counts.zip",
        )

        # GSM / SRA still guided (not auto-resolved)
        with self.assertRaises(RemoteDataError) as ctx_gsm:
            resolve_download_url(
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM12345"
            )
        self.assertIn("GEO", str(ctx_gsm.exception))

        with self.assertRaises(RemoteDataError) as ctx_sra:
            resolve_download_url(
                "https://www.ncbi.nlm.nih.gov/sra/?term=SRR123456"
            )
        self.assertIn("SRA", str(ctx_sra.exception))

        # Collection page (no dataset UUID) → guidance
        with self.assertRaises(RemoteDataError) as ctx2:
            resolve_download_url("https://cellxgene.cziscience.com/collections/abc")
        self.assertIn("CELLxGENE", str(ctx2.exception))

        # Direct archive on GEO FTP-style path is allowed
        ok = resolve_download_url(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345/suppl/GSE12345_RAW.tar.gz"
        )
        self.assertTrue(ok.endswith(".tar.gz") or ".tar.gz" in ok)

    def test_geo_accession_resolves_to_ftp_archive(self) -> None:
        from streamlit_remote_data import resolve_download_url

        # Live NCBI FTP index for a tiny public series (GSE12345 → RAW.tar).
        try:
            resolved = resolve_download_url(
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345"
            )
        except Exception as exc:  # noqa: BLE001 — network / NCBI outage
            raise unittest.SkipTest(f"GEO FTP unreachable: {exc}") from exc
        self.assertIn("ftp.ncbi.nlm.nih.gov/geo/series/", resolved)
        self.assertIn("GSE12345", resolved)
        self.assertTrue(
            resolved.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip")),
            resolved,
        )

    def test_cellxgene_dataset_h5ad_is_rejected_clearly(self) -> None:
        from streamlit_remote_data import RemoteDataError, resolve_download_url

        # Public dataset UUID from CELLxGENE curation API (H5AD-only assets).
        url = (
            "https://cellxgene.cziscience.com/e/"
            "fb90c70d-5917-4d03-920b-c78b230b51e5.cxg/"
        )
        try:
            resolve_download_url(url)
            self.fail("expected RemoteDataError for H5AD-only CELLxGENE dataset")
        except RemoteDataError as exc:
            msg = str(exc)
            self.assertIn("h5ad", msg.lower())
            self.assertIn("10x", msg.lower())
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"CELLxGENE API unreachable: {exc}") from exc

    def test_download_streams_to_disk(self) -> None:
        from streamlit_remote_data import download_url_to_file

        base, _thread, server = _serve_directory(self.fixture.parent)
        try:
            url = f"{base}/{self.fixture.name}"
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "out.bin"
                name = download_url_to_file(url, dest)
                self.assertEqual(name, self.fixture.name)
                self.assertEqual(dest.stat().st_size, self.fixture.stat().st_size)
                self.assertEqual(dest.read_bytes()[:4], b"PK\x03\x04")
        finally:
            server.shutdown()
            server.server_close()

    def test_download_and_import_from_url(self) -> None:
        from data_input_layout import discover_sample_dirs
        from streamlit_remote_data import import_study_from_url

        base, _thread, server = _serve_directory(self.fixture.parent)
        try:
            url = f"{base}/{self.fixture.name}"
            upload_dir = ROOT / "data" / "streamlit_workspace" / "_test_url_import"
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            tokenize_dir, summary, filename = import_study_from_url(
                url, upload_dir, "url_smoke"
            )
            self.assertEqual(filename, self.fixture.name)
            samples = discover_sample_dirs(tokenize_dir)
            self.assertGreaterEqual(len(samples), 2)
            self.assertIn("sample", summary.lower())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
