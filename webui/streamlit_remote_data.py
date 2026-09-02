"""
Download deposited study data from a URL for the Streamlit UI.

Primary format: a .zip of 10x sample folders (same layout as local upload).
Accepts http(s) direct links (Zenodo, Figshare, GitHub Releases, Hugging Face, …)
and rewrites common share-page URLs (Google Drive, Dropbox).
"""
from __future__ import annotations

import html as html_lib
import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from streamlit_upload import (
    import_study_zip_path,
    normalize_study_name,
)

# Soft cap to avoid filling the disk from a mistaken huge URL (override via arg).
DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB
DEFAULT_TIMEOUT_S = 600

_ProgressCb = Callable[[int, int | None], None]  # (downloaded, total_or_None)
_USER_AGENT = "Geneformer-Platform/1.0 (+streamlit-remote-data)"


class RemoteDataError(ValueError):
    """User-facing download / import error."""


def validate_data_url(url: str) -> str:
    """Normalize and validate an http(s) URL. Raises RemoteDataError on bad input."""
    raw = (url or "").strip()
    if not raw:
        raise RemoteDataError("Enter a dataset URL (https://…).")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise RemoteDataError(
            f"Only http(s) URLs are supported (got `{parsed.scheme or 'none'}`)."
        )
    if not parsed.netloc:
        raise RemoteDataError("URL is missing a host name.")
    return raw


def rewrite_share_url(url: str) -> str:
    """
    GAP-URL-2: turn common share-page links into direct-download URLs.

    - Google Drive file/view or open?id= → uc?export=download&id=
    - Dropbox dl=0 / www.dropbox.com → dl=1
    """
    url = validate_data_url(url)
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    # Dropbox
    if "dropbox.com" in host:
        qs = parse_qs(parsed.query, keep_blank_values=True)
        qs["dl"] = ["1"]
        # dropbox.com/s/... and dl.dropboxusercontent.com both work with dl=1
        new_query = urlencode({k: v[0] for k, v in qs.items()})
        return urlunparse(parsed._replace(query=new_query))

    # Google Drive
    if "drive.google.com" in host or "docs.google.com" in host:
        file_id = None
        m = re.search(r"/file/d/([^/]+)", parsed.path)
        if m:
            file_id = m.group(1)
        if not file_id:
            qs = parse_qs(parsed.query)
            if "id" in qs and qs["id"]:
                file_id = qs["id"][0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def _looks_like_archive_url(url: str) -> bool:
    return bool(re.search(r"\.(zip|tar\.gz|tgz|tar)(\?|$)", url.lower()))


def geo_series_ftp_prefix(accession: str) -> str:
    """Map GSE12345 → GSE12nnn (NCBI GEO series FTP folder convention)."""
    acc = accession.strip().upper()
    if not acc.startswith("GSE"):
        raise ValueError(f"Not a GSE accession: {accession}")
    num = acc[3:]
    if not num.isdigit():
        raise ValueError(f"Not a GSE accession: {accession}")
    if len(num) <= 3:
        return "GSEnnn"
    return f"GSE{num[:-3]}nnn"


def _extract_gse_accession(url: str) -> str | None:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    for key in ("acc", "ACC"):
        if key in qs and qs[key]:
            val = qs[key][0].strip().upper()
            if val.startswith("GSE") and val[3:].isdigit():
                return val
    m = re.search(r"(GSE\d+)", url, flags=re.I)
    return m.group(1).upper() if m else None


def _parse_ftp_index_filenames(html: str) -> list[str]:
    """Extract href filenames from a simple Apache-style directory index."""
    names: list[str] = []
    for m in re.finditer(r'href="([^"]+)"', html, flags=re.I):
        name = unquote(m.group(1)).rstrip("/")
        if not name or name in (".", "..") or name.startswith("?") or "/" in name:
            continue
        if name.lower().startswith("parent"):
            continue
        names.append(Path(name).name)
    return names


def pick_geo_supplementary_file(filenames: list[str]) -> str | None:
    """
    Prefer archives likely to hold deposited matrices.

    Order: RAW tar.gz/zip → any tar.gz/tgz/zip → RAW.tar → any .tar
    """
    files = [Path(f).name for f in filenames]
    lower = {f: f.lower() for f in files}

    def _first(pred) -> str | None:
        for f in files:
            if pred(lower[f]):
                return f
        return None

    pick = _first(
        lambda n: "raw" in n and (n.endswith(".tar.gz") or n.endswith(".tgz") or n.endswith(".zip"))
    )
    if pick:
        return pick
    pick = _first(lambda n: n.endswith((".tar.gz", ".tgz", ".zip")))
    if pick:
        return pick
    pick = _first(lambda n: "raw" in n and n.endswith(".tar"))
    if pick:
        return pick
    return _first(lambda n: n.endswith(".tar"))


def resolve_geo_supplementary_url(
    url: str,
    *,
    timeout_s: int = 60,
    user_agent: str = _USER_AGENT,
) -> str | None:
    """
    GAP-URL-1: turn a GEO accession page into an HTTPS FTP supplementary archive URL.

    Returns None if the URL is not a GEO GSE page or no archive is listed.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    if "ncbi.nlm.nih.gov" not in host:
        return None
    if not (
        "/geo/query/acc.cgi" in path
        or "acc=gse" in query
        or re.search(r"/geo/.*/gse\d+", path)
    ):
        return None
    if _looks_like_archive_url(url):
        return None

    acc = _extract_gse_accession(url)
    if not acc:
        return None

    prefix = geo_series_ftp_prefix(acc)
    index_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/suppl/"
    req = Request(index_url, headers={"User-Agent": user_agent})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            html = resp.read(512 * 1024).decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError):
        return None

    chosen = pick_geo_supplementary_file(_parse_ftp_index_filenames(html))
    if not chosen:
        return None
    return f"{index_url}{chosen}"


def _extract_cellxgene_dataset_id(url: str) -> str | None:
    """Pull a dataset UUID from common CELLxGENE explorer / API URL shapes."""
    parsed = urlparse(url)
    path = parsed.path
    # /e/<uuid>.cxg/  or  /datasets/<uuid>
    m = re.search(
        r"(?:/e/|/datasets/)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        path,
        flags=re.I,
    )
    if m:
        return m.group(1).lower()
    m = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.cxg",
        path,
        flags=re.I,
    )
    return m.group(1).lower() if m else None


def resolve_cellxgene_asset_url(
    url: str,
    *,
    timeout_s: int = 60,
    user_agent: str = _USER_AGENT,
) -> str | None:
    """
    GAP-URL-1: resolve a CELLxGENE dataset page to a downloadable asset URL.

    CELLxGENE curation assets are almost always `.h5ad` (not 10x zip). When only
    H5AD is available this raises RemoteDataError with a clear conversion hint.
    Returns None if the URL is not a dataset page (e.g. collection landing).
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "cellxgene" not in host and "cxg-hub" not in host and "cziscience.com" not in host:
        return None
    if _looks_like_archive_url(url) or url.lower().rstrip("/").endswith(".h5ad"):
        return None
    # Collection pages have no single dataset — leave to soft guidance.
    if "/collections/" in parsed.path.lower() and "/datasets/" not in parsed.path.lower():
        return None

    dataset_id = _extract_cellxgene_dataset_id(url)
    if not dataset_id:
        return None

    api = f"https://api.cellxgene.cziscience.com/curation/v1/datasets/{dataset_id}"
    req = Request(api, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None

    assets = payload.get("assets") or []
    zip_like = [
        a
        for a in assets
        if str(a.get("filetype", "")).upper() in {"ZIP", "TAR", "TAR.GZ"}
        or str(a.get("url", "")).lower().endswith((".zip", ".tar.gz", ".tgz"))
    ]
    if zip_like:
        return zip_like[0].get("url") or None

    h5ads = [
        a
        for a in assets
        if str(a.get("filetype", "")).upper() == "H5AD"
        or str(a.get("url", "")).lower().endswith(".h5ad")
    ]
    if h5ads:
        asset_url = h5ads[0].get("url") or "(unknown)"
        title = payload.get("title") or dataset_id
        raise RemoteDataError(
            f"CELLxGENE dataset **{title}** resolves to an `.h5ad` asset "
            f"(`{asset_url}`), but this importer needs a **.zip / .tar.gz of "
            "10x sample folders** (barcodes / features / matrix).\n\n"
            "Convert the H5AD to 10x folders (or re-export from Cell Ranger), "
            "zip them, and paste that archive URL."
        )
    return None


def _soft_link_guidance(url: str) -> str | None:
    """
    GAP-URL-1 fallback: guidance when a landing page cannot be auto-resolved.

    Returns a guidance message, or None if the URL looks like a file download.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    if _looks_like_archive_url(url):
        return None

    if "ncbi.nlm.nih.gov" in host and (
        "/geo/query/acc.cgi" in path or "acc=gse" in query or "acc=gsm" in query
    ):
        return (
            "This looks like an **NCBI GEO accession page**, and no supplementary "
            "`.zip` / `.tar.gz` / `.tar` archive was found automatically.\n\n"
            "Open the GEO record → **Supplementary file** → copy the direct "
            "FTP/HTTPS link (often under "
            "`ftp.ncbi.nlm.nih.gov/geo/series/.../suppl/`), then paste that URL here.\n\n"
            "Note: GEO RAW dumps are only useful here if they already contain "
            "10x count matrices."
        )

    if "ncbi.nlm.nih.gov" in host and ("/sra" in path or "acc=srr" in query):
        return (
            "This looks like an **SRA** page. Geneformer needs 10x count matrices "
            "(barcodes/features/matrix), not raw reads. Process with Cell Ranger "
            "(or equivalent), zip the sample folders, and host that archive."
        )

    if "cellxgene" in host or "cxg-hub" in host:
        if path.rstrip("/").endswith((".h5ad", ".zip", ".tar.gz", ".tgz")):
            return None
        return (
            "This looks like a **CELLxGENE** collection/dataset page.\n\n"
            "Open a **dataset** (not only a collection), or paste a direct "
            "`.zip` / `.tar.gz` of 10x folders. CELLxGENE `.h5ad` downloads are "
            "not imported as-is — convert to 10x folders first."
        )

    return None


def resolve_download_url(url: str) -> str:
    """Validate, rewrite share links, auto-resolve GEO when possible, else guide."""
    url = rewrite_share_url(url)

    if url.lower().rstrip("/").split("?", 1)[0].endswith(".h5ad"):
        raise RemoteDataError(
            "Direct `.h5ad` URLs are not imported. Convert to a **.zip / .tar.gz "
            "of 10x sample folders** (barcodes / features / matrix) and paste that link."
        )

    geo = resolve_geo_supplementary_url(url)
    if geo:
        return geo

    # May raise RemoteDataError when only H5AD is available.
    cxg = resolve_cellxgene_asset_url(url)
    if cxg:
        return cxg

    guidance = _soft_link_guidance(url)
    if guidance:
        raise RemoteDataError(guidance)
    return url


def _filename_from_response(url: str, headers) -> str:
    cd = headers.get("Content-Disposition") or headers.get("content-disposition") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.I)
    if match:
        return Path(unquote(match.group(1).strip())).name
    path_name = Path(unquote(urlparse(url).path)).name
    return path_name or "download.bin"


def _looks_like_zip_file(path: Path, filename: str) -> bool:
    name = filename.lower()
    if name.endswith(".zip"):
        return True
    with path.open("rb") as fh:
        magic = fh.read(4)
    return magic == b"PK\x03\x04" or magic == b"PK\x05\x06"


def _looks_like_tar_gz_file(path: Path, filename: str) -> bool:
    name = filename.lower()
    if name.endswith((".tar.gz", ".tgz")):
        return True
    # Plain .tar is handled separately — gzip magic alone is not enough to
    # claim tar.gz when the name is unknown.
    if name.endswith(".tar"):
        return False
    with path.open("rb") as fh:
        magic = fh.read(2)
    return magic == b"\x1f\x8b"


def _looks_like_tar_file(path: Path, filename: str) -> bool:
    name = filename.lower()
    if name.endswith(".tar") and not name.endswith((".tar.gz", ".tgz")):
        return True
    # ustar at offset 257
    try:
        with path.open("rb") as fh:
            fh.seek(257)
            return fh.read(5) == b"ustar"
    except OSError:
        return False


def _drive_confirm_url(html: str, file_id: str | None) -> str | None:
    """Parse Google Drive virus-scan interstitial for confirm token / form action."""
    # Modern Drive: <form id="download-form" action="...">
    action = re.search(
        r'action="(https://drive\.google\.com/uc\?[^"]+export=download[^"]*)"',
        html,
        flags=re.I,
    )
    if action:
        return html_lib.unescape(action.group(1)).replace("&amp;", "&")

    token = re.search(r"confirm=([0-9A-Za-z_-]+)", html)
    if token and file_id:
        return (
            f"https://drive.google.com/uc?export=download&id={file_id}"
            f"&confirm={token.group(1)}"
        )
    # uuid confirm in download_warning_ form
    uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    if uuid and file_id:
        return (
            f"https://drive.google.com/uc?export=download&id={file_id}"
            f"&confirm=t&uuid={uuid.group(1)}"
        )
    return None


def _gdrive_id_from_url(url: str) -> str | None:
    m = re.search(r"[?&]id=([^&]+)", url)
    return m.group(1) if m else None


def download_url_to_file(
    url: str,
    dest: Path,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    progress: _ProgressCb | None = None,
    user_agent: str = _USER_AGENT,
    _drive_retry: bool = True,
) -> str:
    """
    GAP-URL-3: stream URL body to ``dest`` on disk (not fully into RAM).

    Returns filename_hint from Content-Disposition / URL path.
    """
    url = resolve_download_url(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "*/*"},
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            final_url = resp.geturl()
            filename = _filename_from_response(final_url, resp.headers)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            total_hdr = resp.headers.get("Content-Length") or resp.headers.get(
                "content-length"
            )
            try:
                total = int(total_hdr) if total_hdr else None
            except ValueError:
                total = None
            if total is not None and total > max_bytes:
                raise RemoteDataError(
                    f"Remote file is {total:,} bytes, over the {max_bytes:,} byte limit."
                )

            # Google Drive sometimes returns an HTML interstitial instead of the file.
            peek = resp.read(512)
            head = peek.lstrip()[:64].lower()
            is_html = (
                "text/html" in ctype
                or head.startswith(b"<!doctype")
                or head.startswith(b"<html")
            )
            if _drive_retry and is_html and (
                b"drive.google" in peek.lower()
                or b"google drive" in peek.lower()
                or b"download-form" in peek.lower()
                or b"uc-download-link" in peek.lower()
            ):
                html = peek + resp.read()
                confirm = _drive_confirm_url(
                    html.decode("utf-8", errors="replace"),
                    _gdrive_id_from_url(url) or _gdrive_id_from_url(final_url),
                )
                if confirm and confirm != url:
                    return download_url_to_file(
                        confirm,
                        dest,
                        timeout_s=timeout_s,
                        max_bytes=max_bytes,
                        progress=progress,
                        user_agent=user_agent,
                        _drive_retry=False,
                    )
                raise RemoteDataError(
                    "Google Drive returned an HTML page instead of the file. "
                    "Check sharing is set to **Anyone with the link**, or paste the "
                    "direct `uc?export=download&id=…` URL."
                )

            downloaded = 0
            with dest.open("wb") as out:
                if peek:
                    out.write(peek)
                    downloaded += len(peek)
                    if progress is not None:
                        progress(downloaded, total)
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    downloaded += len(block)
                    if downloaded > max_bytes:
                        raise RemoteDataError(
                            f"Download exceeded the {max_bytes:,} byte limit; aborted."
                        )
                    out.write(block)
                    if progress is not None:
                        progress(downloaded, total)
    except HTTPError as e:
        raise RemoteDataError(f"HTTP {e.code} downloading `{url}`: {e.reason}") from e
    except URLError as e:
        raise RemoteDataError(f"Could not reach `{url}`: {e.reason}") from e
    except TimeoutError as e:
        raise RemoteDataError(f"Timed out downloading `{url}`") from e

    if not dest.is_file() or dest.stat().st_size == 0:
        raise RemoteDataError("Download returned an empty file.")
    return filename


def download_url_bytes(
    url: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    progress: _ProgressCb | None = None,
    user_agent: str = _USER_AGENT,
) -> tuple[bytes, str]:
    """
    Download URL body (streams to a temp file first, then returns bytes).

    Prefer ``download_url_to_file`` / ``import_study_from_url`` for large archives.
    """
    with tempfile.TemporaryDirectory(prefix="gf_dl_") as tmp:
        path = Path(tmp) / "download.bin"
        filename = download_url_to_file(
            url,
            path,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            progress=progress,
            user_agent=user_agent,
        )
        return path.read_bytes(), filename


def _import_tar_path(
    tar_path: Path,
    upload_dir: Path,
    study_name: str,
    *,
    mode: str = "r:*",
) -> tuple[Path, str]:
    """Extract .tar / .tar.gz on disk, re-pack as zip, import without holding tar in RAM."""
    with tempfile.TemporaryDirectory(prefix="gf_tar_") as tmp:
        tmp_path = Path(tmp)
        extract_root = tmp_path / "extracted"
        extract_root.mkdir()
        with tarfile.open(tar_path, mode) as tf:
            try:
                tf.extractall(extract_root, filter=tarfile.data_filter)  # type: ignore[arg-type]
            except (AttributeError, TypeError):
                tf.extractall(extract_root)

        buf_path = tmp_path / "repack.zip"
        import zipfile

        with zipfile.ZipFile(buf_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in extract_root.rglob("*"):
                if fp.is_file():
                    zf.write(fp, arcname=str(fp.relative_to(extract_root)))
        return import_study_zip_path(buf_path, upload_dir, study_name)


def import_study_from_url(
    url: str,
    upload_dir: Path,
    study_name: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    progress: _ProgressCb | None = None,
) -> tuple[Path, str, str]:
    """
    Download a remote archive (streamed to disk) and import it as a study.

    Returns (tokenize_dir, summary_markdown, filename_hint).
    """
    if not normalize_study_name(study_name):
        raise RemoteDataError(
            "Enter an **experiment name** in Study name before downloading."
        )
    with tempfile.TemporaryDirectory(prefix="gf_url_") as tmp:
        archive_path = Path(tmp) / "archive.bin"
        filename = download_url_to_file(
            url,
            archive_path,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
            progress=progress,
        )
        # Prefer extension from filename hint for naming checks.
        named = Path(tmp) / filename
        if named.name != archive_path.name:
            archive_path.rename(named)
            archive_path = named

        if _looks_like_zip_file(archive_path, filename):
            tokenize_dir, summary = import_study_zip_path(
                archive_path, upload_dir, study_name
            )
            return tokenize_dir, summary, filename
        if _looks_like_tar_gz_file(archive_path, filename):
            tokenize_dir, summary = _import_tar_path(
                archive_path, upload_dir, study_name, mode="r:gz"
            )
            return tokenize_dir, summary, filename
        if _looks_like_tar_file(archive_path, filename):
            tokenize_dir, summary = _import_tar_path(
                archive_path, upload_dir, study_name, mode="r:"
            )
            return tokenize_dir, summary, filename
        raise RemoteDataError(
            f"Unsupported archive `{filename}`. "
            "Provide a direct link to a **.zip** (or .tar / .tar.gz) of 10x sample folders."
        )


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TIMEOUT_S",
    "RemoteDataError",
    "download_url_bytes",
    "download_url_to_file",
    "geo_series_ftp_prefix",
    "import_study_from_url",
    "pick_geo_supplementary_file",
    "resolve_cellxgene_asset_url",
    "resolve_download_url",
    "resolve_geo_supplementary_url",
    "rewrite_share_url",
    "validate_data_url",
]
