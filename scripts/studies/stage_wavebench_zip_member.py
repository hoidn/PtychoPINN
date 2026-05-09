from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

import requests


EOCD_SIGNATURE = 0x06054B50
ZIP64_EOCD_SIGNATURE = 0x06064B50
ZIP64_LOCATOR_SIGNATURE = 0x07064B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
ZIP64_EXTRA_FIELD_ID = 0x0001


def _get_with_range(session: requests.Session, url: str, start: int, end: int | None) -> requests.Response:
    headers = {"Range": f"bytes={start}-{'' if end is None else end}"}
    response = session.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    return response


def _fetch_exact_bytes(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = _get_with_range(session, url, start, end)
    data = response.content
    expected = end - start + 1
    if len(data) != expected:
        raise ValueError(
            f"expected {expected} bytes for range {start}-{end}, got {len(data)}"
        )
    return data


def _parse_zip64_extra(extra: bytes, needs_uncompressed: bool, needs_compressed: bool, needs_offset: bool) -> tuple[int | None, int | None, int | None]:
    index = 0
    uncompressed = None
    compressed = None
    offset = None
    while index + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, index)
        index += 4
        field_data = extra[index:index + field_size]
        index += field_size
        if field_id != ZIP64_EXTRA_FIELD_ID:
            continue
        cursor = 0
        if needs_uncompressed:
            uncompressed = struct.unpack_from("<Q", field_data, cursor)[0]
            cursor += 8
        if needs_compressed:
            compressed = struct.unpack_from("<Q", field_data, cursor)[0]
            cursor += 8
        if needs_offset:
            offset = struct.unpack_from("<Q", field_data, cursor)[0]
        break
    return uncompressed, compressed, offset


def _locate_eocd(session: requests.Session, url: str, archive_size: int) -> tuple[int, int]:
    window = min(max(archive_size, 65536), 4 * 1024 * 1024)
    start = archive_size - window
    tail = _fetch_exact_bytes(session, url, start, archive_size - 1)

    eocd_offset = tail.rfind(struct.pack("<I", EOCD_SIGNATURE))
    if eocd_offset == -1:
        raise ValueError("could not locate end-of-central-directory record")
    eocd = tail[eocd_offset:eocd_offset + 22]
    (
        _sig,
        _disk_no,
        _disk_cd,
        _entries_disk,
        entries_total_32,
        cd_size_32,
        cd_offset_32,
        comment_len,
    ) = struct.unpack("<IHHHHIIH", eocd)
    if eocd_offset + 22 + comment_len > len(tail):
        raise ValueError("truncated EOCD comment")

    if cd_offset_32 != 0xFFFFFFFF and cd_size_32 != 0xFFFFFFFF:
        return cd_offset_32, cd_size_32

    locator_offset = tail.rfind(struct.pack("<I", ZIP64_LOCATOR_SIGNATURE), 0, eocd_offset)
    if locator_offset == -1:
        raise ValueError("zip64 archive missing locator")
    locator = tail[locator_offset:locator_offset + 20]
    _sig, _disk_with_zip64, zip64_eocd_offset, _num_disks = struct.unpack("<IIQI", locator)

    zip64_record = _fetch_exact_bytes(session, url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if struct.unpack_from("<I", zip64_record, 0)[0] != ZIP64_EOCD_SIGNATURE:
        raise ValueError("invalid zip64 end-of-central-directory signature")
    cd_size = struct.unpack_from("<Q", zip64_record, 40)[0]
    cd_offset = struct.unpack_from("<Q", zip64_record, 48)[0]
    return cd_offset, cd_size


def _find_member(session: requests.Session, url: str, member_name: str, archive_size: int) -> dict:
    cd_offset, cd_size = _locate_eocd(session, url, archive_size)
    central_directory = _fetch_exact_bytes(session, url, cd_offset, cd_offset + cd_size - 1)

    cursor = 0
    while cursor + 46 <= len(central_directory):
        if struct.unpack_from("<I", central_directory, cursor)[0] != CENTRAL_DIRECTORY_SIGNATURE:
            raise ValueError(f"invalid central directory entry at offset {cursor}")

        header = struct.unpack_from("<IHHHHHHIIIHHHHHII", central_directory, cursor)
        compressed_size = header[8]
        uncompressed_size = header[9]
        name_len = header[10]
        extra_len = header[11]
        comment_len = header[12]
        compression_method = header[4]
        local_header_offset = header[16]

        name_start = cursor + 46
        name_end = name_start + name_len
        extra_end = name_end + extra_len
        file_name = central_directory[name_start:name_end].decode("utf-8")
        extra = central_directory[name_end:extra_end]

        needs_uncompressed = uncompressed_size == 0xFFFFFFFF
        needs_compressed = compressed_size == 0xFFFFFFFF
        needs_offset = local_header_offset == 0xFFFFFFFF
        zip64_uncompressed, zip64_compressed, zip64_offset = _parse_zip64_extra(
            extra, needs_uncompressed, needs_compressed, needs_offset
        )

        if needs_uncompressed:
            uncompressed_size = zip64_uncompressed
        if needs_compressed:
            compressed_size = zip64_compressed
        if needs_offset:
            local_header_offset = zip64_offset

        if file_name == member_name:
            return {
                "compression_method": compression_method,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
            }

        cursor = extra_end + comment_len

    raise FileNotFoundError(member_name)


def _local_data_start(session: requests.Session, url: str, member: dict) -> int:
    header = _fetch_exact_bytes(
        session,
        url,
        member["local_header_offset"],
        member["local_header_offset"] + 29,
    )
    if struct.unpack_from("<I", header, 0)[0] != LOCAL_FILE_HEADER_SIGNATURE:
        raise ValueError("invalid local file header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    return member["local_header_offset"] + 30 + name_len + extra_len


def _stream_member(session: requests.Session, url: str, member: dict, output_path: Path) -> dict:
    data_start = _local_data_start(session, url, member)
    data_end = data_start + member["compressed_size"] - 1
    response = _get_with_range(session, url, data_start, data_end)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    bytes_written = 0
    compression_method = member["compression_method"]
    if compression_method == 8:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    elif compression_method == 0:
        decompressor = None
    else:
        raise ValueError(f"unsupported compression method: {compression_method}")

    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if decompressor is None:
                decoded = chunk
            else:
                decoded = decompressor.decompress(chunk)
            if decoded:
                handle.write(decoded)
                hasher.update(decoded)
                bytes_written += len(decoded)
        if decompressor is not None:
            tail = decompressor.flush()
            if tail:
                handle.write(tail)
                hasher.update(tail)
                bytes_written += len(tail)

    stat = output_path.stat()
    if bytes_written != member["uncompressed_size"] or stat.st_size != member["uncompressed_size"]:
        raise ValueError(
            f"staged size mismatch: wrote {bytes_written} bytes, expected {member['uncompressed_size']}"
        )

    return {
        "actual_path": str(output_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_epoch_s": int(stat.st_mtime),
        "sha256": hasher.hexdigest(),
        "data_range": {"start": data_start, "end": data_end},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a single member from a remote ZIP file.")
    parser.add_argument("--zip-url", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    args = parser.parse_args()

    session = requests.Session()
    head = session.head(args.zip_url, allow_redirects=True, timeout=120)
    head.raise_for_status()
    archive_size = int(head.headers["Content-Length"])

    member = _find_member(session, args.zip_url, args.member, archive_size)
    staged = _stream_member(session, args.zip_url, member, Path(args.output))
    payload = {
        "zip_url": args.zip_url,
        "member": args.member,
        "archive_size_bytes": archive_size,
        "compression_method": member["compression_method"],
        "compressed_size_bytes": member["compressed_size"],
        "uncompressed_size_bytes": member["uncompressed_size"],
        **staged,
    }
    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, indent=2) + os.linesep, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
