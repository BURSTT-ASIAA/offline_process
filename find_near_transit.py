#!/usr/bin/env python3
"""List intensity files whose filename timestamp is near a transit time."""

import argparse
import glob
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{8}_\d{6}|\d{6}_\d{6})(?!\d)")
LOCAL_TZ = timezone(timedelta(hours=8))


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp or a filename-style UTC timestamp."""
    value = value.strip()
    for fmt in ("%Y%m%d_%H%M%S",):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    if re.fullmatch(r"\d{6}_\d{6}", value):
        try:
            return datetime.strptime("20" + value, "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    iso_value = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use YYYYMMDD_HHMMSS, YYMMDD_HHMMSS, or an ISO-8601 timestamp"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_from_name(path: Path) -> datetime | None:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    value = match.group(1)
    try:
        if len(value) == 15:
            local_time = datetime.strptime(value, "%Y%m%d_%H%M%S")
        else:
            local_time = datetime.strptime("20" + value, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return local_time.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)


def planner_transit(target: str, date: str, planner: Path, site: str) -> datetime:
    """Get the target transit occurring on the requested local date."""
    date = date.replace("-", "")
    if not re.fullmatch(r"(?:\d{6}|\d{8})", date):
        raise ValueError("local date must be YYMMDD or YYYYMMDD")

    local_date = datetime.strptime(
        date, "%Y%m%d" if len(date) == 8 else "%y%m%d"
    ).date()
    for utc_date in (local_date - timedelta(days=1), local_date):
        planner_date = utc_date.strftime("%y%m%d")
        result = subprocess.run(
            [
                "python",
                str(planner),
                target,
                "-d",
                planner_date,
                "--utc",
                "--site",
                site,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            fields = line.split()
            if fields and fields[0].lower() == target.lower() and len(fields) >= 5:
                transit = datetime.strptime(
                    f"{planner_date}_{fields[4]}", "%y%m%d_%H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                if transit.astimezone(LOCAL_TZ).date() == local_date:
                    return transit
    raise ValueError(f"planner returned no transit row for {target!r}")


def intensity_files(root: Path, start: datetime, end: datetime):
    """Yield files matching local dates touched by the UTC search window."""
    local_start = start.astimezone(LOCAL_TZ).date()
    local_end = end.astimezone(LOCAL_TZ).date()
    current_date = local_start
    while current_date <= local_end:
        date_token = current_date.strftime("%y%m%d")
        pattern = root / "burstt1?" / "disk*" / "intensity" / f"*_{date_token}_*"
        for filename in glob.iglob(str(pattern)):
            yield Path(filename)
        current_date += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find /burstt1?/disk*/intensity files near a transit time."
    )
    parser.add_argument(
        "transit",
        nargs="?",
        type=parse_timestamp,
        help="UTC transit time: YYYYMMDD_HHMMSS, YYMMDD_HHMMSS, or ISO-8601",
    )
    parser.add_argument("--target", help="source name to pass to obs_planner.py")
    parser.add_argument("--date", help="local date (UTC+8): YYMMDD or YYYYMMDD")
    parser.add_argument(
        "--planner",
        type=Path,
        default=Path("/data/kylin/bin/obs_planner.py"),
        help="path to obs_planner.py",
    )
    parser.add_argument("--site", default="fushan6", help="planner observatory site")
    parser.add_argument(
        "--hours",
        type=float,
        default=1.5,
        help="half-width of the UTC search window in hours (default: 1.5)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/"),
        help="filesystem root containing burstt1?, burstt2?, ... (default: /)",
    )
    args = parser.parse_args()

    if args.transit is None:
        if not args.target or not args.date:
            parser.error("provide TRANSIT, or both --target and --date")
        try:
            args.transit = planner_transit(args.target, args.date, args.planner, args.site)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            parser.error(str(exc))
    elif args.target or args.date:
        parser.error("--target and --date require omitting TRANSIT")

    if args.hours < 0:
        parser.error("--hours must be non-negative")

    start = args.transit - timedelta(hours=args.hours)
    end = args.transit + timedelta(hours=args.hours)
    matches = []

    for path in intensity_files(args.root, start, end):
        timestamp = timestamp_from_name(path)
        if timestamp is None or not (start <= timestamp <= end):
            continue
        matches.append((abs(timestamp - args.transit), timestamp, path))

    for _, timestamp, path in sorted(matches):
        offset = (timestamp - args.transit).total_seconds() / 3600
        print(f"{offset:+.3f} h\t{timestamp:%Y-%m-%dT%H:%M:%SZ}\t{path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
