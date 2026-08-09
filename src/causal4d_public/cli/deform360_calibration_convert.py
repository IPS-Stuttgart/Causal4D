"""Convert pinned Deform360 calibration pickles into a safe sidecar archive."""

from __future__ import annotations

import argparse
import json

from causal4d_public.deform360_calibration import (
    DEFORM360_CALIBRATION_TRUST_ACKNOWLEDGEMENT,
    convert_legacy_deform360_calibration,
    default_deform360_calibration_manifest_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_object_dir")
    parser.add_argument(
        "--output-manifest",
        help=(
            "Safe manifest path. Defaults to the .causal4d sidecar directory next "
            "to the raw 001-rope directory."
        ),
    )
    parser.add_argument(
        "--output-archive",
        help="Safe NPZ path; it must be next to the manifest.",
    )
    parser.add_argument(
        "--trust-acknowledgement",
        required=True,
        metavar="TOKEN",
        help=(
            "Required exact token acknowledging that the one-time converter may "
            "deserialize only the pinned official dataset: "
            f"{DEFORM360_CALIBRATION_TRUST_ACKNOWLEDGEMENT}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing safe sidecar after revalidation.",
    )
    args = parser.parse_args()
    manifest_path = args.output_manifest or str(
        default_deform360_calibration_manifest_path(args.raw_object_dir)
    )
    try:
        summary = convert_legacy_deform360_calibration(
            args.raw_object_dir,
            output_manifest=manifest_path,
            output_archive=args.output_archive,
            trust_acknowledgement=args.trust_acknowledgement,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "output_manifest": manifest_path,
                "camera_count": summary["camera_count"],
                "manifest_file_sha256": summary["safe_manifest"]["sha256"],
                "manifest_content_sha256": summary["safe_manifest"][
                    "content_sha256"
                ],
                "archive_sha256": summary["safe_archive"]["sha256"],
                "legacy_pickle_loaded_during_validation": summary[
                    "legacy_pickle_loaded"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
