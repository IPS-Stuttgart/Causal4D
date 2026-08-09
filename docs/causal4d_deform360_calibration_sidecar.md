# Deform360 calibration sidecar

The official Deform360 `001-rope` release stores refined camera calibration in
three NumPy object-dictionary files:

- `calibration_refined/intrinsics.npy`;
- `calibration_refined/extrinsics.npy`;
- `calibration_refined/dist.npy`.

Opening these files requires pickle deserialization. Normal Causal4D preflight no
longer opens them. A one-time converter is the only pickle-enabled path, and it
requires an exact acknowledgement that the inputs are the pinned official
Deform360 dataset revision.

## One-time conversion

Run the converter before the first preflight:

```bash
causal4d public deform360 convert-calibration \
  /mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw/001-rope \
  --trust-acknowledgement \
  TRUST_PINNED_OFFICIAL_DEFORM360_001_ROPE_PICKLE
```

By default, this writes the safe archive and manifest to a `.causal4d` sidecar
directory next to the raw `001-rope` directory. The raw dataset remains unchanged,
so the locked 908-file inventory is preserved. Use `--output-manifest` and
`--output-archive` only when both outputs should live in another common directory.
Existing outputs are refused unless `--overwrite` is given.

Do not run the converter on downloaded files from another source, a mutable cache,
or arbitrary NumPy object arrays. The acknowledgement authorizes only the pinned
`brownu/deform360` revision
`7fea8e20231a47641d1d2bc8791920ec4e62ec5e`, object `001-rope`.

## Safe preflight

Normal preflight automatically looks for the default sidecar:

```bash
causal4d public deform360 preflight \
  /mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw/001-rope \
  results/causal4d_public/deform360_001_rope_preflight.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --hash-media
```

For a nondefault location, pass `--calibration-manifest PATH`.

The manifest binds:

- the pinned repository, dataset revision, and object identity;
- the exact byte count and SHA-256 of all three legacy source files;
- the exact byte count and SHA-256 of a non-pickled NPZ archive;
- the closed NPZ member inventory;
- camera names, array shapes, and dtypes; and
- a canonical manifest checksum.

Every preflight re-reads and hashes the original calibration files without
unpickling them, validates the sidecar manifest with duplicate-key rejection,
loads the NPZ with `allow_pickle=False`, and checks finite numeric values and the
closed shape contract. A missing sidecar, source change, archive change, symlink,
unknown manifest field, or camera mismatch fails the calibration gate closed.

## Boundary

The conversion changes neither calibration values nor the frozen estimator. It
only changes their transport representation. The sidecar is derived evidence and
must not be counted as an additional raw dataset file, a physical execution, or a
new scientific result.
