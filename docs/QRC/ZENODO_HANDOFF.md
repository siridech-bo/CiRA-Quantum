# Zenodo archive: handoff

The QRC raw-trace archive is ready to build. Two things are left for the person or agent doing the upload:

1. Add the author list.
2. Run the upload, which creates **draft** depositions only.

Publishing is a manual click on zenodo.org after review. Nothing in the tooling publishes.

## What is ready

| item | where | state |
|---|---|---|
| Packaging tool | `backend/scripts/qrc_zenodo_package.py` | plan / build / upload (drafts only) / regenerate features |
| Checksum manifest of every trace | `docs/QRC/data/TRACES_MANIFEST.{json,md}` | regenerated; includes Step 5 + Step 6 runs |
| Record plan | `docs/QRC/data/ZENODO_PLAN.{json,md}` | records A (ERA5 benchmark + Confirmations 1-2), C (Steps 5-6: Confirmations 3-4, drift test, encoding / readout-channel controls), B (everything else) |
| Deposition metadata | `docs/QRC/data/ZENODO_METADATA.json` | complete **except `creators`** (version 1.0, CC-BY-4.0, GitHub link filled) |
| Data package (every reported number) | `docs/QRC/data/` + `MANIFEST.md` | in git; archive the code and data via a GitHub release, see step 5 |

Feature matrices that are a pure function of a saved trace (`X = |FFT(fids_H)|[:, bins]`) are left out.
Their bin selections are kept as `<stem>.bins.npy`, and `--regen-features` rebuilds them.

## Steps

Run all of these on the host that has `backend/artifacts/`, DESKTOP-1A0J7FD. The traces are gitignored.

1. **Authors.** In `docs/QRC/data/ZENODO_METADATA.json`, replace the `creators` entry with the real list:

   ```json
   {"name": "Family, Given", "affiliation": "...", "orcid": "0000-..."}
   ```

   The upload refuses to run while any `TODO` remains in this file.
2. **Refresh the manifest and the plan** if any run was added after the last refresh:

   ```
   cd backend
   python scripts/qrc_trace_manifest.py
   python scripts/qrc_zenodo_package.py
   ```

   This hashes about 60 GB and takes about 15 minutes.
3. **Build** onto a disk with room for about 60 GB:

   ```
   python scripts/qrc_zenodo_package.py --build D:/zenodo_out --part-gb 10
   ```

   Every file is re-hashed against the manifest as it is written. Add `--dry-run` first to see the sizes.
4. **Upload drafts:**

   ```
   set ZENODO_TOKEN=...
   python scripts/qrc_zenodo_package.py --upload D:/zenodo_out
   ```

   - Use `--sandbox` to try sandbox.zenodo.org first.
   - The token is read from the environment only. Never write it into a file.
5. **Review each draft on zenodo.org, then publish by hand.** Afterwards:
   - Put the DOIs into the manuscript's acknowledgments (`docs/QRC/QRC_Manuscript.tex`, which currently says "internal repository").
   - Optionally link the GitHub release of branch `qrc-simulation` through Zenodo's GitHub integration, so the code and `docs/QRC/data/` get their own DOI.

Do not copy credentials from `D:\CiRA tools\cira-serverless\docs\cira-remote-access.md` into the repo or the archive.
