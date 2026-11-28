# Releasing Strataq

The Terminal, the engines and the signed update feed are published as the files of a GitHub release of
`qwertie/Strataq`. The Terminal, setup and `strata update` read
`https://github.com/qwertie/Strataq/releases/latest/download/` (`DEFAULT_FEED`, `PREBUILT_URL`, `install.sh` /
`install.ps1`). `STRATA_FEED` (installers) and `--feed` /
`STRATA_UPDATE_FEED` (`strata update`) override the address. Nothing in this tree uploads anything.

## Keys

The feed is signed with the fork's key (key id `f8aabae844c8ba19`; the secret `feed-key.json` is in
`C:\Users\qwert\strata-keys`, outside git, and must be backed up). `strata_common/update_keys.json` trusts that key and
upstream's `5d1bc23abbeffd0d`, so the fork's feed may list upstream's signed packages it does not rebuild under their
absolute upstream URLs. All signing happens on the release PC, on the finished files; build boxes have no key.

Only release keys live in `update_keys.json`. A tester or local-test key is passed with `STRATA_UPDATE_TRUSTED_KEYS` (a
file of extra public keys; `packaging/make_local_feed.py` writes one as `test-trust.json`). A build whose
`update_keys.json` is empty refuses every feed with "this build trusts no update key".

## Versions

* The engine packages carry CMakeLists.txt's version (0.1.59.1 at the first release), which must be newer than
  upstream's (0.1.59), or `strata update` offers nothing to a PC that has upstream's engine.
* Control has its own `VERSION` file (0.1.150); the Terminal reports 0.2.50. The Control package's version must be newer
  than upstream's for the same reason.
* `--sequence` must be above every sequence a PC may have accepted from upstream's feed (124 on 2026-11-09): the fork's
  first feed uses 1000 and counts up. A PC that accepted the fork's feed refuses upstream's afterwards (a lower
  sequence).

## Packages

Qwen packages carry `strata-q38` and `strata-q36`; DeepSeek packages carry `strata-ds41`. Setup selects the program for
the chosen model family.

* Windows AMD: `tools\hip\build_release.py` per variant (`STRATA_HIP_ARCHS`: rdna3 = gfx1100;gfx1101;gfx1102;gfx1151,
  rx9070 = gfx1200;gfx1201, rdna2 = gfx1030;gfx1031), then `package_engine.py --variant <name>`; the rdna3 build also
  yields the Windows AMD DeepSeek package.
* Linux: `packaging/build_hip_linux.sh` and `build_cuda_linux.sh` in an Ubuntu 22.04 environment (the packages must run
  on 22.04's glibc), `build_dsv41_*.sh` for DeepSeek.
* The Terminal archives (`package_terminal.py`, one per platform) and `install-manifest.txt` go in the same release.

## Steps

    python packaging/feed_tool.py sign-package <engine package folder> --key C:\Users\qwert\strata-keys\feed-key.json   # every engine package
    python packaging/feed_tool.py pack-engine  <engine package folder> --out RELEASE/engine --url-prefix=
    python packaging/feed_tool.py pack-control <control folder> --version 0.1.150 --platform P --out RELEASE/control --url-prefix=   # --format zip for windows-x86_64
    python packaging/feed_tool.py build-feed --out RELEASE --sequence 1000 --key C:\Users\qwert\strata-keys\feed-key.json --engine RELEASE/engine --control RELEASE/control [--dsv41 RELEASE/dsv41] [--shapes RELEASE/shapes]
    python packaging/feed_tool.py verify-feed RELEASE --pub C:\Users\qwert\strata-keys\feed-key.pub.json

Use `--url-prefix=`, not `--url-prefix ""`: PowerShell 5.1 drops an empty argument. GitHub assets are flat, so every
file sits at the release folder's top. The Control version is the `VERSION` file, not the Terminal version: a Control
package newer than the installed one is installed by `strata update`. `feed_tool.py` sets the execute bit of Linux
packages from the file header, so they can be packed on Windows.

`build-feed` also writes `RELEASE/install-manifest.txt`, the file the one-line installers read: one line per Control
package, `platform version url sha256 size`, taken from the `*.entry.json` files in `RELEASE/control`. The Control
package is the Terminal that `install.sh` unpacks (it must hold `bin/strata`). `install.sh` unpacks with `tar -xzf` and
`install.ps1` with `Expand-Archive`, so `windows-x86_64` must be packed with `--format zip` and `linux-x86_64` as
`tar.gz`; `build-feed` leaves a pack in the wrong format out of the manifest and says so. A platform with no line gets
"this release has no Terminal for <platform>".

Copy `install.sh` and `install.ps1` from `packaging/` into RELEASE unchanged (their built-in address is the release
folder), then upload the files as flat assets (PowerShell):

    $assets = Get-ChildItem RELEASE -Recurse -File | Where-Object { $_.Name -notlike '*.entry.json' }
    gh release create v0.1.59.1-strataq --repo qwertie/Strataq --target main `
      --title "Strataq 0.1.59.1" --notes "..." @($assets.FullName)

Publish a normal release: `releases/latest` must point at it, and GitHub picks the newest release that is not a
pre-release, so the installers and `strata update` would get a 404 from a pre-release. The `latest` release must carry
every file listed above, so a later hotfix release repeats them all.
