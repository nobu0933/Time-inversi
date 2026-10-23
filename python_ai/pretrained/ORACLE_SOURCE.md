# 6x6-reversi-oracle coefficient source

`oracle-weight.rs` is the coefficient table from
[`mame/6x6-reversi-oracle`](https://github.com/mame/6x6-reversi-oracle),
`core/data/weight.rs`, commit `3c1469173ed47a4dd604a872e5ffe76532071988`.
SHA-256: `6fe8fb524772986986a0c9ddd8275664ecf4cc81a8915ea7b9bb00bb703f3f6a`.

The same commit's `core/data/opening-tree.dat` and
`core/data/opening-data.dat` are included as `oracle-opening-tree.dat` and
`oracle-opening-data.dat`. Their SHA-256 hashes are respectively
`ddd6a49aa51913050181890f42344afdd863ca636727efb5f740ce73c68d5205`
and `05700af9046fa97f7e5d0621ca6c6db2f90c9f515b6704e3e2462077ff786a03`.

To retrieve the coefficient file again from upstream when permitted:

```powershell
python -m time_inversi.oracle_teacher --download pretrained/oracle-weight-new.rs
```

The Python evaluator in `time_inversi/oracle_teacher.py` implements the four
edge-pattern features, legal mobility, and potential mobility for any 6×6
board. `time_inversi/oracle_opening.py` reads the opening tree for ordinary
game prefixes only. The normal-game oracle's native search engine is not
imported; the Python exact solver and bounded search provide the fallback.
