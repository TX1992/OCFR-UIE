# Dataset Splits

Each dataset directory contains fixed `train.txt`, `val.txt`, and `test.txt` lists. Every
non-empty line has two tab-separated paths relative to the dataset root:

```text
path/to/degraded.png<TAB>path/to/reference.png
```

The loader rejects absolute paths, missing images, duplicate names, and overlap between splits.
Pass the local dataset location through `--data-root` or `DATA_ROOT`; do not edit the lists when
moving the repository to another machine.

| Split directory | Train | Validation | Test | Dataset root expected by the lists |
|---|---:|---:|---:|---|
| `UIEB` | 712 | 89 | 89 | UIEB paired dataset root |
| `LSUI` | 3,423 | 428 | 428 | LSUI root containing `input/` and `GT/` |
| `EUVP-Dark` | 4,440 | 555 | 555 | EUVP root containing `EUVP-Dark/` |
| `EUVP-Scene` | 1,748 | 218 | 219 | EUVP root containing `EUVP-Scene/` |
