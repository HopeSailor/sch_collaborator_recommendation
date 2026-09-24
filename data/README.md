# Data

Place source files in `data/raw/` and run:

```bash
python data/build_sch.py --config configs/sch.yaml
```

The processed graph is written to `data/processed/sch.json`. Source and
processed data are excluded from Git.
