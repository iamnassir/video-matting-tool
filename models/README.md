# Model weights

This folder holds downloaded weights, one sub-folder per model id. Nothing here is bundled with the
application and nothing is downloaded automatically:

    python main.py models list
    python main.py models download rvm
    python main.py models verify

Every file is fetched from its authors' official location and checked against a pinned size and
SHA-256. Licences differ per model (MatAnyone is non-commercial): see docs/MODELS.md.
