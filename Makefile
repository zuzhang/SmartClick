.DEFAULT_GOAL := help

PYTHON ?= python3
APP_CONFIG ?= config/config.yaml
DATASET_CONFIG ?= training/game-controls.yaml
CVAT_SOURCE ?=
OVERWRITE ?= 0

.PHONY: help calibrate run test import-cvat

help:
	@printf '%s\n' \
		'SmartClick commands:' \
		'  make calibrate' \
		'  make run' \
		'  make test' \
		'  make import-cvat CVAT_SOURCE=/path/to/cvat-export.zip'
	@printf '%s\n' \
		'Override APP_CONFIG to select the app config.' \
		'Set OVERWRITE=1 with import-cvat to replace files with matching names.'

calibrate:
	$(PYTHON) calibrate_region.py --config "$(APP_CONFIG)"

run:
	$(PYTHON) app.py --config "$(APP_CONFIG)"

test:
	$(PYTHON) -m unittest discover -s tests -v

import-cvat:
	@test -n "$(CVAT_SOURCE)" || { \
		echo "Set CVAT_SOURCE to a CVAT ZIP archive or extracted folder." >&2; \
		exit 2; \
	}
	$(PYTHON) training/import_cvat.py "$(CVAT_SOURCE)" --config "$(DATASET_CONFIG)" $(if $(filter 1 true yes,$(OVERWRITE)),--overwrite,)