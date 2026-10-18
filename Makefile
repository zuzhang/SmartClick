PYTHON ?= python3
APP_CONFIG ?= config/config.yaml

.PHONY: calibrate

calibrate:
	$(PYTHON) calibrate_region.py --config "$(APP_CONFIG)"
