.PHONY: data api-local api-gcs install-data install-api

install-data:
	python -m pip install -r data_gen/requirements.txt

install-api:
	python -m pip install -r api/requirements.txt

data: install-data
	cd data_gen && python build_assets.py --seed 10

api-local:
	bash scripts/run_local.sh

api-gcs:
	bash scripts/run_gcs.sh
