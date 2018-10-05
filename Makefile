ROOT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# 初期操作として data folder のzipをunzipする必要がある。
unzip-data:
	unzip data/ml-20m.zip -d data/

docker-build:
	docker build -t hoge:latest .

docker-run-A01:
	docker run -d --rm -v $(ROOT_DIR):/docker-work \
		hoge:latest \
		python3 run_script/A01_get_pickle_for_validation.py

docker-run-A02:
	docker run -d --rm -v $(ROOT_DIR):/docker-work \
		hoge:latest \
		python3 run_script/A02_aggregate_peformances_from_validation_result_pickle.py
