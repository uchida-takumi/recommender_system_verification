ROOT_DIR=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# 初期操作として data folder のzipをunzipする必要がある。
unzip-data:
	unzip data/ml-20m.zip -d data/

mv_pickle:
	mv pickle/* ../backup/

docker-build:
	docker build -t hoge:latest .

docker-run-A01:
	docker run --rm -d -v $(ROOT_DIR):/docker-work \
		hoge:latest \
		python3 -m src.A01_get_pickle_for_validation

docker-run-A02:
	docker run --rm -d -v $(ROOT_DIR):/docker-work \
		hoge:latest \
		python3 -m src.A02_aggregate_peformances_from_validation_result_pickle

docker-run-B10:
    docker run --rm -d -v $(ROOT_DIR):/docker-work \
        hoge:latest \
        python3 -m src.B10_topN_dispersion
        