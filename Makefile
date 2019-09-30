include .env
export

remote-train:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "superresolution train" \
		--projectId "pr3sinqyo" \
		--machineType "P5000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--command "make train"

train:
	pip3 install -r requirements.txt
	python3 src/train.py \
		--train-dataset=/storage/datasets/imagenet/train/ \
		--test-dataset=/storage/datasets/imagenet/val/ \
		--validation-path=/storage/datasets/colorization-val/ \
		--batch-size=32 \
		--epochs=1 \
		--input-w=128 \
		--input-h=128 \
		--scale=2 \
		--model-save-path=/artifacts/

test:
	PYTHONPATH=src/ python -m pytest tests/

lint:
	python -m pycodestyle src/
