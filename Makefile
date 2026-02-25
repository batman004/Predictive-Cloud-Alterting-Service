.PHONY: train evaluate predict docker-build docker-train docker-predict clean

ARTIFACTS ?= artifacts

# ── Local ────────────────────────────────────────────────────────────────────

train:
	python cli.py train --artifacts $(ARTIFACTS)

evaluate:
	python cli.py evaluate --artifacts $(ARTIFACTS)

predict:
	@echo "Usage: make predict SOURCE=ec2_cpu_utilization_fe7f93"
	python cli.py predict --source $(SOURCE) --artifacts $(ARTIFACTS)

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t cloud-alerting .

docker-train:
	docker run --rm -v $$(pwd)/$(ARTIFACTS):/app/$(ARTIFACTS) cloud-alerting train

docker-predict:
	@echo "Usage: make docker-predict SOURCE=ec2_cpu_utilization_fe7f93"
	docker run --rm -v $$(pwd)/$(ARTIFACTS):/app/$(ARTIFACTS) cloud-alerting predict --source $(SOURCE)

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	rm -rf $(ARTIFACTS)
