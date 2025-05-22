# AI_agent/Makefile

# Имя образа
IMAGE_NAME = ai_agent

.DEFAULT_GOAL := help
CURDIR := $(CURDIR) # Текущая папка, чтобы монтировать её в контейнер

# 📌 1. Справка по командам
help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Проверка наличия Docker
check-docker:
	@command -v docker > /dev/null 2>&1 || { \
		echo "Docker не установлен. Пожалуйста, установите Docker. Подробнее: https://docs.docker.com/get-docker/"; \
		exit 1; \
	}

build: check-docker ## Сборка образа
	docker build -t $(IMAGE_NAME) .

clean: check-docker ## Удалить локальный образ
	@if docker inspect --type=image $(IMAGE_NAME) > /dev/null 2>&1; then \
		docker rmi -f $(IMAGE_NAME); \
	else \
		echo "Образ $(IMAGE_NAME) не найден."; \
	fi

review: ## Код-ревью. Пример: make review FILES="путь/к/файлу1.py путь/к/файлу2.js"
ifndef FILES
	$(error Укажите FILES="file1.py file2.js")
endif
	@$(eval ABS_FILES := $(abspath $(FILES)))
	@echo "📂 Анализ следующих файлов:"
	@for file in $(ABS_FILES); do echo $$file; done
	docker run --rm \
		$(foreach f,$(ABS_FILES), -v $f:$f:ro) \
		-w /app \
		$(IMAGE_NAME) \
		python agent.py $(ABS_FILES)