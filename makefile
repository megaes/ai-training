# Check to see if we can use ash, in Alpine images, or default to BASH.
SHELL_PATH = /bin/ash
SHELL = $(if $(wildcard $(SHELL_PATH)),/bin/ash,/bin/bash)

# ==============================================================================
# Remove Ollama Auto-Run
#
# We have discovered that Ollama is installing itself to run at login on all OS. 
# MacOS
# To remove this on the Mac go to `Settings/General/Login Items & Extensions`
# and remove Ollama as a startup item. Then navigate to `~/Library/LaunchAgents`
# and remove the Ollama file you will find.
#
# Linux
# sudo systemctl stop ollama.service
# sudo systemctl disable ollama.service
#

# ==============================================================================
# Mongo support
#
# db.book.find({id: 300})

# ==============================================================================
# Examples

example1:
	go run cmd/examples/example1/main.go

example2:
	go run cmd/examples/example2/main.go

example3:
	go run -exec "env DYLD_LIBRARY_PATH=$$GOPATH/src/github.com/ardanlabs/ai-training/foundation/word2vec/libw2v/lib" cmd/examples/example3/main.go

example4:
	go run cmd/examples/example4/main.go

example5:
	go run cmd/examples/example5/main.go

example6:
	go run cmd/examples/example6/main.go

example7:
	go run cmd/examples/example7/main.go

example8:
	go run cmd/examples/example8/main.go

example9:
	go run cmd/examples/example9/main.go

example10-step1:
	go run cmd/examples/example10/step1/main.go

example10-step2:
	go run cmd/examples/example10/step2/main.go

example10-step3:
	go run cmd/examples/example10/step3/main.go

example10-step4:
	go run cmd/examples/example10/step4/main.go

# ==============================================================================
# Install dependencies

install:
	brew install mongosh
	brew install ollama
	brew install mplayer
	brew install pgcli
	brew install uv

docker:
	docker pull mongodb/mongodb-atlas-local
	docker pull ghcr.io/open-webui/open-webui:v0.6.18
	docker pull postgres:17.5

ollama-pull:
	ollama pull mxbai-embed-large
	ollama pull llama3.2
	ollama pull llama3.2-vision
	ollama pull gpt-oss:latest

python-install:
	rm -rf .venv
	uv venv --python 3.12 && uv lock && uv sync

# ==============================================================================
# Manage project

compose-up:
	docker compose -f zarf/docker/compose.yaml up

compose-down:
	docker compose -f zarf/docker/compose.yaml down

compose-logs:
	docker compose logs -n 100

ollama-up:
	export OLLAMA_MODELS="zarf/docker/ollama/models" && \
	ollama serve

ollama-logs:
	tail -f -n 100 ~/.ollama/logs/server.log

# ==============================================================================
# Run Tooling

download-data:
	curl -o zarf/data/example3.gz -X GET https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz \
	&& gunzip -k -d zarf/data/example3.gz \
	&& mv zarf/data/example3 zarf/data/example3.json

clean-data:
	go run cmd/cleaner/main.go

mongo:
	mongosh -u ardan -p ardan mongodb://localhost:27017

pgcli:
	pgcli postgresql://postgres:postgres@localhost

openwebui:
	open -a "Google Chrome" http://localhost:3000/

# ==============================================================================
# VLLM
# You need to add this to your .env file
# 	export VLLM_CPU_KVCACHE_SPACE=26

vllm-install:
	uv pip install vllm

vllm-update:
	uv pip install --upgrade vllm

vllm-run:
	source .env && uv run vllm serve --host 0.0.0.0 --port 8000 "NousResearch/Hermes-3-Llama-3.1-8B"

vllm-compose-up:
	docker compose -f zarf/docker/compose-owu-vllm.yaml up

vllm-compose-down:
	docker compose -f zarf/docker/compose-owi-vllm.yaml down

vllm-test:
	curl -X POST "http://localhost:8000/v1/chat/completions" \
		-H "Content-Type: application/json" \
		--data '{ \
			"model": "NousResearch/Hermes-3-Llama-3.1-8B", \
			"messages": [ \
				{"role": "system", "content": [{"type": "text", "text": "You are an expert developer and you are helping the user with their question."}]}, \
				{"role": "user", "content": [{"type": "text", "text": "How do you declare a variable in Python?"}]} \
			] \
		}'

# ==============================================================================
# Go Modules support

tidy:
	go mod tidy
	go mod vendor

deps-upgrade:
	go get -u -v ./...
	go mod tidy
	go mod vendor

# ==============================================================================
# Python Dependencies

deps-python-sync:
	uv sync

deps-python-upgrade:
	uv lock --upgrade && uv sync

deps-python-outdated:
	uv pip list --outdated

# ==============================================================================

curl-tooling:
	curl http://localhost:11434/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{ \
	"model": "qwen3:8b", \
	"messages": [ \
		{ \
			"role": "user", \
			"content": "What is the weather like in New York, NY?" \
		} \
	], \
	"stream": false, \
	"tools": [ \
		{ \
			"type": "function", \
			"function": { \
				"name": "get_current_weather", \
				"description": "Get the current weather for a location", \
				"parameters": { \
					"type": "object", \
					"properties": { \
						"location": { \
							"type": "string", \
							"description": "The location to get the weather for, e.g. San Francisco, CA" \
						} \
					}, \
					"required": ["location"] \
				} \
			} \
		} \
  	], \
	"tool_selection": "auto", \
	"options": { "num_ctx": 32000 } \
	}'

curl-tooling-stream:
	curl http://localhost:11434/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{ \
	"model": "qwen3:8b", \
	"messages": [ \
		{ \
			"role": "user", \
			"content": "How are you?" \
		} \
	], \
	"stream": true \
	}'
