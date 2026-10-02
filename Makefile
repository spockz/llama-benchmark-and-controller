# This Makefile builds, tests, and maintains the llama.cpp benchmark harness executable.
SHELL := /bin/sh

GO ?= go
GOLANGCI_LINT ?= golangci-lint
BIN_DIR ?= bin
BINARY := $(BIN_DIR)/llama-bench-harness
GO_STATE := .scratch/go
export GOCACHE := $(CURDIR)/$(GO_STATE)/build
export GOMODCACHE := $(CURDIR)/$(GO_STATE)/mod
export GOTMPDIR := $(CURDIR)/$(GO_STATE)/tmp
export GOPATH := $(CURDIR)/$(GO_STATE)

.PHONY: all go-env build test vet lint fmt clean help

all: build

go-env:
	@mkdir -p "$(GOCACHE)" "$(GOMODCACHE)" "$(GOTMPDIR)"

build: go-env ## Build the llama.cpp benchmark harness
	@mkdir -p "$(BIN_DIR)"
	$(GO) build -buildvcs=false -o "$(BINARY)" .

test: go-env ## Run Go tests
	$(GO) test ./...

vet: go-env ## Run go vet
	$(GO) vet ./...

lint: go-env ## Run golangci-lint with the repository configuration
	$(GOLANGCI_LINT) run --config .golangci.yaml ./...

fmt: go-env ## Format Go source files
	$(GO) fmt ./...

clean: ## Remove generated binaries and Go test cache
	rm -rf "$(BIN_DIR)"
	$(GO) clean -testcache

help: ## Show available targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z_-]+:.*## / { printf "%-12s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
