# https://github.com/casey/just

# List available recipes in the order in which they appear in this file
_default:
  @just --list --unsorted

dev:
  cargo watch -x run

download-dataset:
  scripts/download-dataset
