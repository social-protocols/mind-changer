# https://github.com/casey/just

# List available recipes in the order in which they appear in this file
_default:
  @just --list --unsorted

dev:
  cargo watch -x run

fix:
  cargo clippy --fix --allow-dirty

download-dataset:
  scripts/download-dataset
