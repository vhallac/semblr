#!/bin/sh
# Pick the first `biome` on PATH that can actually run, then delegate to it.
#
# Why this exists: `@biomejs/biome` (devDependency) provides the pinned binary
# for CI and generic Linux hosts. On NixOS that binary is dynamically linked and
# cannot execute, so an environment-provided biome (shell.nix / user profile)
# must be used instead. npm prepends node_modules/.bin to PATH when running
# package scripts, so plain `biome` resolution would pick the broken wrapper on
# NixOS. We probe candidates with `--version` and skip the ones that cannot run.
set -eu

old_ifs=$IFS
IFS=:
for dir in $PATH; do
	[ -n "$dir" ] || dir=.
	candidate=$dir/biome
	if [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
		IFS=$old_ifs
		exec "$candidate" "$@"
	fi
done
IFS=$old_ifs

echo "run-biome: no runnable 'biome' found on PATH: $PATH" >&2
exit 127
