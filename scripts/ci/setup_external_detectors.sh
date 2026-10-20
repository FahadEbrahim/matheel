#!/usr/bin/env bash
# Linux x86_64 provisioning for the explicit weekly/manual integration workflow.
set -euo pipefail

if [[ $# != 2 || "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Usage: bash setup_external_detectors.sh NEW_TOOLS_DIRECTORY ENV_FILE (Linux x86_64)" >&2
  exit 2
fi

tools_directory="$1"
environment_file="$2"
repository_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir "$tools_directory"
tools_directory="$(cd -- "$tools_directory" && pwd)"
mkdir "$tools_directory/downloads" "$tools_directory/java" "$tools_directory/node" "$tools_directory/dolos"

download_verified() {
  local url="$1" filename="$2" digest="$3"
  curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
    "$url" --output "$tools_directory/downloads/$filename"
  printf '%s  %s\n' "$digest" "$tools_directory/downloads/$filename" | sha256sum --check --strict
}

download_verified \
  'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz' \
  java.tar.gz dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e
download_verified \
  'https://nodejs.org/dist/v22.23.3/node-v22.23.3-linux-x64.tar.xz' \
  node.tar.xz df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de
download_verified \
  'https://github.com/jplag/JPlag/releases/download/v6.3.0/jplag-6.3.0-jar-with-dependencies.jar' \
  jplag.jar 5f2c21e8b88ed77134effcb3a5a3ab13d188f6a3e16d401387f7479e92db9aa2

tar --extract --gzip --file "$tools_directory/downloads/java.tar.gz" --directory "$tools_directory/java" --strip-components=1
tar --extract --xz --file "$tools_directory/downloads/node.tar.xz" --directory "$tools_directory/node" --strip-components=1
cp "$repository_directory/examples/external_detectors/package.json" "$repository_directory/examples/external_detectors/package-lock.json" "$tools_directory/dolos/"
printf '%s  %s\n' 29f32c0af8eb8e824a9d22b5fd1c0335a8545ba275c76920e675a59a24d804bc \
  "$tools_directory/dolos/package-lock.json" | sha256sum --check --strict

# Use the pinned Node distribution's npm and headers for native parser builds.
export PATH="$tools_directory/node/bin:$PATH"
export npm_config_nodedir="$tools_directory/node"
export npm_config_python
npm_config_python="$(command -v python)"
export npm_config_build_from_source=true
npm ci --prefix "$tools_directory/dolos" --no-audit --no-fund

{
  printf 'MATHEEL_JPLAG_JAR=%s\n' "$tools_directory/downloads/jplag.jar"
  printf 'MATHEEL_JAVA=%s\n' "$tools_directory/java/bin/java"
  printf 'MATHEEL_DOLOS_PREFIX=%s\n' "$tools_directory/dolos"
  printf 'MATHEEL_NODE=%s\n' "$tools_directory/node/bin/node"
} >> "$environment_file"
