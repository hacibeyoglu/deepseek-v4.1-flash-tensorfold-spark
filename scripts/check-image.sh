#!/usr/bin/env bash
# Check that a native TensorFold image holds both engines, both asset tools and
# every shared library the engine opens at run time.
#   scripts/check-image.sh IMAGE    from the build host
#   check-image.sh --inside         inside the image, e.g. as a Dockerfile RUN step
# libproof checks the pinned cuBLAS / cuBLASLt hashes, NCCL, verbs, the mlx5
# provider, libtfgrammar and the pinned JPEG/WebP decoders.
set -uo pipefail
BIN=${BIN:-/opt/tensorfold/bin}
if [[ "${1:-}" != --inside ]]; then
    image=${1:?usage: check-image.sh IMAGE | --inside}
    platform=$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$image") || exit 2
    exec docker run --rm -i --platform "$platform" --entrypoint bash "$image" \
        -s -- --inside < "${BASH_SOURCE[0]}"
fi
fail=0
for binary in tensorfold-dsv41 tf-dsv41 tf-dsv41-token-map tf-dsv41-assets; do
    [[ -x "$BIN/$binary" ]] || { echo "  $binary: MISSING"; fail=1; }
done
[[ -f /opt/tensorfold/share/tp_mailbox.fatbin ]] || { echo "  tp_mailbox.fatbin: MISSING"; fail=1; }
[[ -x /usr/bin/python3 ]] || { echo "  /usr/bin/python3: MISSING (compute lane)"; fail=1; }
bash /opt/tensorfold/share/libproof.sh || fail=1
(( fail == 0 )) && echo "check-zig-image: OK" || echo "check-zig-image: FAILED"
exit $fail
