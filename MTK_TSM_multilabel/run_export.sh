#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "Usage: $0 --checkpoint <path_to_pth> --output <output_onnx_path> [extra export_onnx.py args]"
    echo ""
    echo "  --checkpoint   Path to .pth model checkpoint (required)"
    echo "  --output       Path for the final fixed .onnx file (required)"
    echo "  Remaining args are forwarded to export_onnx.py"
    exit 1
}

CHECKPOINT=""
OUTPUT_ONNX=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --output)
            OUTPUT_ONNX="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$OUTPUT_ONNX" ]]; then
    echo "Error: --checkpoint and --output are required."
    usage
fi

INTERMEDIATE_ONNX="${OUTPUT_ONNX%.onnx}_raw.onnx"

# ── Step 1: export to ONNX ──────────────────────────────────────────────────
echo "==> [1/2] Exporting model to ONNX..."
cd "$SCRIPT_DIR"
python export_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --onnx_path  "$INTERMEDIATE_ONNX" \
    "${EXTRA_ARGS[@]}"

echo "✅ Intermediate ONNX saved to: $INTERMEDIATE_ONNX"

# ── Step 2: replace Sub nodes with Neg+Add ──────────────────────────────────
echo "==> [2/2] Removing Sub nodes..."
python - "$INTERMEDIATE_ONNX" "$OUTPUT_ONNX" <<'PYEOF'
import sys
import onnx
from onnx import helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)

new_nodes = []
for node in model.graph.node:
    if node.op_type == "Sub":
        print(f"  Fixing Sub node: {node.name}")
        neg_out = node.input[1] + "_neg"
        new_nodes.append(helper.make_node(
            "Neg",
            inputs=[node.input[1]],
            outputs=[neg_out],
            name=node.name + "_neg"
        ))
        new_nodes.append(helper.make_node(
            "Add",
            inputs=[node.input[0], neg_out],
            outputs=node.output,
            name=node.name + "_add"
        ))
    else:
        new_nodes.append(node)

model.graph.ClearField("node")
model.graph.node.extend(new_nodes)
onnx.save(model, dst)
print(f"✅ All Sub nodes fixed, saved to: {dst}")
PYEOF

# ── Clean up intermediate file ───────────────────────────────────────────────
rm -f "$INTERMEDIATE_ONNX"
echo ""
echo "Done! Final ONNX: $OUTPUT_ONNX"
