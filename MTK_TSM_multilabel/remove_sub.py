import onnx
from onnx import helper

# 讀入原始模型
model = onnx.load("/data/ivs01/MTK_TSM/TSM_other/To_eNeural/checkpoint/20240924_V2_lr_0.01_fix_interval_0.125s_cos/tsm_MBV2_0116.onnx")

new_nodes = []
for node in model.graph.node:
    if node.op_type == "Sub":
        print(f"🔧 修正 Sub 節點: {node.name}")

        # 生成 Neg 節點（將第二個輸入變負）
        neg_output = node.input[1] + "_neg"
        neg_node = helper.make_node(
            "Neg",
            inputs=[node.input[1]],
            outputs=[neg_output],
            name=node.name + "_neg"
        )

        # 將 Sub(x, y) 改為 Add(x, -y)
        add_node = helper.make_node(
            "Add",
            inputs=[node.input[0], neg_output],
            outputs=node.output,
            name=node.name + "_add"
        )

        new_nodes.extend([neg_node, add_node])
    else:
        new_nodes.append(node)

# 替換整個 node list
model.graph.ClearField("node")
model.graph.node.extend(new_nodes)

# 儲存修改後的模型
onnx.save(model, "tsm_MBV2_fixed_all_sub.onnx")
print("✅ 所有 Sub 節點已修正，儲存為 tsm_MBV2_fixed_all_sub.onnx")
