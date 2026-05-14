def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            processed_line = ' '.join(parts[:-2]) + '.avi\n'
            processed_lines.append(processed_line)

    with open(output_file, 'w') as f:
        f.writelines(processed_lines)

input_file = "/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_label/count_RGBIR.txt"
output_file = "/data/ivs01/MTK_TSM/mtk_dms_20240924/mtk_dms_data_20240924_label/count_RGBIR.txt"

process_file(input_file, output_file)
