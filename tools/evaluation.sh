#! /bin/bash

BASE_DIR="/lisiyuan/jx/AGVBench/work_dirs/classification/_starlknet_ok/vera220"

RELATIVE_PATHS=(
    r18_vanilla_bs32_sz224/20260118_000449.log.json
    r18_cutmix_bs32_sz224/20260118_005122.log.json
    r18_mixup_bs32_sz224/20260118_013341.log.json
    r18_saliencymix_bs32_sz224/20260118_021837.log.json
    r18_starmix_bs32_sz224/20260118_031246.log.json
    r18_starmixplus_bs32_sz224/20260118_035806.log.json
    r50_vanilla_bs32_sz224/20260118_045336.log.json
    r50_mixup_bs32_sz224/20260118_055020.log.json
    r50_cutmix_bs32_sz224/20260118_064834.log.json
    r50_saliencymix_bs32_sz224/20260118_074520.log.json
    r50_starmix_bs32_sz224/20260118_084445.log.json
    r50_starmixplus_bs32_sz224/20260118_094306.log.json
    ampvnet_vanilla_sz224_bs32/20260118_110911.log.json
    ampvnet_mixup_sz224_bs32/20260118_114920.log.json
    ampvnet_cutmix_sz224_bs32/20260118_123029.log.json
    ampvnet_saliencymix_sz224_bs32/20260118_131113.log.json
    ampvnet_starmix_sz224_bs32/20260118_135311.log.json
    ampvnet_starmixplus_sz224_bs32/20260118_143530.log.json
    fvrasnet_vanilla_sz224_bs32/20260118_152610.log.json
    fvrasnet_mixup_sz224_bs32/20260118_160554.log.json
    fvrasnet_cutmix_sz224_bs32/20260118_164613.log.json
    fvrasnet_saliencymix_sz224_bs32/20260118_172626.log.json
    fvrasnet_starmix_sz224_bs32/20260118_181106.log.json
    fvrasnet_starmixplus_sz224_bs32/20260118_185346.log.json
    vgg16_vanilla_sz224_bs32/20260118_194736.log.json
    vgg16_mixup_sz224_bs32/20260118_210746.log.json
    vgg16_cutmix_sz224_bs32/20260118_222653.log.json
    vgg16_saliencymix_sz224_bs32/20260118_234704.log.json
    vgg16_starmix_sz224_bs32/20260119_010956.log.json
    vgg16_starmixplus_sz224_bs32/20260119_022922.log.json
)

# Process each JSON file in the array
for rel_path in "${RELATIVE_PATHS[@]}"; do
    json_path="${BASE_DIR}/${rel_path}"
    echo "Processing: $json_path"
    python /lisiyuan/jx/AGVBench/tools/summary/find_classification_val_median.py \
        "$json_path" 600 10 head0_top1
    echo "---"
done