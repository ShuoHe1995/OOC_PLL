# OOC_PLL
 Code for "Partial-label Learning with Mixed Closed-set and Open-set Out-of-candidate Examples" (KDD 2023).

Requirements:

python==3.9.12

torch>=1.12.1

torchvision>=0.13.1

spicy

The directory of datasets:

data_dir: "../../datasets/"

run:

sh run_c-s.sh

sh run_c-c.sh

sh run_c-i.sh

Notes:
All comparing algorithms employed the same generated partially-labeled data in "utils/utils_data.py" and the uniform training scheme.