#!/bin/bash
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate agents
streamlit run app/study_tool.py