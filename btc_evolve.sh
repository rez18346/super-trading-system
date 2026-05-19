#!/bin/bash
cd /home/ksysha/.openclaw/industrial_super_system
source venv/bin/activate
export PYTHONPATH="/home/ksysha/.openclaw/industrial_super_system:$PYTHONPATH"
python3 -c "
from auto_evolve import evolve_btc
evolve_btc()
" >> /tmp/btc_evolve.log 2>&1
