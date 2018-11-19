#!/bin/bash
if [ -f "./modelAtten_3_Alexnet_00342_0.71621.h5" ]; then
    echo "modelAtten_3_Alexnet_00342_0.71621.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/2l3krpy6p3u0pue/modelAtten_3_Alexnet_00342_0.71621.h5
fi
if [ -f "./model_reloadAlexnet_0.72588l.h5" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "model_reloadAlexnet_0.72588l.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/mrq5gfslicjkjy5/model_reloadAlexnet_0.72588l.h5
fi
python hw3_test.py $1 $2