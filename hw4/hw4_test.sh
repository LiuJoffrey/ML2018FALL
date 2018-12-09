#!/bin/bash
if [ -f "./modelW2V-00003-0.76110.h5" ]; then
    echo "modelW2V-00003-0.76110.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/9c86ik0v2vu422o/modelW2V-00003-0.76110.h5
    
fi
if [ -f "./0_modelW2V-00004-0.75780.h5" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "0_modelW2V-00004-0.75780.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/c8f20g5m0bv6xl7/0_modelW2V-00004-0.75780.h5
fi


#if [ -f "./1_modelW2V-00002-0.75650.h5" ]; then
#    # 檔案 /path/to/dir/filename 存在
#    echo "1_modelW2V-00002-0.75650.h5 exists, skip download."
#else
#    wget https://www.dropbox.com/s/mjgynhfyyhsppg3/1_modelW2V-00002-0.75650.h5
#fi

#if [ -f "./2_modelW2V-00003-0.75340.h5" ]; then
#    # 檔案 /path/to/dir/filename 存在
#    echo "2_modelW2V-00003-0.75340.h5 exists, skip download."
#else
#    wget https://www.dropbox.com/s/qzh6emuqz5wutp5/2_modelW2V-00003-0.75340.h5
#fi



python hw4_test.py $1 $2 $3
