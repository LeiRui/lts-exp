#!/bin/bash
HOME_PATH=/data/rl/v1

IOTDB_SBIN_HOME=$HOME_PATH/iotdb-server-0.12.4/sbin
QUERY_JAR_PATH=$HOME_PATH/QueryData*.jar # check TODO

# only true in run-more-baselines.sh for saving query result csv for DSSIM exp
REP_ONCE=false
#SAVE_QUERY_RESULT_PATH=NONE

echo 3 | sudo tee /proc/sys/vm/drop_caches
cd $IOTDB_SBIN_HOME

echo $REP_ONCE

#if [ $# -eq 8 ]
if $REP_ONCE
then
  a=1
else # default
  a=1
fi
echo "rep=$a"

for((i=0;i<a;i++)) do
    echo $i
    ./start-server.sh /dev/null 2>&1 &
    sleep 15s

    # device measurement timestamp_precision dataMinTime dataMaxTime range m approach save_query_result save_query_path
    java -jar $QUERY_JAR_PATH $1 $2 $3 $4 $5 $6 $7 $8 false NONE

    ./stop-server.sh
    sleep 5s
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sleep 5s
done
