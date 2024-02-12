BASE_PATH=/root

TRI_VISUALIZATION_EXP=${BASE_PATH}/lts-exp
HOME_PATH=${BASE_PATH}/exp_home

VALUE_ENCODING=PLAIN # RLE for int/long, GORILLA for float/double
TIME_ENCODING=PLAIN # TS_2DIFF
COMPRESSOR=UNCOMPRESSED
DATA_TYPE=double

mkdir -p $HOME_PATH

find $TRI_VISUALIZATION_EXP -type f -iname "*.sh" -exec chmod +x {} \;
find $TRI_VISUALIZATION_EXP -type f -iname "*.sh" -exec sed -i -e 's/\r$//' {} \;

# check bc installed
REQUIRED_PKG="bc"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

#====prepare general environment====
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/tools/tool.sh .
cp $TRI_VISUALIZATION_EXP/jars/WriteDataUCR-*.jar .
cp $TRI_VISUALIZATION_EXP/jars/QueryDataUCR-*.jar .
cp $TRI_VISUALIZATION_EXP/tools/query_experiment.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH $HOME_PATH/query_experiment.sh
scp -r $TRI_VISUALIZATION_EXP/iotdb-server-0.12.4 .
scp -r $TRI_VISUALIZATION_EXP/iotdb-cli-0.12.4 .
cp $TRI_VISUALIZATION_EXP/tools/iotdb-engine-example.properties .
cp $TRI_VISUALIZATION_EXP/tools/ProcessResult.java .
cp $TRI_VISUALIZATION_EXP/tools/SumResultUnify.java .
# remove the line starting with "package" in the java file
sed '/^package/d' ProcessResult.java > ProcessResult2.java
rm ProcessResult.java
mv ProcessResult2.java ProcessResult.java
# then javac it
javac ProcessResult.java
# remove the line starting with "package" in the java file
sed '/^package/d' SumResultUnify.java > SumResultUnify2.java
rm SumResultUnify.java
mv SumResultUnify2.java SumResultUnify.java
# then javac it
javac SumResultUnify.java

#############################################
#====prepare run bash for efficiency exp====
#############################################
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET Wine_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Wine" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 140380000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 140380000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-Wine_TEST-efficiency-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET OliveOil_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.OliveOil" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 170990000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 170990000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-OliveOil_TEST-efficiency-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET Mallat_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Mallat" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 307190000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 307190000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-Mallat_TEST-efficiency-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET Lightning7_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Lightning7" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 191380000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 191380000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-Lightning7_TEST-efficiency-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET HouseTwenty_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.HouseTwenty" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 359994000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 359994000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-HouseTwenty_TEST-efficiency-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-efficiency-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-efficiency-exp.sh
$HOME_PATH/tool.sh DATASET FreezerRegularTrain_TEST run-efficiency-exp.sh
$HOME_PATH/tool.sh DEVICE "root.FreezerRegularTrain" run-efficiency-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-efficiency-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 180580000 run-efficiency-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 180580000 run-efficiency-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-efficiency-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-efficiency-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-efficiency-exp.sh
cp run-efficiency-exp.sh run-FreezerRegularTrain_TEST-efficiency-exp.sh

rm run-efficiency-exp.sh

#############################################
#====prepare run bash for scalability exp====
#############################################
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET Wine_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Wine" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 140380000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 140380000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-Wine_TEST-scalability-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET OliveOil_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.OliveOil" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 170990000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 170990000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-OliveOil_TEST-scalability-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET Mallat_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Mallat" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 307190000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 307190000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-Mallat_TEST-scalability-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET Lightning7_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Lightning7" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 191380000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 191380000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-Lightning7_TEST-scalability-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET HouseTwenty_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.HouseTwenty" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 359994000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 359994000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-HouseTwenty_TEST-scalability-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-scalability-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-scalability-exp.sh
$HOME_PATH/tool.sh DATASET FreezerRegularTrain_TEST run-scalability-exp.sh
$HOME_PATH/tool.sh DEVICE "root.FreezerRegularTrain" run-scalability-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-scalability-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-scalability-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 180580000 run-scalability-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 180580000 run-scalability-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-scalability-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-scalability-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-scalability-exp.sh
cp run-scalability-exp.sh run-FreezerRegularTrain_TEST-scalability-exp.sh

rm run-scalability-exp.sh

#############################################
#====prepare run bash for ablation exp====
#############################################
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET Wine_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Wine" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 140380000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 140380000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-Wine_TEST-ablation-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET OliveOil_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.OliveOil" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 170990000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 170990000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-OliveOil_TEST-ablation-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET Mallat_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Mallat" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 307190000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 307190000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-Mallat_TEST-ablation-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET Lightning7_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Lightning7" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 191380000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 191380000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-Lightning7_TEST-ablation-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET HouseTwenty_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.HouseTwenty" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 359994000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 359994000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-HouseTwenty_TEST-ablation-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-ablation-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-ablation-exp.sh
$HOME_PATH/tool.sh DATASET FreezerRegularTrain_TEST run-ablation-exp.sh
$HOME_PATH/tool.sh DEVICE "root.FreezerRegularTrain" run-ablation-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-ablation-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-ablation-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 180580000 run-ablation-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 180580000 run-ablation-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-ablation-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-ablation-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-ablation-exp.sh
cp run-ablation-exp.sh run-FreezerRegularTrain_TEST-ablation-exp.sh

rm run-ablation-exp.sh

#############################################
#====prepare run bash for overhead exp====
#############################################
cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET Wine_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Wine" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 140380000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 140380000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-Wine_TEST-overhead-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET OliveOil_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.OliveOil" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 170990000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 170990000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-OliveOil_TEST-overhead-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET Mallat_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Mallat" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 307190000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 307190000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-Mallat_TEST-overhead-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET Lightning7_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.Lightning7" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 191380000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 191380000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-Lightning7_TEST-overhead-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET HouseTwenty_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.HouseTwenty" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 359994000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 359994000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-HouseTwenty_TEST-overhead-exp.sh

cd $HOME_PATH
cp $TRI_VISUALIZATION_EXP/bash/run-overhead-exp.sh .
$HOME_PATH/tool.sh HOME_PATH $HOME_PATH run-overhead-exp.sh
$HOME_PATH/tool.sh DATASET FreezerRegularTrain_TEST run-overhead-exp.sh
$HOME_PATH/tool.sh DEVICE "root.FreezerRegularTrain" run-overhead-exp.sh
$HOME_PATH/tool.sh MEASUREMENT "test" run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_TYPE ${DATA_TYPE} run-overhead-exp.sh
$HOME_PATH/tool.sh TIMESTAMP_PRECISION ms run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MIN_TIME 1 run-overhead-exp.sh
$HOME_PATH/tool.sh DATA_MAX_TIME 180580000 run-overhead-exp.sh
$HOME_PATH/tool.sh TOTAL_POINT_NUMBER 180580000 run-overhead-exp.sh
$HOME_PATH/tool.sh IOTDB_CHUNK_POINT_SIZE 100000 run-overhead-exp.sh
$HOME_PATH/tool.sh VALUE_ENCODING ${VALUE_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh TIME_ENCODING ${TIME_ENCODING} run-overhead-exp.sh
$HOME_PATH/tool.sh COMPRESSOR ${COMPRESSOR} run-overhead-exp.sh
cp run-overhead-exp.sh run-FreezerRegularTrain_TEST-overhead-exp.sh

rm run-overhead-exp.sh

#############################################
#====prepare directory for each dataset====
#############################################
r=(20000 10000 10000 20000 6000 20000)
i=0
datasetArray=("Wine_TEST" "OliveOil_TEST" "Mallat_TEST" "Lightning7_TEST" "HouseTwenty_TEST" "FreezerRegularTrain_TEST");
for value in ${datasetArray[@]};
do
  echo "prepare data directory";
  cd $HOME_PATH
  mkdir $value
  cd $value
  cp $TRI_VISUALIZATION_EXP/datasets/$value.csv .
  cp $TRI_VISUALIZATION_EXP/tools/Enlarge.py .
  python3 Enlarge.py -i $value.csv -o $value-cp.csv -r ${r[i]}
  rm $value.csv
  rm Enlarge.py
  mv $value-cp.csv $value.csv

  echo "prepare testspace directory";
  cd $HOME_PATH
  mkdir ${value}_testspace

  let i+=1

done;

find $HOME_PATH -type f -iname "*.sh" -exec chmod +x {} \;

echo "finish"
