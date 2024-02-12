# Experimental Guidance

## Download

For easier download of this repository, we provide a compressed zip on Kaggle, which can be downloaded using the following commands:

```
# First install kaggle.
pip install kaggle
pip show kaggle 

# Then set up kaggle API credentials.
mkdir ~/.kaggle # or /root/.kaggle
cd ~/.kaggle # or /root/.kaggle
vim kaggle.json # input your Kaggle API, in the format of {"username":"xx","key":"xx"}

# In the following, we assume that the downloaded path is /root/lts-exp.
cd /root
kaggle datasets download ANONYMOUS1111111/lts-exp
unzip lts-exp.zip
```

The structure of this repository is as follows:

-   `bash`: Folder of scripts for running experiments.
-   `datasets`: Folder of datasets used in experiments.
-   `iotdb-cli-0.12.4`: Folder of the IoTDB client.
-   `iotdb-server-0.12.4`: Folder of the IoTDB server.
-   `jarCode`: Folder of JAVA source codes for jars used in experiments.
-   `jars`: Folder of jars used in experiments to write data to IoTDB and query data from IoTDB.
-   `notebook`: Folder for the Python Jupyter Notebooks for experiments on visualization efficacy.
-   `tools`: Folder of tools to assist automated experiment scripts.

## 1. Visualization Efficacy Comparison

### Figure 6: SSIM comparison experiments

notebook/exp1-ssim-compare.ipynb

### Figure 7: ILTS convergence and effective area comparison on HouseTwenty dataset

notebook/exp2-convergence.ipynb

## 2. Query Efficiency Comparison

### Figure 8: Vary the number of output points m

1. Enter the `bash` folder and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `TRI_VISUALIZATION_EXP` folder.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments. After this step is completed, "finish" will be printed on the screen. This step will take some time to complete, please be patient.

2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]_TEST-efficiency-exp.sh 2>&1 &`, where `[datasetName]` is `FreezerRegularTrain`/`HouseTwenty`/`Lightning7`/`Mallat`/`OliveOil`/`Wine`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-[datasetName]_TEST-efficiency.csv`.


### Figure 9: Vary the number of input points n

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `TRI_VISUALIZATION_EXP` folder.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.

2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]_TEST-scalability-exp.sh 2>&1 &`, where `[datasetName]` is `FreezerRegularTrain`/`HouseTwenty`/`Lightning7`/`Mallat`/`OliveOil`/`Wine`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-[datasetName]_TEST-scalability.csv`.

## 3. Ablation Study

Corresponding to Figure 11 and 12.

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `TRI_VISUALIZATION_EXP` folder.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.

2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-HouseTwenty_TEST-ablation-exp.sh 2>&1 &`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.

3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in `HOME_PATH/res-HouseTwenty_TEST-ablation.csv`.

## 4. Overhead Evaluation

Corresponding to Figure 13.

1. (If you have done this preparation step in the previous experiments, you can ignore it here and directly go to step 2.) Enter the `bash` folder and then:

    -   Make all scripts executable by executing `chmod +x *.sh`.

    -   Update `prepare-all.sh` as follows:

        -   Update `TRI_VISUALIZATION_EXP` as the downloaded path of the `TRI_VISUALIZATION_EXP` folder.

        -   Update `HOME_PATH` as an **empty** folder where you want the experiments to be executed.

    -   Run `prepare-all.sh` and then the folder at `HOME_PATH` will be ready for experiments.
2. Enter the folder at `HOME_PATH`, and run experiments using `nohup ./run-[datasetName]_TEST-overhead-exp.sh 2>&1 &`, where `[datasetName]` is `FreezerRegularTrain`/`HouseTwenty`/`Lightning7`/`Mallat`/`OliveOil`/`Wine`. The running logs are saved in nohup.out, which can be checked by the command: `tail nohup.out`.
3. When the experiment script finishes running ("ALL FINISHED!" appears in nohup.out), the corresponding experimental results are in running logs in the form of "write latency of [datasetName]_TEST (without convex hull) is: duration_ns ns", where the "duration_ns" is the write latency result when **without** convex hull precomputation.
4. The write latency results when **with** convex hull precomputation are similarly shown in the running logs when you first run the previous query efficiency comparison experiment for Figure 8.
5. The space consumption can be checked using the command `du -s *` under the path `HOME_PATH/dataSpace_noConvexHull/[datasetName]_TEST_O_10_D_0_0/data/unsequence/root.[datasetName]/0` for **without** convex hull case, and `HOME_PATH/dataSpace/[datasetName]_TEST_O_10_D_0_0/data/unsequence/root.[datasetName]/0` for **with** convex hull case.
