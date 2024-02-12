from typing import Union

import numpy as np

from .downsampling_interface import AbstractDownsampler

def _effective_area(x_bucket: np.ndarray, y_bucket: np.ndarray, sampled_x: np.ndarray):
    # 计算采样序列的effective area
    total_area=0
    for i in range(1,len(sampled_x)-1): # 第二个点到倒数第二个点
        l=sampled_x[i-1]
        c=sampled_x[i]
        r=sampled_x[i+1]

        t1=x_bucket[l]
        t2=x_bucket[c]
        t3=x_bucket[r]

        v1=y_bucket[l]
        v2=y_bucket[c]
        v3=y_bucket[r]

        total_area+=np.abs((t1-t3)*(v2-v3)-(t2-t3)*(v1-v3))/2

    return total_area


def _calculate_average_point(x_bucket: np.ndarray, y_bucket: np.ndarray):
    return np.sum(x_bucket)/len(x_bucket),np.sum(y_bucket)/len(y_bucket)

def _calculate_linear_regression_coefficients(x_bucket: np.ndarray, y_bucket: np.ndarray):
    avg_x,avg_y=_calculate_average_point(x_bucket,y_bucket)
   
    aNumerator = 0.0
    aDenominator = 0.0
    for i in range(len(x_bucket)):
        aNumerator += (x_bucket[i] - avg_x) * (y_bucket[i] - avg_y)
        aDenominator += (x_bucket[i] - avg_x) * (x_bucket[i] - avg_x)

    a = aNumerator / aDenominator
    b = avg_y - a*avg_x 

    return a,b

def _calculate_sse_for_bucket(x_bucket: np.ndarray, y_bucket: np.ndarray):
    a,b=_calculate_linear_regression_coefficients(x_bucket,y_bucket)
    sse = 0.0
    for i in range(len(x_bucket)):
        error = y_bucket[i] - (a*x_bucket[i] + b)
        sse += error * error

    return np.sqrt(sse/len(x_bucket))
    # return sse

def _get_ltd_bin_idxs(x: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    # adapted from https://github.com/haoel/downsampling/blob/master/core/ltd.go
    # 服务于构造非均匀分桶
    # 对标等点数分桶offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
    # 要求满足第一个数是1，最后一个数是y.shape[0]-1，一共有nout-1个数字，形成【nout-2】个intervals
    # 思路：
    # 1. 先分成等点数分桶
    # 2. 然后计算每个桶的一元多项式回归误差SSE （overlap前后桶各1个点是为了当桶内只有1个点的时候体现context比如台阶拐点）
    # 3. 然后找出SSE最大的桶和相邻SSE之和最小的桶（排除最大的桶）
    # 4. 然后分裂SSE最大的桶
    # 5. 然后合并SSE最小的桶（注意index下标判断是否因为上一步而变化）
    # 6. 循环直到满足迭代停止次数条件
    # 7. 最后得到非均匀分桶，之后就可以在该分桶上应用LTOB或者LTTB采样
    ###############################################################

    # 1. 构造（基本）等点数分桶
    # Bucket size. Leave room for start and end data points
    block_size = (y.shape[0] - 2) / (n_out - 2)
    # Note this 'astype' cast must take place after array creation (and not with the
    # aranage() its dtype argument) or it will cast the `block_size` step to an int
    # before the arange array creation
    offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
    # nout-2 buckets
    # .astype(np.int64)是floor操作
    # number of intervals = floor((stop-start)/step)=n-2+floor((n-2)/(N-2))=n-2!
    # 也就是说最后一个数一定是y.shape[0]-1
    # 1, 1+bs, 1+2*bs, ..., 1+(nout-2)*bs=y.shape[0]-1
    # 得到上述nout-2个等间隔（浮点数bs间隔）之后，最后才astype(np.int64)，把中间的浮点数floor成整数
    # 这样虽然最后得到的不是严格的等点数buckets可能稍微加减1个点之类的，但是buckets个数是不变的，buckets的连贯性也是不变的

    # numIterations = int(np.floor(len(x) * 10 / n_out))
    numIterations = 1000
    sse=np.zeros(n_out-2)
    for c in range(numIterations): # 6. 循环直到满足迭代停止次数条件
        # 2. 然后计算每个桶的一元多项式回归误差SSE （overlap前后桶各1个点是为了当桶内只有1个点的时候体现context比如台阶拐点）
        # "The last point in the previous bucket and the first point in the next bucket are also included in the regression."
        for i in range(n_out - 2):
            # （overlap前后桶各1个点是为了当桶内只有1个点的时候体现context比如台阶拐点）
            x_bucket = x[offset[i]-1 : offset[i + 1]+1]
            y_bucket = y[offset[i]-1 : offset[i + 1]+1]
            sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket) # replace all data

        # 3. 然后找出SSE最大的桶和相邻SSE之和最小的桶（排除最大的桶）
        # 注意offset里有nout-1个数字，形成了nout-2个buckets，对应nout-2个sse

        # 以下寻找maxSSEIndex是错误实现：会出现argmax找到的bucket是只包含了一个台阶拐点的，然后因为该bucket内只有1个点而停止迭代。
        # 正确做法应该是手动找max中跳过那些只包含1个点的buckets。
        # 顺便好像理解了为什么sse要用前后桶overlap一个点来计算：这样才能体现台阶拐点
        # maxSSEIndex = np.argmax(sse)
        # if offset[maxSSEIndex+1]-offset[maxSSEIndex] <= 1: 
        #     # 此时就是说这个bucket里只有1个点，那为什么还会这个bucket的SSE最大呢
        #     # 这是因为sse用了前后bucket各自1个点overlap，所以一个例子是这个点周围两个点相差很大
        #     # 比如一个台阶的拐点，加上其两边一个点，这三个点用一次多项式拟合之后的SSE往往大
        #     break

        maxSSEIndex = -1
        maxSSE = np.finfo(np.float64).min
        for i in range(n_out - 2):
            if offset[i+1]-offset[i] <= 1:
                # 此时就是说这个bucket里只有1个点，那为什么还会这个bucket的SSE最大呢
                # 这是因为sse用了前后bucket各自1个点overlap，所以一个例子是这个点周围两个点相差很大
                # 比如一个台阶的拐点，加上其两边一个点，这三个点用一次多项式拟合之后的SSE往往大
                continue
            if sse[i] > maxSSE:
                maxSSE = sse[i]
                maxSSEIndex = i
        if maxSSEIndex < 0:
            print(c)
            print(maxSSEIndex)
            print('break max')
            break

        minSSEIndex = -1
        minSSE = np.finfo(np.float64).max
        for i in range(n_out-3):
            if i==maxSSEIndex or i+1==maxSSEIndex:
                continue
            if sse[i]+sse[i+1] < minSSE:
                minSSE = sse[i]+sse[i+1]
                minSSEIndex = i
        if minSSEIndex < 0:
            print(c)
            print(minSSEIndex)
            print('break min')
            break

        # print('maxSSEIndex',maxSSEIndex,'minSSEIndex',minSSEIndex)

        # 4. 然后分裂SSE最大的桶
        startIdx = offset[maxSSEIndex]
        endIdx = offset[maxSSEIndex+1]
        middleIdx = int(np.floor((startIdx+endIdx)/2))
        tmp=list(offset)
        tmp.insert(maxSSEIndex+1,middleIdx)
        offset=np.array(tmp)
 
        # 5. 然后合并SSE最小的桶（注意index下标判断是否因为上一步而变化）
        # 注意sse是interval的，offset是intervals的边界点
        # 把第minSSEIndex个interval的右边界点删掉，即达到merge第minSSEIndex与第minSSEIndex+1个intervals的目的
        # 从0开始计数
        if minSSEIndex > maxSSEIndex: 
            # 注意这个细节
            minSSEIndex += 1 
        tmp=list(offset)
        tmp.pop(minSSEIndex+1)
        offset=np.array(tmp)

    # 7. 最后得到非均匀分桶
    for i in range(n_out - 2):
        # （overlap前后桶各1个点是为了当桶内只有1个点的时候体现context比如台阶拐点）
        x_bucket = x[offset[i]-1 : offset[i + 1]+1]
        y_bucket = y[offset[i]-1 : offset[i + 1]+1]
        sse[i]=_calculate_sse_for_bucket(x_bucket,y_bucket) # replace all data
    print('final offset',offset)
    print('sse of final offset',sse)
    return offset

def _get_bin_idxs_nofirstlast_gapAware_deprecated(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 专门服务于等时间间隔分桶且空桶保留的方法，见LTTBETGapDownsampler
    # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成nb_bins份，前面都是左闭右开，最后一个左闭右闭。
    # 输出的是把时间分桶转换成对应数据点下标的分界
    # 输出的第i个桶[bins[i]:bins[i+1]]是那些时间戳落在第i个时间分桶内的点的index范围，注意python里[bins[i]:bins[i+1]]右边是开的
    # starts from 1（即原序列的第二个点，闭）, ends at len(x)-1（即原序列的倒数第一个点开，等价于闭在原序列的倒数第二个点）
    # 留出首尾点，是为了照顾到LTOBET需要首尾点留出来padding计算三角形面积

    # 没有加np.unique的时候结果输出nb_bins+1个点，形成nb_bins个桶
    # 加了np.unique之后注意可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了，所以等时间分桶的输出桶数小于等于nb_bins
    # 就不保证一定是nb_bins个桶！
    # 这里不加np.unique！

    # 现在这样定义的等时间分桶和其它的等点数分桶统一了，它们都是第一个数是1、最后一个数是len(x)-1
    # 三角采样统一做法是给定nout采样点数，把预留的首尾点也算在输出里，从而分nout-2个桶，即这里nb_bins=nout-2
    # 而MinMax,M4,LTOBET给定nout采样点数，就是分nout个桶，即_get_bin_idxs那里nb_bins=nout

    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    # x[1]是第二个点的时间戳，x[-2]是倒数第二个点的时间戳
    # 把时间范围[x[1],x[-2]]均匀分成nb_bins份，前面都是左闭右开，最后一个左闭右闭。
    bins = np.searchsorted(x, np.linspace(x[1], x[-2], nb_bins + 1), side="left")
    bins[-1] = len(x)-1 # 为了右开实际形成把倒数第二个点闭进来，否则bins[-1]=len(x)-2会把倒数第二个点开出去
    # 或者用bins[-1] = bins[-1]+1也行

    # Returns the sorted unique elements of an array.
    # 有的等时间分桶里就是没有点的话就应该是两个相同的indexes表征这个桶是空的！
    # 我知道为什么要np.unique了，就是考虑到等时间分桶不同于等点数分桶，前者可能有的桶里没有点，就表现为连续两个相同的下标，
    # 所以用unique就能排除掉这个空桶!
    # 这里不加np.unique！
    return bins

def _get_bin_idxs_stepLTTBET(x: np.ndarray, t2:int, tn:int, nb_bins: int) -> np.ndarray:
    bins = np.searchsorted(x, np.linspace(t2, tn, nb_bins + 1), side="left")
    return np.unique(bins)

def _get_bin_idxs_nofirstlast_gapAware(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 专门服务于等时间间隔分桶且空桶保留的方法，见LTTBETGapDownsampler
    # tmin是原序列的第二个点的时间戳，tmax是原序列的最后一个点的时间戳，把[tmin,tmax]均匀分成nb_bins份，都是左闭右开。
    # 输出的是把时间分桶转换成对应数据点下标的分界
    # 输出的第i个桶[bins[i]:bins[i+1]]是那些时间戳落在第i个时间分桶内的点的index范围，注意python里[bins[i]:bins[i+1]]右边是开的
    # starts from 1（即原序列的第二个点，闭）, ends at len(x)-1（即原序列的倒数第一个点开，等价于闭在原序列的倒数第二个点）
    # 留出首尾点，是为了照顾到LTOBET需要首尾点留出来padding计算三角形面积

    # 没有加np.unique的时候结果输出nb_bins+1个点，形成nb_bins个桶
    # 加了np.unique之后注意可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了，所以等时间分桶的输出桶数小于等于nb_bins
    # 就不保证一定是nb_bins个桶！
    # 这里不加np.unique！

    # 现在这样定义的等时间分桶和其它的等点数分桶统一了，它们都是第一个数是1、最后一个数是len(x)-1
    # 三角采样统一做法是给定nout采样点数，把预留的首尾点也算在输出里，从而分nout-2个桶，即这里nb_bins=nout-2
    # 而MinMax,M4,LTOBET给定nout采样点数，就是分nout个桶，即_get_bin_idxs那里nb_bins=nout

    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    # x[1]是第二个点的时间戳，x[-1]是最后一个点的时间戳
    # 把时间范围[x[1],x[-1]]均匀分成nb_bins份，都是左闭右开
    bins = np.searchsorted(x, np.linspace(x[1], x[-1], nb_bins + 1), side="left")

    # Returns the sorted unique elements of an array.
    # 有的等时间分桶里就是没有点的话就应该是两个相同的indexes表征这个桶是空的！
    # 我知道为什么要np.unique了，就是考虑到等时间分桶不同于等点数分桶，前者可能有的桶里没有点，就表现为连续两个相同的下标，
    # 所以用unique就能排除掉这个空桶!
    # 这里不加np.unique！
    return bins

def _get_bin_idxs_nofirstlast(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 专门服务于等时间间隔分桶的方法，见LTOBETDownsampler, LTTBETDownsampler
    # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成nb_bins份，前面都是左闭右开，最后一个左闭右闭。
    # 输出的是把时间分桶转换成对应数据点下标的分界
    # 输出的第i个桶[bins[i]:bins[i+1]]是那些时间戳落在第i个时间分桶内的点的index范围，注意python里[bins[i]:bins[i+1]]右边是开的
    # starts from 1（即原序列的第二个点，闭）, ends at len(x)-1（即原序列的倒数第一个点开，等价于闭在原序列的倒数第二个点）
    # 留出首尾点，是为了照顾到LTOBET需要首尾点留出来padding计算三角形面积

    # 没有加np.unique的时候结果输出nb_bins+1个点，形成nb_bins个桶
    # 加了np.unique之后注意可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了，所以等时间分桶的输出桶数小于等于nb_bins
    # 就不保证一定是nb_bins个桶！

    # 现在这样定义的等时间分桶和其它的等点数分桶统一了，它们都是第一个数是1、最后一个数是len(x)-1
    # 三角采样统一做法是给定nout采样点数，把预留的首尾点也算在输出里，从而分nout-2个桶，即这里nb_bins=nout-2
    # 而MinMax,M4,LTOBET给定nout采样点数，就是分nout个桶，即_get_bin_idxs那里nb_bins=nout

    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    
    bins=_get_bin_idxs_nofirstlast_gapAware(x,nb_bins)
    # 这样相当于把连续的空桶和左边最近的非空桶合并起来，即后者的右边界现在是连续空桶的最右边界
    # 以此消除了空桶
    return np.unique(bins)

def _get_bin_idxs_gapAware(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 服务于等时间间隔分桶的方法，见MinMaxGapDownsampler
    # tmin是原序列的第一个点的时间戳，tmax是原序列的最后一个点的时间戳，把[tmin,tmax]均匀分成nb_bins份，前面都是左闭右开，最后一个左闭右闭。
    # 输出的是把时间分桶转换成对应数据点下标的分界
    # 输出的第i个桶[bins[i]:bins[i+1]]是那些时间戳落在第i个时间分桶内的点的index范围，注意python里[bins[i]:bins[i+1]]右边是开的
    # starts from 0（即原序列的第一个点，闭）, ends at len(x)（等价于闭在原序列的最后一个点）

    # 不加np.unique的时候结果是nb_bins+1个点，第一个数是0（即原序列的第1个点，闭），最后一个数是len(x)（即囊括了原序列的最后一个点），形成nb_bins个桶
    # 加了np.unique之后注意可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了，所以等时间分桶的输出桶数小于等于nb_bins
    # 就不保证一定是nb_bins个桶！

    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    # x[0]是第一个点的时间戳，x[-1]是最后一个点的时间戳
    # 把时间范围[x[0],x[-1]]均匀分成nb_bins份，前面都是左开右闭，最后一个桶是左闭右闭
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="left")
    # bins[0] = 0 # left的话本来就是等于这个
    bins[-1] = len(x) # 为了右开实际形成把最后一个点闭进来，否则bins[-1]=len(x)-1会把最后一个点开出去
    # 或者用bins[-1] = bins[-1]+1也行
    
    # Returns the sorted unique elements of an array.
    # 有的等时间分桶里就是没有点的话就应该是两个相同的indexes表征这个桶是空的！
    # 我知道为什么要np.unique了，就是考虑到等时间分桶不同于等点数分桶，前者可能有的桶里没有点，就表现为连续两个相同的下标，
    # 所以用unique就能排除掉这个空桶!
    # 这里不用np.unique!
    return bins

def _get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    # 服务于等时间间隔分桶的方法，见MinMaxDownsampler、M4Downsampler
    # tmin是原序列的第一个点的时间戳，tmax是原序列的最后一个点的时间戳，把[tmin,tmax]均匀分成nb_bins份，前面都是左闭右开，最后一个左闭右闭。
    # 输出的是把时间分桶转换成对应数据点下标的分界
    # 输出的第i个桶[bins[i]:bins[i+1]]是那些时间戳落在第i个时间分桶内的点的index范围，注意python里[bins[i]:bins[i+1]]右边是开的
    # starts from 0（即原序列的第一个点，闭）, ends at len(x)（等价于闭在原序列的最后一个点）

    # 不加np.unique的时候结果是nb_bins+1个点，第一个数是0（即原序列的第1个点，闭），最后一个数是len(x)（即囊括了原序列的最后一个点），形成nb_bins个桶
    # 加了np.unique之后注意可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了，所以等时间分桶的输出桶数小于等于nb_bins
    # 就不保证一定是nb_bins个桶！

    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    
    # Returns the sorted unique elements of an array.
    # 有的等时间分桶里就是没有点的话就应该是两个相同的indexes表征这个桶是空的！
    # 我知道为什么要np.unique了，就是考虑到等时间分桶不同于等点数分桶，前者可能有的桶里没有点，就表现为连续两个相同的下标，
    # 所以用unique就能排除掉这个空桶!
    # print(np.unique(bins))
    bins=_get_bin_idxs_gapAware(x,nb_bins)
    return np.unique(bins)

# # deprecated LLBDownsampler
# class LLBDownsampler(AbstractDownsampler): 
# # NOTE: 这个目前是类似LTOB的实现，即连线仅仅是直接相邻的两点，而不是前后桶！！！
# # TODO: 参照LTTB改成前后桶连线的实现
#     @staticmethod
#     def _argmax_area(x_bucket, y_bucket) -> int:
#         """Vectorized triangular area argmax comput ation.

#         Parameters
#         ----------
#         x_bucket : np.ndarray
#             All x values in the bucket, 
#             padded with the last point of the previous bucket and the first point of the afterward bucket,
#             so that the first and last points of the current bucket can calculate their triangle area.
#         y_bucket : np.ndarray
#             All y values in the bucket,
#             padded with the last point of the previous bucket and the first point of the afterward bucket,
#             so that the first and last points of the current bucket can calculate their triangle area.

#         Returns
#         -------
#         int
#             The index of the point with the longest line.
#         """

#         result=[]
#         for i in range(len(x_bucket)-2):
#             j=i+1
#             p1x=x_bucket[j-1]
#             p2x=x_bucket[j]
#             p3x=x_bucket[j+1]
#             p1y=y_bucket[j-1]
#             p2y=y_bucket[j]
#             p3y=y_bucket[j+1]
#             l1=(p2x-p1x)**2+(p2y-p1y)**2
#             l1=np.sqrt(l1)
#             l2=(p2x-p3x)**2+(p2y-p3y)**2
#             l2=np.sqrt(l2)
#             result.append(l1+l2)

#         result = np.array(result)
#         return result.argmax()

#     def _downsample(
#         self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
#     ) -> np.ndarray:
#         """TODO complete docs"""
#         if x is None:
#             # Is fine for this implementation as this is only used for testing
#             x = np.arange(y.shape[0])

#         # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
#         # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
#         bins = _get_bin_idxs_nofirstlast(x, n_out-2)

#         # Construct the output array
#         # 等时间分桶不能用固定长度的数组，因为有可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了
#         # 所以等时间分桶的输出桶数小于等于n_out
#         sampled_x = []
#         sampled_x.append(0)

#         # Convert x & y to int if it is boolean
#         if x.dtype == np.bool_:
#             x = x.astype(np.int8)
#         if y.dtype == np.bool_:
#             y = y.astype(np.int8)

#         # 注意这里不能固定遍历n_out，因为等时间分桶未必有n_out个桶，排除掉空桶
#         for lower, upper in zip(bins, bins[1:]):
#             # print(lower,upper)

#             if upper==lower: # 空桶，但是不可能出现这种情况，因为_get_bin_idxs最后用np.unique处理了
#                 continue

#             a = (
#                 LLBDownsampler._argmax_area(
#                     x_bucket=x[lower-1 : upper+1],
#                     y_bucket=y[lower-1 : upper+1],
#                     # offset starts from 1, ends at len(x)-1
#                     # so lower-1 >= 0, upper+1 <= len(x)
#                     # and x[] is left included and right excluded, so correct.
#                     # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
#                     # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
#                 )
#                 + lower
#             )
#             sampled_x.append(a)

#         sampled_x.append(len(x)-1)

#         # Returns the sorted unique elements of an array.
#         return np.unique(sampled_x)

class LTOBETGapDownsampler(AbstractDownsampler):
    # 和LTOB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTOB的区别：LTOB使用等点数分桶，LTOBET使用等时间分桶
    # 和LTOBET的区别：LTOBETGap采取空桶感知策略，遇到空桶的话就断开，以此保留空桶gap两端点
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        x_bucket : np.ndarray
            All x values in the bucket, 
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.
        y_bucket : np.ndarray
            All y values in the bucket,
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """

        # the following vectorized triangular area argmax computation is adapted from 
        # https://github.com/Permafacture/Py-Visvalingam-Whyatt/blob/master/polysimplify.py

        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        # print(result)

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for
            # testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后没有执行np.unique来合并空桶，所以bins形成的桶数就是n_out-2个，但是有可能有空桶
        # starts at 1, ends at len(x)-1
        bins = _get_bin_idxs_nofirstlast_gapAware(x, n_out-2)
        # print(bins)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        for i in range(len(bins)-1): # len(bins)-1个分桶，len(bins)-1=n_out-2 
            # 如果当前桶是空桶就跳过
            if bins[i]==bins[i + 1]: # 当前桶是空的
                continue

            # 现在当前桶非空
            # offset starts from 1, ends at len(x)-1
            # so bins[i]-1 >= 0, bins[i + 1]+1 <= len(x)
            # and x[] is left included and right excluded, so correct.
            # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
            # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
            # aaaaaaaaahhhhhhhhhhhhhhhhhhhhhhh debug a bug:
            # a=np.array([1,2,3,4])
            # b=a[1:4]
            # b[0]=-1
            # b[-1]=a[0]
            # print(a)
            # print(b)
            # ndarray change b will also change a!!!!!!
            # So .copy() is necessary here!! as i will revise x_bucket later but I don't want to revise x!!
            x_bucket=x[bins[i]-1 : bins[i + 1]+1].copy()
            y_bucket=y[bins[i]-1 : bins[i + 1]+1].copy()

            # 判断前一个桶是否空
            # i=0的时候前一个桶就是首点一定非空
            # 事实上i=1的时候也就是中间第二个桶的前一个桶也一定非空，因为我们默认第二个点一定存在，但是不管了
            if i>0 and bins[i-1]==bins[i]: # 前一个桶是空的
                
                # 以当前桶的第一个点作为新的全局第一个点
                sampled_x.append(bins[i]) # 最后会去重排序的所以没关系
                
                # 修改x_bucket里padding left point为当前桶的第一个点
                # 于是x_bucket里是x[bins[i]],x[bins[i]],x[bins[i+1]],...,x[bins[i+1]]
                # 对应计算三角形面积是x[bins[i]],x[bins[i+1]],...,x[bins[i+1]-1]点的
                # 此时x[bins[i]]点的三角形面积应该算出来面积是0，因为padding left point为当前桶的第一个点
                # 所以不会干扰选中maxTri（反正这个桶里第一个点已经单独append进结果了）
                x_bucket[0]=x[bins[i]] # 原本是x[bins[i]-1]
                y_bucket[0]=y[bins[i]]

            # 判断后一个桶是否空
            # i=len(bins)-2的时候后一个桶就是尾点一定非空
            # 事实上i=len(bins-3)的时候也就是中间倒数第二个桶的后一个桶也一定非空，因为我们默认倒数第二个点一定存在，但是不管了
            if i<len(bins)-2 and bins[i+1]==bins[i+2]: # 后一个桶是空的
                
                # 以当前桶的最后一个点作为新的全局最后一个点
                sampled_x.append(bins[i+1]-1) # 最后会去重排序的所以没关系
                
                # 修改x_bucket里padding right point为当前桶的最后一个点
                # 于是x_bucket里是x[bins[i]]或x[bins[i]-1],x[bins[i]],x[bins[i+1]],...,x[bins[i+1]-1],x[bins[i+1]-1]
                # 对应计算三角形面积是x[bins[i]],x[bins[i+1]],...,x[bins[i+1]-1]点的
                # 此时x[bins[i+1]-1]点的三角形面积应该算出来面积是0，因为padding right point为当前桶的最后一个点
                # 所以不会干扰选中maxTri（反正这个桶里最后一个点已经单独append进结果了）
                x_bucket[-1]=x[bins[i+1]-1] # 原本是x[bins[i+1]]。注意x_bucket=x[bins[i]-1 : bins[i + 1]+1]的格式右边取不到，而这里是取到的赋值！
                y_bucket[-1]=y[bins[i+1]-1]


            a = (
                LTOBETGapDownsampler._argmax_area(
                    x_bucket=x_bucket,
                    y_bucket=y_bucket,
                )
                + bins[i]
            )
            sampled_x.append(a)

        sampled_x.append(len(x)-1) # 最后一个点

        # Returns the sorted unique elements of an array.
        return np.unique(sampled_x) # unique既会去重还会排序

class LTOBETDownsampler(AbstractDownsampler):
    # 和LTOB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTOB的区别：LTOB使用等点数分桶，LTOBET使用等时间分桶
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        x_bucket : np.ndarray
            All x values in the bucket, 
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.
        y_bucket : np.ndarray
            All y values in the bucket,
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """

        # the following vectorized triangular area argmax computation is adapted from 
        # https://github.com/Permafacture/Py-Visvalingam-Whyatt/blob/master/polysimplify.py

        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        
        c=10000 # TODO
        
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        # print(result)

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for
            # testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)

        # Construct the output list
        # 等时间分桶不能用固定长度的数组，因为有可能有的桶里没有点就直接在生成等时间分桶的时候通过np.unique排除掉这个桶了
        # 所以等时间分桶的输出桶数小于等于n_out-2
        # 所以这里sampled_x用list，不用需要提前确定长度的数组
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        # 注意这里不能固定遍历n_out，因为等时间分桶未必有n_out个桶，排除掉空桶
        for lower, upper in zip(bins, bins[1:]):
            # print(lower,upper)

            if upper==lower: # 空桶，但是不可能出现这种情况，因为_get_bin_idxs最后用np.unique处理了
                continue

            a = (
                LTOBETDownsampler._argmax_area(
                    x_bucket=x[lower-1 : upper+1].copy(),
                    y_bucket=y[lower-1 : upper+1].copy(),
                    # offset starts from 1, ends at len(x)-1
                    # so lower-1 >= 0, upper+1 <= len(x)
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + lower
            )
            sampled_x.append(a)

        sampled_x.append(len(x)-1) # 最后一个点

        # Returns the sorted unique elements of an array.
        return np.unique(sampled_x)

class LTOBDownsampler(AbstractDownsampler):
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        x_bucket : np.ndarray
            All x values in the bucket, 
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.
        y_bucket : np.ndarray
            All y values in the bucket,
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """

        # the following vectorized triangular area argmax computation is adapted from 
        # https://github.com/Permafacture/Py-Visvalingam-Whyatt/blob/master/polysimplify.py

        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造（基本）等点数分桶
        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2)
        # Note this 'astype' cast must take place after array creation (and not with the
        # aranage() its dtype argument) or it will cast the `block_size` step to an int
        # before the arange array creation
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
        # nout-2 buckets
        # .astype(np.int64)是floor操作
        # number of intervals = floor((stop-start)/step)=n-2+floor((n-2)/(N-2))=n-2!
        # 也就是说最后一个数一定是y.shape[0]-1
        # 1, 1+bs, 1+2*bs, ..., 1+(nout-2)*bs=y.shape[0]-1
        # 得到上述nout-2个等间隔（浮点数bs间隔）之后，最后才astype(np.int64)，把中间的浮点数floor成整数
        # 这样虽然最后得到的不是严格的等点数buckets可能稍微加减1个点之类的，但是buckets个数是不变的，buckets的连贯性也是不变的

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 2):
            a = (
                LTOBDownsampler._argmax_area(
                    x_bucket=x[offset[i]-1 : offset[i + 1]+1].copy(),
                    y_bucket=y[offset[i]-1 : offset[i + 1]+1].copy(),
                    # offset starts from 1, ends at y.shape[0]-1
                    # so offset[i]-1 >= 0, offset[i + 1]+1<=y.shape[0]
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        return sampled_x

class LTDOBDownsampler(AbstractDownsampler):
    # LTOB based on dynamic buckets
    @staticmethod
    def _argmax_area(x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        x_bucket : np.ndarray
            All x values in the bucket, 
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.
        y_bucket : np.ndarray
            All y values in the bucket,
            padded with the last point of the previous bucket and the first point of the afterward bucket,
            so that the first and last points of the current bucket can calculate their triangle area.

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """

        # the following vectorized triangular area argmax computation is adapted from 
        # https://github.com/Permafacture/Py-Visvalingam-Whyatt/blob/master/polysimplify.py

        arr=np.column_stack((x_bucket, y_bucket))
        result = np.empty((len(arr)-2,),arr.dtype) 
        # calculate the triangle area for the points of the current bucket
        # except the padded first and last points borrowed from the prior or afterward buckets
        p1 = arr[:-2]
        p2 = arr[1:-1]
        p3 = arr[2:]
        #an accumulators to avoid unnecessary intermediate arrays
        accr = result #Accumulate directly into result
        acc1 = np.empty_like(accr)
        np.subtract(p2[:,1], p3[:,1], out = accr) # v2-v3
        np.multiply(p1[:,0], accr,    out = accr) # t1*(v2-v3)
        np.subtract(p3[:,1], p1[:,1], out = acc1  ) # v3-v1
        np.multiply(p2[:,0], acc1,    out = acc1  ) # t2*(v3-v1)
        np.add(acc1, accr,            out = accr) # t1*(v2-v3)+t2*(v3-v1)
        np.subtract(p1[:,1], p2[:,1], out = acc1  ) # v1-v2
        np.multiply(p3[:,0], acc1,    out = acc1  ) # t3*(v1-v2)
        np.add(acc1, accr,            out = accr) # t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)
        np.abs(accr, out = accr) # |t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|
        accr /= 2. # 1/2*|t3*(v1-v2) + t1*(v2-v3)+t2*(v3-v1)|

        return result.argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造动态分桶
        offset = _get_ltd_bin_idxs(x,y,n_out)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 2):
            a = (
                LTDOBDownsampler._argmax_area(
                    x_bucket=x[offset[i]-1 : offset[i + 1]+1].copy(),
                    y_bucket=y[offset[i]-1 : offset[i + 1]+1].copy(),
                    # offset starts from 1, ends at y.shape[0]-1
                    # so offset[i]-1 >= 0, offset[i + 1]+1<=y.shape[0]
                    # and x[] is left included and right excluded, so correct.
                    # -1 +1 for padding as the triangle areas of the first and last points of the current bucket needs calculation
                    # otherwise might result in points that has the max triangle area but at the border of the bucket not selected
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        return sampled_x

class LTSDownsampler(AbstractDownsampler):
    # optimal
    
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out == 6, "n_out must be 6, for toy example of LTS"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        assert len(x)<100, "len(x) must be smaller than 100, for toy example of LTS"

        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        # print('LTTBETFurther bins=',bins)

        # 因为合并了非空桶，所以现在非空桶数不一定是nout-2，而是要现算
        # len(bins)-1个分桶，len(bins)-1<=n_out-2 
        nbins=len(bins)-1

        assert nbins == 4, "nbins must be 4, for toy example of LTS" # 排除有空桶的情况

        # Construct the output list
        optimal_x = np.empty(nbins+2, dtype="int64") # +2是因为全局首尾点
        largest_area = -1

        sampled_x = np.empty(nbins+2, dtype="int64") # +2是因为全局首尾点
        sampled_x[0] = 0 # 第一个点
        sampled_x[-1] = x.shape[0] - 1 # 最后一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        # 四个非空桶
        for i1 in np.arange(bins[0],bins[1]):
            sampled_x[1]=i1
            for i2 in np.arange(bins[1],bins[2]):
                sampled_x[2]=i2
                for i3 in np.arange(bins[2],bins[3]):
                    sampled_x[3]=i3
                    for i4 in np.arange(bins[3],bins[4]):
                        sampled_x[4]=i4
                        area=_effective_area(x,y,sampled_x)
                        if area > largest_area:
                            largest_area = area
                            optimal_x = np.array(sampled_x) # NOTE: use deep copy!!!!

        print('LTSDownsampler max area',largest_area)
        print(optimal_x)
        return optimal_x

class StepLTTBETDownsampler_deprecated(AbstractDownsampler):
    # 和LTTBETDownsampler的区别：这里专门接受t2参数，而不是使用实际数据的第二个点的时间戳，为了和MinMax preselection使用的时间分桶对齐
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        # print(np.abs(
        #     x_bucket * (prev_y - avg_next_y)
        #     + y_bucket * (avg_next_x - prev_x)
        #     + (prev_x * avg_next_y - avg_next_x * prev_y)
        # )/2)
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_stepLTTBET(x, 100, 2100, n_out-2)
        # print('LTTBETDownsampler bins=',bins)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            # print(x[bins[i] : bins[i + 1]])
            # len(bins)-1个分桶，len(bins)-1<=n_out-2 
            # len(bins)-2是因为最后一个分桶特殊处理不在这里处理
            a = ( # 括号很重要不能少！！
                LTTBETDownsampler._argmax_area(
                        prev_x=x[a],
                        prev_y=y[a],

                        # 这里不用担心桶里没有点，因为提前合并过空桶了
                        avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                        avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            ) # 括号很重要不能少！！
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        # print(x[bins[-2] : bins[-1]])
        sampled_x.append(
            LTTBETDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1) # 最后一个点
        return np.unique(sampled_x)

class LTTBETDownsampler(AbstractDownsampler):
    # 和LTTB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTTB的区别：LTTB使用等点数分桶，LTTBET使用等时间分桶
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        # print(np.abs(
        #     x_bucket * (prev_y - avg_next_y)
        #     + y_bucket * (avg_next_x - prev_x)
        #     + (prev_x * avg_next_y - avg_next_x * prev_y)
        # )/2)
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        # print('LTTBETDownsampler bins=',bins)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            # print(x[bins[i] : bins[i + 1]])
            # len(bins)-1个分桶，len(bins)-1<=n_out-2 
            # len(bins)-2是因为最后一个分桶特殊处理不在这里处理
            a = ( # 括号很重要不能少！！
                LTTBETDownsampler._argmax_area(
                        prev_x=x[a],
                        prev_y=y[a],

                        # 这里不用担心桶里没有点，因为提前合并过空桶了
                        avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                        avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            ) # 括号很重要不能少！！
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        # print(x[bins[-2] : bins[-1]])
        sampled_x.append(
            LTTBETDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1) # 最后一个点

        print('LTTBETDownsampler area=', _effective_area(x,y,np.unique(sampled_x)))
        return np.unique(sampled_x)

class LTTBETFurtherDownsampler(AbstractDownsampler):
    # 和LTTB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTTB的区别：LTTB使用等点数分桶，LTTBET使用等时间分桶
    # 和LTTBET的区别：这里得到第一轮采样点之后，继续进行第二轮，第二轮使用后一个桶的上一轮的采样点而不是“后一个桶的平均点”
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        # print('LTTBETFurther bins=',bins)

        # 因为合并了非空桶，所以现在非空桶数不一定是nout-2，而是要现算
        # len(bins)-1个分桶，len(bins)-1<=n_out-2 
        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64") # +2是因为全局首尾点
        sampled_x[0] = 0 # 第一个点
        sampled_x[-1] = x.shape[0] - 1 # 最后一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        numIterations=8 # 实验里基本4次迭代以内就不会再提高effective area，达到收敛了
        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0 # 每次迭代开始这个初始化选中首点不能少，因为在每次迭代中a会被改变
            if num==0: # 第一次迭代，等价于原版LTTB，用的是“后一个桶的平均点”
                for i in range(nbins-1): 
                    # nbins-1是因为最后一个分桶特殊处理不在这里处理
                    a = ( # 括号很重要不能少！！
                        LTTBETDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # 这里不用担心桶里没有点，因为提前合并过空桶了
                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    ) # 括号很重要不能少！！
                    sampled_x[i+1] = a # 注意+1

            else: # 后续迭代，用上一轮的采样点取代“后一个桶的平均点”
                for i in range(nbins-1): 
                    # nbins-1而不是nbins是因为最后一个分桶特殊处理不在这里处理
                    # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                    a = ( # 括号很重要不能少！！
                        LTTBETDownsampler._argmax_area(
                                prev_x=x[a],
                                prev_y=y[a],

                                # 用后一个桶的上一轮的采样点而不是“后一个桶的平均点”，但是就不改名了
                                # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                                # 所以当前桶的后一个桶的采样点下标是sampled_x[i+2]
                                avg_next_x=x[sampled_x[i+2]],
                                avg_next_y=y[sampled_x[i+2]],

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    ) # 括号很重要不能少！！

                    # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                    sampled_x[i+1] = a # 注意+1

            # ------------ EDGE CASE ------------
            # next-average of last bucket = last point
            sampled_x[-2]=(
                LTTBETDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=x[-1],  # last point
                    avg_next_y=y[-1],  # last point

                    x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                    y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                )
                + bins[-2]
            )

            # 打印这一轮迭代的采样结果及其effective area
            # 第一轮num=0的采样结果和LTTBETDownsampler是一样的
            # print('LTTBETFurtherDownsampler sampling result of each iteration',num+1,sampled_x)
            # print(_effective_area(x,y,sampled_x))
            areas[num]=_effective_area(x,y,sampled_x)

        print('LTTBETFurtherDownsampler effective area of all iterations',areas)
        return sampled_x


class ILTSParallelDownsampler(AbstractDownsampler):
    # 和LTTBETFurther的区别：这里不依赖前一个桶的当前迭代最新采样点，从而一轮迭代内部各个桶之间是没有前后依赖的
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        print('ILTSParallel bins=',bins)

        # 因为合并了非空桶，所以现在非空桶数不一定是nout-2，而是要现算
        # len(bins)-1个分桶，len(bins)-1<=n_out-2 
        nbins=len(bins)-1

        # Construct the output list
        sampled_x = np.empty(nbins+2, dtype="int64") # +2是因为全局首尾点
        sampled_x[0] = 0 # 第一个点
        sampled_x[-1] = x.shape[0] - 1 # 最后一个点

        # Construct the result of last round
        lastIter_sampled_x = np.empty(nbins+2, dtype="int64") # +2是因为全局首尾点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        numIterations=8 # 实验里基本4次迭代以内就不会再提高effective area，达到收敛了
        areas=np.zeros(numIterations)
        for num in range(numIterations):
            a = 0
            if num==0: # 第一次迭代，左右固定点用的是初始化桶的平均点
                # 第一轮的第一个桶的l固定点用全局首点而不是桶平均点，因为这里桶特指B1到Bk
                sampled_x[1]=(
                    ILTSParallelDownsampler._argmax_area(
                        prev_x=x[0],
                        prev_y=y[0],

                        # 这里不用担心桶里没有点，因为提前合并过空桶了
                        avg_next_x=np.mean(x[bins[1] : bins[2]]),
                        avg_next_y=y[bins[1] : bins[2]].mean(),

                        x_bucket=x[bins[0] : bins[1]].copy(),
                        y_bucket=y[bins[0] : bins[1]].copy(),
                    )
                    + bins[0]
                )
                for i in range(1,nbins-1): 
                    # nbins-1是因为最后一个分桶特殊处理不在这里处理
                    # 1是因为第一轮的第一个桶l固定点用全局首点而不是桶平均点
                    a = ( # 括号很重要不能少！！
                        ILTSParallelDownsampler._argmax_area(
                                # 这里不用前一个桶的当前采样点，而是用初始的前一个桶平均点
                                prev_x=np.mean(x[bins[i-1] : bins[i]]),
                                prev_y=y[bins[i-1] : bins[i]].mean(),

                                # 这里不用担心桶里没有点，因为提前合并过空桶了
                                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]]),
                                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean(),

                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    ) # 括号很重要不能少！！
                    sampled_x[i+1] = a # 注意+1

                # next-average of last bucket = last point 
                # 倒数第二个采样点
                sampled_x[-2]=( 
                    ILTSParallelDownsampler._argmax_area(
                        prev_x=np.mean(x[bins[-3] : bins[-2]]), # 倒数第二个桶的平均点
                        prev_y=y[bins[-3] : bins[-2]].mean(), # 倒数第二个桶的平均点

                        avg_next_x=x[-1],  # last point 全局尾点
                        avg_next_y=y[-1],  # last point 全局尾点

                        x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                        y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                    )
                    + bins[-2]
                )

            else: # 后续迭代，用上一轮的采样点取代“后一个桶的平均点”
                for i in range(nbins): 
                    # 最后一个分桶一样处理，所以涵盖在里面了
                    # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                    a = ( # 括号很重要不能少！！
                        ILTSParallelDownsampler._argmax_area(
                                # 这里用上一轮迭代而不是当前迭代的左边桶的采样点
                                # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                                prev_x=x[lastIter_sampled_x[i]],
                                prev_y=y[lastIter_sampled_x[i]],

                                # 用后一个桶的上一轮的采样点而不是“后一个桶的平均点”，但是就不改名了
                                # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                                # 所以当前桶的后一个桶的采样点下标是sampled_x[i+2]
                                avg_next_x=x[lastIter_sampled_x[i+2]],
                                avg_next_y=y[lastIter_sampled_x[i+2]],

                                # bins的i不用+1
                                x_bucket=x[bins[i] : bins[i + 1]].copy(),
                                y_bucket=y[bins[i] : bins[i + 1]].copy(),
                            )
                        + bins[i]
                    ) # 括号很重要不能少！！

                    # 注意这里i从0开始，对应到sampled_x从1开始，因为sampled_x的0位置放的是全局首点
                    sampled_x[i+1] = a # 注意+1

            # # ------------ EDGE CASE ------------
            # # next-average of last bucket = last point 
            # # 倒数第二个采样点
            # sampled_x[-2]=( 
            #     ILTSParallelDownsampler._argmax_area(
            #         prev_x=x[lastIter_sampled_x[-3]], # 倒数第三个采样点
            #         prev_y=y[lastIter_sampled_x[-3]], # 倒数第三个采样点

            #         avg_next_x=x[-1],  # last point 全局尾点
            #         avg_next_y=y[-1],  # last point 全局尾点

            #         x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            #         y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            #     )
            #     + bins[-2]
            # )

            # 打印这一轮迭代的采样结果及其effective area
            # 第一轮num=0的采样结果和LTTBETDownsampler是一样的
            print('ILTSParallelDownsampler sampling result of each iteration',num+1,sampled_x)
            print(_effective_area(x,y,sampled_x))
            areas[num]=_effective_area(x,y,sampled_x)

            lastIter_sampled_x=np.array(sampled_x) # 注意一定要是deep copy

        print('ILTSParallelDownsampler effective area of all iterations',areas)
        return sampled_x

class LTTBETNewDownsampler(AbstractDownsampler):
    # 和LTTB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTTB的区别：LTTB使用等点数分桶，LTTBET使用等时间分桶
    # 和LTTBETDownsampler区别：这里三角形的底边是前两个采样点，而不是前后桶的点。
    # 初始两个点用第一个桶的最大值点和最小值点，后面找当前最新两个采样点形成三角形面积最大的点就可以
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于n_out-2个
        bins = _get_bin_idxs_nofirstlast(x, n_out-2)
        # print('LTTBETNewDownsampler bins=',bins)

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        # Construct the output list
        sampled_x = []
        # sampled_x.append(0) # 第一个点，取全局第一个点 TODO：改成取第一个桶的最小值点
        sampled_x.append(bins[0] + y[bins[0] : bins[1]].argmin()) # 第一个点，取来自第一个非空桶的最小值点。
        sampled_x.append(bins[0] + y[bins[0] : bins[1]].argmax()) # 第二个点，取来自第一个非空桶的最大值点。
        # 不用担心遇到空桶，因为这里提前合并过空桶了

        a = 0
        for i in range(1,len(bins)-1):# 从第二个桶开始，直到最后一个分桶（取到）
            # len(bins)-1个分桶，len(bins)-1<=n_out-2 
            a = ( # 括号很重要不能少！！
                LTTBETNewDownsampler._argmax_area(
                        prev_x=x[sampled_x[-2]],
                        prev_y=y[sampled_x[-2]],

                        # 这里其实不是后一个桶的平均点了，而是当前最后一个采样点，但是就不改名了
                        avg_next_x=x[sampled_x[-1]],
                        avg_next_y=y[sampled_x[-1]],

                        x_bucket=x[bins[i] : bins[i + 1]].copy(),
                        y_bucket=y[bins[i] : bins[i + 1]].copy(),
                    )
                + bins[i]
            ) # 括号很重要不能少！！
            sampled_x.append(a)

        sampled_x.append(x.shape[0] - 1) # 最后一个点
        return np.unique(sampled_x)

class LTTBETGapDownsampler(AbstractDownsampler):
    # 和LTTB的相同点：给定nout个点输出，都预留首尾点，从而中间分nout-2个桶
    # 和LTTB的区别：LTTB使用等点数分桶，LTTBETGap使用等时间分桶
    # 和LTTBET的区别：LTTBETGap采取空桶感知策略，遇到空桶的话就断开，以此保留空桶gap两端点
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的倒数第二个点的时间戳，把[tmin,tmax]均匀分成n_out-2份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后没有执行np.unique来合并空桶，所以bins形成的桶数就是n_out-2个，但是有可能有空桶
        bins = _get_bin_idxs_nofirstlast_gapAware(x, n_out-2)
        # print(bins)

        # Construct the output list
        sampled_x = []
        sampled_x.append(0) # 第一个点

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(len(bins)-2): 
            # len(bins)-1个分桶，len(bins)-1=n_out-2 
            # len(bins)-2是因为最后一个分桶特殊处理不在这里处理

            if bins[i]==bins[i + 1]: # 当前桶是空的
                # 以gap之后的第一个点作为新的全局第一个点
                # 相当于和前面的断开，从新开始做LTTB采样
                # 虽然现在这个点在后面被用到三角形计算里了，但是应该算出来面积是0所以不会干扰
                # 计算上有重叠，但是角色还是和原来一样的
                a=bins[i]
                sampled_x.append(a) # gap之后第一个点，最后会去重排序的所以没关系
                continue

            # 现在当前桶非空，判断下一个桶是否空
            if bins[i + 1]==bins[i + 2]: # 下一个桶是空的
                # 以当前桶的最后一个点作为下一个桶的平均点
                avg_next_x=x[bins[i + 1]-1]
                avg_next_y=y[bins[i + 1]-1]
                sampled_x.append(bins[i + 1]-1) # gap之前的最后一个点，最后会去重排序的所以没关系
            else: # 正常计算下一个桶的平均点
                avg_next_x=np.mean(x[bins[i + 1] : bins[i + 2]])
                avg_next_y=y[bins[i + 1] : bins[i + 2]].mean()


            a = (
                LTTBETGapDownsampler._argmax_area(
                    # 虽然现在这个点有可能就是a=bins[i]当前桶的第一个点也就是gap之后的第一个点，但是应该算出来面积是0所以不会干扰
                    prev_x=x[a],
                    prev_y=y[a],

                    # 虽然现在这个点有可能就是bins[i+1]-1当前桶的最后一个点也就是gap之前的最后一个点，但是应该算出来面积是0所以不会干扰
                    avg_next_x=avg_next_x,
                    avg_next_y=avg_next_y,

                    x_bucket=x[bins[i] : bins[i + 1]].copy(),
                    y_bucket=y[bins[i] : bins[i + 1]].copy(),
                ) 
                + bins[i]
            )
            sampled_x.append(a)

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        # 第一个桶和最后一个桶总是非空的，因为至少第一个桶里有第二个点，最后一个桶里有倒数第二个点
        # （不考虑整条时间序列点数小于4的情况）
        # 比如从第二个点到倒数第二个点之间都没有点，于是桶边界下标bins=1,len(x)-2,len(x)-2,...,len(x)-2,len(x)-1
        # 第一个和最后一个桶里都各自有1个点，其余桶里都是空的
        # 所以最多这里a=len(x)-2，不会取到len(x)-1的
        sampled_x.append(
            LTTBETGapDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                y_bucket=y[bins[-2] : bins[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            )
            + bins[-2]
        )

        sampled_x.append(x.shape[0] - 1) # 最后一个点
        return np.unique(sampled_x) # unique既会去重还会排序

class LTTBDownsampler(AbstractDownsampler):
    # 注意这个是等点数间隔分桶
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造（基本）等点数分桶
        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2) # 是浮点数
        # Note this 'astype' cast must take place after array creation (and not with the
        # aranage() its dtype argument) or it will cast the `block_size` step to an int
        # before the arange array creation
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)
        # Values are generated within the half-open interval [start, stop), with spacing between values given by step.
        # .astype(np.int64)是floor操作
        # number of intervals = floor((stop-start)/step)=n-2+floor((n-2)/(N-2))=n-2!
        # 也就是说最后一个数一定是y.shape[0]-1
        # 1, 1+bs, 1+2*bs, ..., 1+(nout-2)*bs=y.shape[0]-1
        # 得到上述nout-2个等间隔（浮点数bs间隔）之后，最后才astype(np.int64)，把中间的浮点数floor成整数
        # 这样虽然最后得到的不是严格的等点数buckets可能稍微加减1个点之类的，但是buckets个数是不变的，buckets的连贯性也是不变的
        # 注意offset本身是nout-1个数字，形成了nout-2个intervals

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3):
            a = (
                LTTBDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),

                    x_bucket=x[offset[i] : offset[i + 1]].copy(),
                    y_bucket=y[offset[i] : offset[i + 1]].copy(),
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            LTTBDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[offset[-2] : offset[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                y_bucket=y[offset[-2] : offset[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            )
            + offset[-2]
        )
        return sampled_x

class LTDDownsampler(AbstractDownsampler):
    # 注意这个是等点数间隔分桶
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 构造动态分桶
        offset = _get_ltd_bin_idxs(x,y,n_out)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3): # nout-4,nout-3;nout-3,nout-2
            a = (
                LTDDownsampler._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],

                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),

                    x_bucket=x[offset[i] : offset[i + 1]].copy(),
                    y_bucket=y[offset[i] : offset[i + 1]].copy(),
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            LTDDownsampler._argmax_area(
                prev_x=x[a],
                prev_y=y[a],

                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],  # last point

                x_bucket=x[offset[-2] : offset[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
                y_bucket=y[offset[-2] : offset[-1]].copy(), # 但是这样是不是永远把y.shape[0]-1位置的点排除不考虑了？虽然可能问题不大
            )
            + offset[-2]
        )
        return sampled_x

class MinMaxDownsampler(AbstractDownsampler):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.
    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 2)
        # print('MinMaxDownsampler bins=',bins)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        return np.unique(rel_idxs)

class MinMaxFPLPDownsampler(AbstractDownsampler):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.
    """
    # MinMaxFPLPDownsampler是预留了首尾点，然后把[t2,tn)等分成(Nout-2)/2个桶，每个桶里采集MinMax点
    # 而MinMaxDownsampler是把[t1,tn]等分成nout/2个桶，前面左闭右开，最后一个桶左闭右闭，每个桶里采集MinMax点

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert (n_out-2) % 2 == 0, "n_out-2 must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        # bins = _get_bin_idxs(x, n_out // 2)

        # 构造等时间分桶，注意n_out-2，因为首尾点预留在采样输出结果里
        # tmin是原序列的第二个点的时间戳，tmax是原序列的最后一个点的时间戳，把[tmin,tmax]均匀分成int((n_out-2)/2)份，前面都是左闭右开，最后一个左闭右闭。
        # 注意由于这个函数最后执行了np.unique来合并空桶，所以bins形成的桶数是小于等于int((n_out-2)/2)个
        bins = _get_bin_idxs_nofirstlast(x, int((n_out-2)/2))
        # print('MinMaxFPLPDownsampler bins=',bins)

        # # 因为合并了非空桶，所以现在非空桶数不一定是nout-2，而是要现算
        # # len(bins)-1个分桶，len(bins)-1<=n_out-2 
        # nbins=len(bins)-1

        rel_idxs = []
        rel_idxs.append(0) # 预留首尾点
        rel_idxs.append(x.shape[0] - 1) # 预留首尾点

        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        print('MinMaxFPLPDownsampler area=', _effective_area(x,y,np.unique(rel_idxs)))
        return np.unique(rel_idxs)


class MinMaxGapDownsampler(AbstractDownsampler):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.
    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs_gapAware(x, n_out // 2) # 有可能有空桶

        rel_idxs = []
        for i in range(len(bins)-1): # len(bins)-1个桶
            # 判断当前是否空桶
            if bins[i]==bins[i+1]:
                continue

            # 判断前一个桶是否空桶
            if i>0 and bins[i-1]==bins[i]:
                # 保留当前桶的第一个点
                rel_idxs.append(bins[i])

            # 判断下一个桶是否空桶
            if i<len(bins)-2 and bins[i+1]==bins[i+2]:
                # 保留当前桶的最后一个点
                rel_idxs.append(bins[i+1]-1)

            # 处理当前桶，非空
            y_slice = y[bins[i]:bins[i+1]]
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(bins[i] + y_slice.argmin())
            rel_idxs.append(bins[i] + y_slice.argmax())

        # Returns the sorted unique elements of an array.
        return np.unique(rel_idxs) # 会去重和排序

class M4Downsampler(AbstractDownsampler):
    """Aggregation method which selects the 4 M-s, i.e y-argmin, y-argmax, x-argmin, and
    x-argmax per bin.

    .. note::
        When `n_out` is 4 * the canvas its pixel widht it should create a pixel-perfect
        visualization w.r.t. the raw data.

    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 4 == 0, "n_out must be a multiple of 4"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 4)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue

            # calculate the min(idx), argmin(slice), argmax(slice), max(idx)
            rel_idxs.append(lower)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())
            rel_idxs.append(upper - 1)

        # NOTE: we do not use the np.unique so that all indices are retained
        # return np.array(sorted(rel_idxs))
        # Returns the sorted unique elements of an array.
        print('M4Downsampler area=', _effective_area(x,y,np.unique(rel_idxs)))
        return np.unique(rel_idxs)

class EveryNthDownsampler(AbstractDownsampler):
    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **_
    ) -> np.ndarray:
        if x is not None:
            name = self.__class__.__name__
            warnings.warn(
                f"x is passed to downsample method of {name}, but is not taken "
                "into account by the current implementation of the EveryNth algorithm."
            )
        step = max(1, len(y) / n_out)
        return np.arange(start=0, stop=len(y) - 0.1, step=step).astype(np.uint)
