# U4
    . 更新于2020年3月7日
    . U4版本通过使用Intel AVX指令集和通过对计算时内存优化计算速度有了进一步的提升。
    . U4版本使用C++模板进行编写，但很可惜目前仅支持float和double两种类型，若使用int则会遇到很糟糕因为舍入误差
         而造成计算结果错位。
    . 相较于UMatrix版本减少了部分运算符的支持，毕竟这是一个花费将近两天的快速版。。。。。。
    . 与UMatrix版本相比推荐使用引用而非指针，同时删除了内存管理函数。
    . 与UMatrix版本相比，U4版本直接申请一段连续的内存，而非之前的根据矩阵行数申请一致的内存块。
    
   ![image](https://github.com/baozhixue/pMatrix/blob/master/pic/U4.png)
    
# UMatrix
    . 这是进步最大的一次完善和更新，部分函数速度与Matrix版本相比速度提升很大。下方为在10x10至1000x1000时的部分函
        数比较．其中Inverse,AdjointMatrix(...)等其余几个函数未在此处列出，原因是他们的速度可能需要等到我的电脑
        报废才会出结果．
    
    . 在此次更新中，删除了SVD(...)和Jacobi(...)函数，因为我发现这两个函数的计算结果有一些问题。
    . UMatrix可以在windows和ubuntu下使用，因为在ubuntu下第一个版本因为使用AVX指令集总是报错所以此版本删除了所有
        AVX指令集的内容。
    . 针对删除AVX指令集在UMatrix内替换使用了omp头文件。所以在ubuntu下编译时建议使用
            $：g++ ***.cpp UMatrix.h -fopenmp -O2
    . 速度虽然与python或Matlab矩阵计算速度依旧相差很大，但是。。。。。。。。。。。。。
   
   ![image](https://github.com/baozhixue/pMatrix/blob/master/pic/UMatrix.jpg)
 
 
# Matrix [舍弃版本]
    . 这是一个简单的矩阵计算库，目前支持了矩阵的加法，减法，乘法（点乘，叉乘）等。
    . SVD(以雅克比迭代法计算，目前仅支持方阵，并且对复数不支持）。
    . 幂法求解方阵的特征值和特征向量。
    . 若你使用该库，希望你不要在你的程序内部进行任何以指针形式的显式声明，否则可能会造成很严重的内存泄漏问题。
        针对此问题，可以详细阅读topic部分。
    . 目前整个版本是基于windows的，所以在ubuntu或其他系统上面是否存在问题我不能进行任何的保证。
    . 同时也希望你为该库提出意见。谢谢.
