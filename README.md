# UMatrix
    . 这是进步最大的一次完善和更新，部分函数速度与Matrix版本相比速度提升了几万倍。虽然部分函数在类似10x10矩阵时
        表现可能有点弱势，但是当矩阵变大时UMatrix将击败Matrix版本。下面是他们的速度比较，仅在10x10的矩阵上。
   ![Image](https://github.com/baozhixue/pMatrix/blob/master/10x10VS.png)
    
    . 下面是UMatrix版本在10x10至1000x1000矩阵的函数速度的展示。在本次更新中，QR()函数提升最高，在1000x1000矩阵
        下计算速度提示将近18倍。
   ![Image](https://github.com/baozhixue/pMatrix/blob/master/mat.png.jpg)
    
    . 在此次更新中，删除了SVD(...)和Jacobi(...)函数，因为我发现这两个函数的计算结果有一些问题。
    . UMatrix可以在windows和ubuntu下使用，因为在ubuntu下第一个版本因为使用AVX指令集总是报错所以此版本删除了所有
        AVX指令集的内容。
    . 针对删除AVX指令集在UMatrix内替换使用了omp头文件。所以在ubuntu下编译时建议使用
            $：g++ ***.cpp UMatrix.h -fopenmp -O2
    . 速度虽然与python或Matlab矩阵计算速度依旧相差很大，但是。。。。。。。。。。。。。
    
    
# Matrix [舍弃版本]
    . 这是一个简单的矩阵计算库，目前支持了矩阵的加法，减法，乘法（点乘，叉乘）等。
    . SVD(以雅克比迭代法计算，目前仅支持方阵，并且对复数不支持）。
    . 幂法求解方阵的特征值和特征向量。
    . 若你使用该库，希望你不要在你的程序内部进行任何以指针形式的显式声明，否则可能会造成很严重的内存泄漏问题。
        针对此问题，可以详细阅读topic部分。
    . 目前整个版本是基于windows的，所以在ubuntu或其他系统上面是否存在问题我不能进行任何的保证。
    . 同时也希望你为该库提出意见。谢谢.

# 更新日历
    # 2020/2/10
        (1)依靠我边看边写的渣渣技术，使我的矩阵计算库可以进行了网页访问。如果你准备进行一些尝试，
            可以使用MyWeb头文件内的run_server(),并且在本地电脑浏览器输入127.0.0.1即可。
            同时我还通过属于我的域名baozhixue.com使用我的电脑作为服务器进行了访问，感觉还是很舒服的。
            虽然界面真的很炸。🙂
        (2)对Matrix内的部分函数进行了修改。
    # 2020/2/9
        (1)QR分解进行了一些优化，使计算速度平均有8%左右的提升。
        (2)随机数生成矩阵进行了修复。
        (3)对部分例如控制矩阵计算精度，计算循环数量统一定义在库文件开始处。
    # 2020/2/8
        (1)更新添加支持QR分解，范数2,LU分解，PLU分解的计算。
        (2)添加了‘()’操作符的支持。
            该操作符可以截取矩阵的某一块。
        (3)part_set(Mat,r,c,r2,c2)函数可以将矩阵的某一块根据Mat和给定的位置进行更新。
            很遗憾此操作暂时不能通过操作符'()'进行控制，虽然这样显得更加自然。

# topic
    [1]当你准备使用指针传递一个矩阵时：
      Matrix A,B;
      A = &(A+B);
      A = &(eye(3,3));
      而无需将A声明为一个指针。
    [2]当你使用指针指向一个矩阵时，若希望两者同时更新位于内存中的内容，则应该如下所示
      Matrix A = eye(5);
      Matrix B;
      B = &A;       //此时B为A的指针。
      B = eye(3);   //此时A和B皆为3x3单位对角矩阵。
      B = &eye(10); // 此时B指向一个新的矩阵，而A保持不变；
      A = &eye(11); // 之前的3x3矩阵被释放，A为新的矩阵。
     [3]显式的释放“虚拟”指针
        Matrix A = eye(5);   // 此时A即为一个指针
        A.clear();           // 自动释放A.Mat所占内存；若该内存位置被多次引用，则递减计数器。

# 目前存在的问题：
    [1]pMatrix版本存在内存泄漏问题，虽然因此他的速度很快。
    [2]正如1所述，建议使用Matrix，在后续也只将更新Matrix。其中matrix_copy为Matrix的最初版本。

# Basic
    [1]若你准备正常的使用此头文件，应确定你的设备支持AVX，因为部分函数使用了该技术。
    [2]你需要确定你的编译器支持C++ 11或者其之后的任何一个版本，在库内使用了许多之前C++标准所未拥有的特性。
