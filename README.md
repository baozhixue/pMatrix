# Matrix
    . 这是一个简单的矩阵计算库，目前支持了矩阵的加法，减法，乘法（点乘，叉乘）等。
    . SVD(以雅克比迭代法计算，目前仅支持方阵，并且对复数不支持）。
    . 幂法求解方阵的特征值和特征向量。
    . 同时也希望你为该库提出意见。谢谢.

# 更新日历
    # 2020/2/8
        (1)更新添加支持QR分解，范数2的计算。
        (2)添加了‘()’操作符的支持。
            该操作符可以截取矩阵的某一块。
        (3)part_set(Mat,r,c,r2,c2)函数可以将矩阵的某一块根据Mat和给定的位置进行更新。
            很遗憾此操作暂时不能通过操作符'()'进行控制，虽然这样显得更加自然。

# 目前存在的问题：
    [1]pMatrix版本存在内存泄漏问题，虽然因此他的速度很快。
    [2]正如1所述，建议使用Matrix，在后续也只将更新Matrix。其中matrix_copy为Matrix的最初版本。

# Basic
    [1]若你准备正常的使用此头文件，应确定你的设备支持AVX，因为部分函数使用了该技术。
    [2]你需要确定你的编译器支持C++ 11或者其之后的任何一个版本，在库内使用了许多之前C++标准所未拥有的特性。

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
