/*
    created by baozhixue
    email: baozhixue@hnu.edu.cn
*/

#ifndef LALIB_MATRIX_H
#define LALIB_MATRIX_H

#include <iostream>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <random>
#include <ctime>
#include <tuple>
#include <string>

namespace bzx {

    constexpr double PI = 3.1415926;
    constexpr double DMIN = std::numeric_limits<double>::min();  // double min
    constexpr double DMAX = std::numeric_limits<double>::max();
    constexpr double PRECISION = 1e-10;   // JACOBI
    constexpr double DETERMINANT_PRECISION = 1e-8;
    constexpr double MIN_DELTA = 1e-5;  // Power_Method
    constexpr size_t MAX_ITER = int(1e4);   // JACOBI,Power_Method
    constexpr double OUT_PRECISION = 1e-8;  // 输出时当绝对值小于此值，则输出为0；

    class Matrix;
    // 记录当前矩阵描述信息
    class Matrixdsec {
    public:
        Matrixdsec(size_t rs, size_t cs, size_t uc) {
            row_size = rs;
            col_size = cs;
            use_counter = uc;
        }
        size_t row_size = 0;
        size_t col_size = 0;
        size_t use_counter = 0;   //当作为指针传递时，记录被引用数量
        std::vector<Matrix*> memAsyc;  //全局矩阵的更新
    };  

    Matrix operator+(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator-(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator/(const Matrix& Mat, const double& num);
    Matrix operator==(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator!=(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator>(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator<(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator>=(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator<=(const Matrix& Mat, const Matrix& Mat2);
    std::ostream& operator<<(std::ostream& out, const Matrix& m);
    Matrix operator*(const Matrix& Mat, const Matrix& Mat2);   // 叉乘

    Matrix SQRT(const Matrix& Mat);  
    double MAX(const Matrix& Mat);
    double MIN(const Matrix& Mat);
    Matrix DOT(const Matrix& Mat, const Matrix& Mat2);
    Matrix DOT(const Matrix& Mat, const double& num);
    Matrix INVERSE(const Matrix& Mat);
    Matrix ADJOINT_Matrix(const Matrix& Mat);
    Matrix DIAGONAL(const Matrix& Mat);
    Matrix DIAGONAL(const size_t& _size, std::initializer_list<double> src);
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, std::initializer_list<double> src);
    Matrix DIAGONAL(const size_t& _size, const Matrix& Mat);
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, const Matrix& Mat);
    Matrix EYE(const size_t& rs, const size_t& cs);
    Matrix EYE(const size_t& _size);
    Matrix RandI_Matrix(const size_t& rs, const size_t& cs, const int& low, const int& high);
    Matrix RandD_Matrix(const size_t& rs, const size_t& cs, const double& low, const double& high);
    Matrix RandN_Matrix(const size_t& rs, const size_t& cs, const double& M=0.0, const double& S2=1.0);
    double DETERMINANT(const Matrix& Mat);
    std::tuple<size_t, Matrix> Rank(const Matrix& Mat);
    Matrix TRANSPOSITION(const Matrix& Mat);
    std::tuple<Matrix, Matrix, Matrix> SVD(const Matrix& Mat);
    std::tuple<Matrix, Matrix> JACOBI(const Matrix& Mat);
    std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta = MIN_DELTA, const size_t& max_iter= MAX_ITER);
    std::tuple<Matrix, Matrix> QR(const Matrix& Mat);
    double Norm_2(const Matrix& Mat);
    std::tuple<Matrix, Matrix, Matrix> PLU(const Matrix& Mat);
    std::tuple<Matrix, Matrix> LU(const Matrix& Mat);
    Matrix Ceil(const Matrix& Mat);
    Matrix Floor(const Matrix& Mat);
    Matrix MMul(std::initializer_list <Matrix> Mats);
    // 功能函数
    bool fast_copy(Matrix& dst, const Matrix& src);
    void row_swap_PLU(Matrix& Mat, size_t i, size_t ii, size_t col_index, bool Left = true);
    void row_swap(Matrix& Mat, size_t i, size_t ii);

    class Matrix{
        // double version
        friend std::ostream& operator<<(std::ostream& out, const Matrix& m);
    public:
        Matrix(const size_t& _row_size, const size_t& _col_size, const double &init_num=0.0);
        Matrix(std::initializer_list<std::initializer_list<double >> src);
        Matrix() = default;// { this->MDesc = new Matrixdsec(0, 0, 1); this->MDesc->memAsyc.push_back(this); };
        Matrix(const Matrix& Mat);
        Matrix& operator=(const Matrix& Mat);
        Matrix& operator=(Matrix* Mat);
        void operator+=(const Matrix& Mat);
        void operator-=(const Matrix& Mat);
        void operator*=(const Matrix& Mat);
        double* operator[](const size_t& index);
        double* operator[](const size_t& index) const;
        Matrix operator()(const size_t& Low_r, const size_t& High_r,const size_t& Low_c, const size_t& High_c) const;
        Matrix operator()(const size_t& r, const size_t& c) const;
        void clear();
        std::string to_str() const;
        bool memAsycEqual(const Matrix& Mat);
        int use_count()const;
        bool part_set(const Matrix& Mat, const size_t _Low_r, const size_t& _High_r, const size_t& _Low_c, const size_t& _High_c);
        Matrix& TRANS();
        std::tuple<size_t, size_t> shape() const;
        void CEIL() { *this = &Ceil(*this); };
        void FLOOR() { *this = &Floor(*this); };
        ~Matrix();
    private:
        double** Mat = NULL; // 2d
        Matrixdsec* MDesc = NULL;

    };


    /*
        返回矩阵的行和列
    */
    std::tuple<size_t, size_t> Matrix::shape() const
    {
        if (this->MDesc == NULL) {
            return std::make_tuple(0, 0);
        }
        return std::make_tuple(this->MDesc->row_size, this->MDesc->col_size);
        
    }

    Matrix::~Matrix() {
        clear();
    }
    

    /*
        返回矩阵内存被使用数量,
    */
    int Matrix::use_count()const {
        if (this->MDesc == NULL) {
            return -1;
        }
        
        return this->MDesc->use_counter;
    }


    /*
        若usecounter计数为0，则释放所属内存;否则，仅将指针置为NULL
    */
    void Matrix::clear() {
        //std::cout << "ready clear!\n";
        if (this->MDesc != NULL) {
            //std::cout << "waiting clear!\n";
            --MDesc->use_counter;
            if (MDesc->use_counter == 0) {
                //std::cout << "clear\n";
                for (size_t i = 0; i < MDesc->row_size; ++i) {
                    delete[]this->Mat[i];
                    this->Mat[i] = NULL;
                }
                delete[]this->Mat;
                delete this->MDesc;
            }
            for (size_t i = 0; i < MDesc->memAsyc.size(); ++i) {
                if (MDesc->memAsyc[i] == this) {
                    MDesc->memAsyc[i] = MDesc->memAsyc.back();
                    MDesc->memAsyc.pop_back();
                    break;
                }
            }
            this->Mat = NULL;
            this->MDesc = NULL;
        }
    }

    void Matrix::operator+=(const Matrix& Mat) {
        *this = &(*this + Mat);
    }
    void Matrix::operator-=(const Matrix& Mat) {
        *this = &(*this - Mat);
    }
    void Matrix::operator*=(const Matrix& Mat) {
        *this = &(*this * Mat);
    }

    /*
     * @ Purpose  ：拷贝赋值
     */
    Matrix& Matrix::operator=(const Matrix& Mat) {

        if (this->Mat == Mat.Mat) {
            return *this;
        }

        //重新指向新的内存
        if (this->MDesc==NULL || this->MDesc->use_counter <=1) {
            if (this->shape() == Mat.shape()) {
                fast_copy(*this, Mat);
                return *this;
            }
            clear();
            // 内存为NULL
            
            size_t _mSize_r, _mSize_c;
            std::tie(_mSize_r, _mSize_c) = Mat.shape();
            this->MDesc = new Matrixdsec(_mSize_r, _mSize_c, 1);

            size_t i = 0, j = 0;
            
            double** tmp2, * tmp;
            tmp2 = new double* [_mSize_r];
            if (tmp2 == NULL) {
                throw "failed to alloc new memory!\n";
            }

            for (i = 0; i < _mSize_r; ++i) {
                tmp = new double[_mSize_c];
                if (tmp == NULL) {
                    throw "failed to alloc new memory!\n";
                }

                for (j = 0; j + 4 <= _mSize_c; j += 4) {
                    _mm256_store_pd(tmp + j, _mm256_load_pd(Mat[i] + j));
                }
                while (j < _mSize_c) {
                    *(tmp + j) = Mat[i][j];
                    ++j;
                }
                tmp2[i] = tmp;
            }
            this->Mat = tmp2;
            this->MDesc->memAsyc.push_back(this);
        }
        else {
            // 更新所有指针所指的地址
            this->memAsycEqual(Mat);
        }
        
        return *this;
    }

    /*
     * @ Purpose  指针更新赋值
     */
    Matrix& Matrix::operator=(Matrix* Mat) {
        if (this->Mat == Mat->Mat) {
            return *this;
        }
        Mat->MDesc->use_counter += 1;
        clear(); // 更新usecounter
        this->Mat = Mat->Mat;
        this->MDesc = Mat->MDesc;
        this->MDesc->memAsyc.push_back(this);
       
        return *this;
    }


    /*
     * @ Purpose :计算Mat与Mat2的矩阵叉乘
     * @ Return : 结果矩阵  
     */
    Matrix operator*(const Matrix& Mat, const Matrix& Mat2) {

        size_t _mSize_r, _mSize_c;
        size_t _m2Size_r, _m2Size_c;

        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        std::tie(_m2Size_r, _m2Size_c) = Mat2.shape();

        if (_mSize_c != _m2Size_r) {
            std::cerr << "row_size must equal col_size in operator*!\n";
        }

        Matrix dst(_mSize_r, _m2Size_c, 0);
        // version 0.2
        size_t k = 0;
        __m256d dst_m256d = _mm256_setzero_pd(); // _mm256_set_pd(0, 0, 0, 0);
        __m256d Mat2_m256d;
        double dst_array[4];
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _m2Size_c; ++j) {
                for (k = 0; k + 4 <= _m2Size_r; k += 4) {
                    Mat2_m256d = _mm256_set_pd(Mat2[k + 3][j], Mat2[k + 2][j], Mat2[k + 1][j], Mat2[k][j]);
                    dst_m256d = _mm256_add_pd(dst_m256d, _mm256_mul_pd(_mm256_load_pd(Mat[i] + k), Mat2_m256d));
                }
                _mm256_store_pd(dst_array, dst_m256d);
                dst[i][j] = dst_array[0] + dst_array[1] + dst_array[2] + dst_array[3];
                while (k < _mSize_c) {
                    dst[i][j] += (Mat[i][k] * Mat2[k][j]);
                    ++k;
                }
                dst_m256d = _mm256_setzero_pd(); // _mm256_set_pd(0, 0, 0, 0);
            }
        }

        return dst;
    }

    /*
        @ Purpose : 计算矩阵与矩阵的点乘
        @ Return  ： 矩阵
        @ Other ：Mat2与Mat的行列数一致，或Mat2的行数与Mat行数一致且列数为1
    */
    Matrix DOT(const Matrix& Mat, const Matrix& Mat2) {
        size_t _mSize_r, _mSize_c;
        size_t _m2Size_r, _m2Size_c;
        
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        std::tie(_m2Size_r, _m2Size_c) = Mat2.shape();

        if (_m2Size_r != _mSize_r) {
            std::cerr << "Dot must Mat.row_size == Mat2.row_size";
            return Matrix();
        }

        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        if (_m2Size_c == _mSize_c) {
            for (i = 0; i < _mSize_r; ++i) {
                for (j = 0; j + 4 <= _mSize_c; j += 4) {
                    _mm256_store_pd(dst[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j)));
                }
                while (j < _mSize_c) {
                    dst[i][j] = Mat[i][j] * Mat2[i][j];
                    ++j;
                }
            }
        }
        else {
            if (_m2Size_c != 1) {
                std::cerr << "in Dot,if col_size is not equal,then second Mat.col_size need equal 1.\n";
                return Matrix();
            }
            for (i = 0; i < _mSize_r; ++i) {
                __m256d m256d = _mm256_set_pd(Mat2[i][0], Mat2[i][0], Mat2[i][0], Mat2[i][0]);
                for (j = 0; j + 4 <= _mSize_c; j += 4) {
                    _mm256_store_pd(dst[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat[i] + j), m256d));
                }
                while (j < _mSize_c) {
                    dst[i][j] = Mat[i][j] * Mat2[i][0];
                    ++j;
                }
            }
        }

        return dst;
    }

    /*
        @ Purpose : 计算矩阵与数值的点乘
        @ Return  ： 矩阵
    */
    Matrix DOT(const Matrix& Mat, const double& num) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat[i] + j), m256));
            }
            while (j < _mSize_c) {
                dst[i][j] = Mat[i][j] * num;
                ++j;
            }
        }
        return dst;
    }


    /*
        @ Purpose :矩阵除以一个数值
        @ Return :矩阵
    */
    Matrix operator/(const Matrix& Mat, const double& num) {
        if (num == 0) {
            std::cerr << "in div(/),every ele must not equal 0.\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst[i] + j, _mm256_div_pd(_mm256_load_pd(Mat[i] + j), m256));
            }
            while (j < _mSize_c) {
                dst[i][j] = Mat[i][j] / num;
                ++j;
            }
        }
        return dst;
    }

    /*
        @ Purpose :矩阵加法
    */
    Matrix operator+(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape())
        {
            std::cerr << "in add(+) two Mat's shape must equal!\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst[i] + j, _mm256_add_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j)));
            }
            while (j < _mSize_c) {
                dst[i][j] = Mat[i][j] + Mat2[i][j];
                ++j;
            }
        }
        return dst;
    }

    /*
        @ Purpose :矩阵减法
    */
    Matrix operator-(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape())
        {
            std::cerr << "in sub(-) two Mat's shape must equal!\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst[i] + j, _mm256_sub_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j)));
            }
            while (j < _mSize_c) {
                dst[i][j] = Mat[i][j] - Mat2[i][j];
                ++j;
            }
        }
        return dst;
    }

    /*
        @ Purpose :创建一个[_row_size,_col_size]的以init_num为初始值的矩阵
    */
    Matrix::Matrix(const size_t &_row_size, const size_t& _col_size, const double &init_num) {
        clear();
        this->MDesc = new Matrixdsec(_row_size, _col_size, 1);
        size_t  i = 0, j = 0;

        // 分配内存并初始化
        double** tmp2 = new double* [_row_size];
        double* tmp;
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->Mat = tmp2;

        __m256d m256 = _mm256_set_pd(init_num, init_num, init_num, init_num);
        for (i = 0; i < _row_size; ++i) {
            tmp = new double[_col_size];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }
            for (j = 0; j + 4 <= _col_size; j += 4) {
                _mm256_store_pd(tmp + j, m256);
            }
            while (j < _col_size) {
                *(tmp + j) = init_num;
                ++j;
            }
            this->Mat[i] = tmp;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
        @ Purpose : 根据src创建一个新的矩阵
    */
    Matrix::Matrix(std::initializer_list<std::initializer_list<double >> src) {

        if (src.size() == 0 || src.begin()->size() == 0) {
            std::cerr << "if you want to init a new Mat from such as {{1,2,3},{1,2,3}},then row and col size must not equal 0.\n";
            return;
        }

        ////////////////////
        clear();

        size_t r = src.size(), c = src.begin()->size();
        
        size_t _mSize_r = src.size();
        size_t _mSize_c = src.begin()->size();
        
        this->MDesc = new Matrixdsec(r, c, 1);

        size_t i = 0, j = 0;
        double** tmp2, * tmp;
        tmp2 = new double* [_mSize_r];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->Mat = tmp2;

        for (auto row = src.begin(); row != src.end(); ++row) {
            j = 0;
            tmp = new double[_mSize_c];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }
            for (auto ele = row->begin(); ele != row->end(); ++ele) {
                *(tmp + j) = *ele;
                ++j;
            }
            this->Mat[i] = tmp;
            ++i;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
        @ Purpose :以拷贝赋值的构造函数
    */
    Matrix::Matrix(const Matrix& Mat) {

        clear();
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        
        this->MDesc = new Matrixdsec(_mSize_r, _mSize_c, 1);

        size_t i = 0, j = 0;
        double** tmp2, * tmp;
        tmp2 = new double* [_mSize_r];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->Mat = tmp2;

        for (i = 0; i < _mSize_r; ++i) {
            tmp = new double[_mSize_c];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }

            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat[i] + j));
            }
            while (j < _mSize_c) {
                *(tmp + j) = Mat[i][j];
                ++j;
            }
            this->Mat[i] = tmp;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
     * @ Purpose : 标准输出
     */
    std::ostream& operator<<(std::ostream& out, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        if (_mSize_c == 0 || _mSize_r == 0) {
            return out;
        }

        out << "[";
        for (size_t i = 0; i < _mSize_r; ++i) {
            out << " ";
            for (size_t j = 0; j < _mSize_c; ++j) {
                if (abs(Mat[i][j]) < OUT_PRECISION) {
                    out <<0<<" ";
                }
                else {
                    out << Mat[i][j] << " ";
                }
            }
            if (i + 1 == _mSize_r) {
                out << "] , (" << _mSize_r << ", " << _mSize_c << ")";
            }
            out << "\n";
        }
        return out;
    }

    /*
        @ Purpose : 根据传入矩阵Mat更新[Low_r,High_r)->[Low_c,High_c)位置的数值
        @ Para  :
                    Mat 
                    [Low_r,High_r]->[Low_c,High_c] 准备更新的位置
        @ Example :
                    Matrix A = {{1,2}};
                    Matrix B = {{5,5,3},{4,5,6}};
                    B.part_set(A,0,1,0,2);  // 此时B更新为{{1,2,3},{4,5,6}}
    */
    bool Matrix::part_set(const Matrix& Mat, const size_t _Low_r, const size_t &_High_r, const size_t &_Low_c, const size_t &_High_c) {
        size_t High_c, High_r;
        std::tie(High_r, High_c) = this->shape();
        if (_High_c < High_c) {
            High_c = _High_c;
        }
        if (_High_r < High_r) {
            High_r = _High_r;
        }

        for (size_t i = _Low_r; i < High_r; ++i) {
            for (size_t j = _Low_c; j < High_c; ++j) {
                this->Mat[i][j] = Mat[i - _Low_r][j - _Low_c];
            }
        }

        return true;
    }

    /*
        @ Purpose : 截取矩阵的某行或某列
        @ Para  :
                r, 若此参数为-1，则取相应的c列
                c，若此参数为-1，则取相应的r行
        @ Other ： r与c其中一个参数必须为-1
        @ Return ：某行或某列的拷贝
        @ Example：
                Matrix A = {{1,2,3},{4,5,6}};
                Matrix B = A(-1,1);  // B={{2},{5}};
    */
    Matrix Matrix::operator()(const size_t& r, const size_t& c) const{
        if (r != -1 && c != -1) {
            std::cerr << "if you want to choose a row or col in a Mat, you need set [r] or [c] to -1 , another in it's range!\n";
            return Matrix();
        }
        Matrix child;
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = this->shape();
        
        if (r == -1) {
            child = Matrix(_mSize_r, 1);
            for (size_t i = 0; i < _mSize_r; ++i) {
                child[i][0] = this->Mat[i][c];
            }
        }
        else {
            child = Matrix(1, _mSize_c);
            for (size_t i = 0; i < _mSize_c; ++i) {
                child[0][i] = this->Mat[r][i];
            }
        }

        return child;
    }

    /*
        @ Purpose ：将矩阵转为string返回
        @ Return ： string
    */
    std::string Matrix::to_str() const{
        std::string res = "";

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = this->shape();
        std::string ZERO = "    0    ";
        std::string tmp;
        res += "[";
        for (size_t i = 0; i < _mSize_r; ++i) {
            if (i != 0) {
                res += " ";
            }
            res += "    ";
            for (size_t j = 0; j < _mSize_c; ++j) {
                if (abs(this->Mat[i][j]) < OUT_PRECISION) {
                    res += ZERO;
                }
                else {
                    tmp = (std::to_string(this->Mat[i][j]));
                    tmp.resize(12);
                    res +=  tmp;
                }
                res += "   ";
            }
            if (i + 1 == _mSize_r) {
                res += ("]");
            }
            res += "\n";
        }
        return res;
    }

    /*
       @ Purpose  : 根据索引返回数值
    */
    double* Matrix::operator[](const size_t& index) {
        return this->Mat[index];
    }
    /*
        @ Purpose  : 根据索引返回数值
        @ Other : 当矩阵以const传递时需要使用此函数
    */
    double* Matrix::operator[](const size_t& index) const {
        return this->Mat[index];
    }


    /*
        @ Purpose  : 返回当前矩阵的子矩阵
        @ Other    ：Low_r,Low_c必须小于当前矩阵行数或列数
    */
    Matrix Matrix::operator()(const size_t& _Low_r, const size_t& _High_r,
        const size_t& _Low_c, const size_t& _High_c) const{
        size_t High_c, High_r;
        std::tie(High_r, High_c) = this->shape();
        if (_High_c < High_c) {
            High_c = _High_c;
        }
        if (_High_r < High_r) {
            High_r = _High_r;
        }
        Matrix child(High_r-_Low_r,High_c-_Low_c);

        for (size_t i = _Low_r; i < High_r; ++i) {
            for (size_t j = _Low_c; j < High_c; ++j) {
                child[i - _Low_r][j - _Low_c] = this->Mat[i][j];
            }
        }
        return child;// res;
    }



    /*
     * @ Purpose  :转置矩阵自身
     */
    Matrix& Matrix::TRANS() {
        *this = &TRANSPOSITION(*this);
        return *this;
    }

    Matrix TRANSPOSITION(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_c, _mSize_r);
        size_t j = 0;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                res[j][i] = Mat[i][j];
                res[j + 1][i] = Mat[i][j + 1];
                res[j + 2][i] = Mat[i][j + 2];
                res[j + 3][i] = Mat[i][j + 3];
            }
            while (j < _mSize_c) {
                res[j][i] = Mat[i][j];
                ++j;
            }
        }
        return res;
    }


    /*
     * @ Purpose  : 求逆矩阵
     */
    Matrix INVERSE(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r != _mSize_c){
            std::cerr << "inverse must row==col size.\n";
            return Matrix();
        }
        if (_mSize_r == 0 || _mSize_c == 0)
        {
            std::cerr << "inverse row and col size must big than 0.\n";
            return Matrix();
        }
        return ADJOINT_Matrix(Mat) / DETERMINANT(Mat);
    }

    /*
     * @ Purpose  :返回Mat对角线元素
     */
    Matrix DIAGONAL(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        size_t _size = std::min(_mSize_r, _mSize_c);
        Matrix res(_size, 1);
        for (size_t i = 0; i < _size; ++i) {
            res[i][0] = Mat[i][i];
        }
        return res;
    }

    /*
        @ Purpose  :以src内数值创建一个[_size ,_size]对角矩阵
    */
    Matrix DIAGONAL(const size_t& _size, std::initializer_list<double> src) {
        Matrix res(_size, _size);
        size_t i = 0;
        for (auto el : src) {
            res[i][i] = el;
            ++i;
            if (i >= _size) {
                break;
            }
        }
        return res;
    }

    /*
       @ Purpose  : 以src内数值创建一个[rs,cs]对角矩阵
    */
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, std::initializer_list<double> src) {
        Matrix res(rs, cs);
        size_t i = 0;
        for (auto el : src) {
            res[i][i] = el;
            ++i;
            if (i >= rs || i >= cs) {
                break;
            }
        }
        return res;
    }

    /*
        @ Purpose  :根据Mat内的数值创建一个对角矩阵，Mat必须为[1,n]数组
    */
    Matrix DIAGONAL(const size_t& _size, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        if (_mSize_r != 1) {
            std::cerr << "if you init a new Diagonal Mat from Mat,Mat.shape must is [1,n]!\n";
            return Matrix();
        }
        Matrix res(_size, _size);
        size_t i = 0;
        for (i = 0; i < _size && i < _mSize_c; ++i) {
            res[i][i] = Mat[0][i];
        }
        return res;
    }

    /*
        @ Purpose  :根据Mat内的数值创建一个对角矩阵，Mat必须为[1,n]数组
    */
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        if (_mSize_r != 1) {
            std::cerr << "if you init a new Diagonal Mat from Mat,Mat.shape must is [1,n]!\n";
            return Matrix();
        }

        Matrix res(rs, cs);
        size_t i = 0;
        size_t n2 = std::min(rs, cs);
        for (i = 0; i < n2 && i < _mSize_c; ++i) {
            res[i][i] = Mat[0][i];
        }
        return res;
    }

    /*
     * @ Purpose  :获得一个[_size,_size]大小的单位对角矩阵
     */
    Matrix EYE(const size_t& _size) {
        Matrix res(_size, _size);
        for (size_t i = 0; i < _size; ++i) {
            res[i][i] = 1;
        }
        return res;
    }

    /*
     * @ Purpose  :获得一个[rs,cs]大小的单位对角矩阵
     */
    Matrix EYE(const size_t& rs, const size_t& cs) {
        Matrix res(rs, cs);
        for (size_t i = 0; i < rs && i < cs; ++i) {
            res[i][i] = 1;
        }

        return res;
    }
    
    /*
        @ Purpose  :生成一个rs,cs大小的[low,high)的整型随机数矩阵
    */
    Matrix RandI_Matrix(const size_t& rs, const size_t& cs, const int& low, const int& high) {
        size_t j = 0;
        Matrix res(rs, cs);
        std::random_device rd;
        std::uniform_int_distribution<int> dist(low, high);

        __m256d m256;
        for (size_t i = 0; i < rs; ++i) {
            for (j = 0; j + 4 <= cs; j += 4) {
                m256 = _mm256_set_pd(dist(rd), dist(rd), dist(rd), dist(rd));
                _mm256_store_pd(res[i] + j, m256);
            }
            while (j < cs) {
                res[i][j] = res[i][j] = dist(rd), dist(rd), dist(rd), dist(rd);
                ++j;
            }
        }
        return res;
    }

    /*
        @ Purpose  :生成一个rs,cs大小的[low,high)的双精度随机数矩阵
    */
    Matrix RandD_Matrix(const size_t& rs, const size_t& cs, const double& low, const double& high)
    {
        std::random_device rd;
        std::uniform_real_distribution<double> dist(low,high);
        size_t j = 0;
        Matrix res(rs, cs);
        __m256d m256;
        for (size_t i = 0; i < rs; ++i) {
            for (j = 0; j+4 <= cs; j+=4) {
                m256 = _mm256_set_pd(dist(rd), dist(rd), dist(rd), dist(rd));
                _mm256_store_pd(res[i] + j, m256);
            }
            while (j < cs) {
                res[i][j] = dist(rd);
                ++j;
            }
        }

        return res;
    }

    /*
        @ Purpose  :生成一个rs,cs大小的（M,S2）分布的正态矩阵
    */
    Matrix RandN_Matrix(const size_t& rs, const size_t& cs, const double& M, const double& S2)
    {
        std::random_device rd;
        std::normal_distribution<double> dist(M,S2);
        size_t j = 0;
        Matrix res(rs, cs);
        __m256d m256;
        for (size_t i = 0; i < rs; ++i) {
            for (j = 0; j + 4 <= cs; j += 4) {
                m256 = _mm256_set_pd(dist(rd), dist(rd), dist(rd), dist(rd));
                _mm256_store_pd(res[i] + j, m256);
            }
            while (j < cs) {
                res[i][j] = dist(rd);
                ++j;
            }
        }

        return res;
    }


    /*
     *  @ Purpose  :计算矩阵的行列式
     *  @ Iner_Para:
     *          i,j 控制子矩阵位置
     *          sub_i,sub_j 记录删除行列位置
     */
    double DETERMINANT(const Matrix& subMat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = subMat.shape();

        if (_mSize_r != _mSize_c) {
            std::cerr << "in func DETERMINANT(),Mat must row==col!\n";
            return 0;
        }
        if (_mSize_r == 0 || _mSize_c == 0) {
            std::cerr << "in func DETERMINANT(), Mat row and col size must big than 0;\n";
            return 0;
        }

        double res = 0.0;
        if (_mSize_r == 1) {
            res = subMat[0][0];
            return res;
        }
        if (_mSize_r == 2) {
            res = subMat[0][0] * subMat[1][1] - subMat[0][1] * subMat[1][0];
            return res;
        }

        Matrix new_subMat(_mSize_r - 1, _mSize_c - 1,0);
        for (size_t j = 0; j < _mSize_c; ++j) {
            for (size_t r = 0; r < _mSize_r; ++r) {
                for (size_t c = 0; c < _mSize_c; ++c) {
                    if (r != 0 && c != j) {
                        new_subMat[r > 0 ? r - 1 : r][c > j ? c - 1 : c] = subMat[r][c];
                    }
                }
            }
            res += (subMat[0][j] * (pow(-1, j) * DETERMINANT(new_subMat)));
        }
        return abs(res - (DETERMINANT_PRECISION)) > 0 ? res : 0;
    }

    /*
        @ Purpose  : 计算原矩阵的伴随矩阵
    */
    Matrix ADJOINT_Matrix(const Matrix& Mat) {
        std::size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r != _mSize_c) {
            std::cerr << "in func ADJOINT_Matrix(),Mat must row==col!\n";
            return Matrix();
        }
        if (_mSize_r == 0 || _mSize_c == 0) {
            std::cerr << "in func ADJOINT_Matrix(), Mat row and col size must big than 0;\n";
            return Matrix();
        }
        Matrix res(_mSize_r, _mSize_c);
        Matrix sub_Mat(_mSize_r - 1, _mSize_c - 1);

        if (_mSize_r == 1) {
            res = { {1} };
            return res;
        }
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _mSize_c; ++j) {

                for (size_t r = 0; r < _mSize_r; ++r) {
                    for (size_t c = 0; c < _mSize_c; ++c) {
                        if (r != i && c != j) {
                            sub_Mat[r > i ? r - 1 : r][c > j ? c - 1 : c] = Mat[r][c];
                        }
                    }
                }
                res[i][j] = (pow(-1, i + j) * DETERMINANT(sub_Mat));
            }
        }
        res.TRANS();
        return res;
    }

    /*
       @ Purpose  : 返回矩阵最大值
    */
    double MAX(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        double tmp[4] = { DMIN,DMIN,DMIN,DMIN };
        __m256d max_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        double res = DMIN;
        size_t j = 0;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                max_256 = _mm256_max_pd(_mm256_load_pd(Mat[i] + j), max_256);
            }
            _mm256_store_pd(tmp, max_256);
            res = std::max(res, std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3])));
            while (j < _mSize_c) {
                res = std::max(res, Mat[i][j]);
                ++j;
            }
            max_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        }
        return res;
    }

    /*
        @ Purpose  :返回矩阵最小值
    */  
    double MIN(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        double tmp[4] = { DMAX,DMAX,DMAX,DMAX };
        __m256d max_256 = _mm256_set_pd(DMAX, DMAX, DMAX, DMAX);
        double res = DMAX;
        size_t j = 0;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                max_256 = _mm256_min_pd(_mm256_load_pd(Mat[i] + j), max_256);
            }
            _mm256_store_pd(tmp, max_256);
            res = std::min(res, std::min(std::min(tmp[0], tmp[1]), std::min(tmp[2], tmp[3])));
            while (j < _mSize_c) {
                res = std::min(res, Mat[i][j]);
                ++j;
            }
            max_256 = _mm256_set_pd(DMAX, DMAX, DMAX, DMAX);
        }
        return res;
    }

    /*
         @ Purpose  : A = U*Sigma*VT
         @ Para :
                     Mat 计算的原矩阵
         @ Return : 返回原矩阵对应的U，Sigma，V
         @ Other : 暂不支持复数结果
         @ Example :
                 Matrix A = {{1,2,3},{4,5,6},{7,8,9}};
                 Matrix U,S,V;
                 std::tie(U,S,V) = SVD(Mat);
     */
    std::tuple<Matrix, Matrix, Matrix> SVD(const Matrix& Mat) {

        Matrix ATA, AAT, tmp, U, Sigma, V;
        tmp = &TRANSPOSITION(Mat);
        ATA = &(tmp * Mat);
        AAT = &(Mat * tmp);
        std::tie(std::ignore, V) = JACOBI(ATA);
        std::tie(Sigma, U) = JACOBI(AAT);

        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Sigma.TRANS();
        Sigma = &DIAGONAL(_mSize_r, _mSize_c, Sigma);
        Sigma = &SQRT(Sigma);

        return std::make_tuple(U, Sigma, V);
    }

     /*
         @ Purpose  : 雅克比法计算特征值与特征向量
         @ Para :
                     Mat 计算的原矩阵
         @ Return : 返回原矩阵的特征值(参数1)和相应的特征向量(参数2)。
         @ Other : 暂不支持复数结果
         @ Example :
                 Matrix A = {{1,2,3},{4,5,6},{7,8,9}};
                 Matrix Rv,R;
                 std::tie(Rv,R) = JACOBI(Mat);
     */
    std::tuple<Matrix, Matrix> JACOBI(const Matrix& Mat) {
        std::size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix EigenVector;
        EigenVector = &EYE(_mSize_r, _mSize_c);
        Matrix U;
        U = &EYE(_mSize_r, _mSize_c);
        Matrix UT; // U's trans
        Matrix EigenValue(_mSize_r, 1);
        Matrix copy_Mat = Mat; 
        size_t Iter = 0;
        double dbangle, sintheta, costheta;

        while (Iter < MAX_ITER)
        {
            // 寻找非对角线元素最大值，及位置
            double non_dia_max_value_abs = abs(copy_Mat[0][1]);
            size_t non_dia_max_row = 0, non_dia_max_col = 1;
            for (size_t i = 0; i < _mSize_r; ++i) {
                for (size_t j = 0; j < _mSize_c; ++j) {
                    if (i != j && abs(copy_Mat[i][j]) > non_dia_max_value_abs)
                    {
                        non_dia_max_row = i;
                        non_dia_max_col = j;
                        non_dia_max_value_abs = abs(copy_Mat[i][j]);
                    }
                }
            }

            // 检车是否需要退出循环
            if (non_dia_max_value_abs < PRECISION) {
                break;
            }

            // 计算旋转矩阵
            if (copy_Mat[non_dia_max_col][non_dia_max_col] == copy_Mat[non_dia_max_row][non_dia_max_row]) {
                dbangle = PI / 4;
            }
            else {
                dbangle = 0.5 * atan2(2 * copy_Mat[non_dia_max_row][non_dia_max_col],
                    copy_Mat[non_dia_max_col][non_dia_max_col] - copy_Mat[non_dia_max_row][non_dia_max_row]);
            }

            sintheta = sin(dbangle);
            costheta = cos(dbangle);

            // 计算特征向量 ,givens rotation Matrix
            U[non_dia_max_row][non_dia_max_row] = costheta;
            U[non_dia_max_row][non_dia_max_col] = -sintheta;
            U[non_dia_max_col][non_dia_max_row] = sintheta;
            U[non_dia_max_col][non_dia_max_col] = costheta;
            UT = &TRANSPOSITION(U);
            copy_Mat = &(U * copy_Mat * UT);

            EigenVector = &(EigenVector * (UT));
            fast_copy(EigenVector, EigenVector * (UT));
            U[non_dia_max_row][non_dia_max_row] = 1;
            U[non_dia_max_row][non_dia_max_col] = 0;
            U[non_dia_max_col][non_dia_max_row] = 0;
            U[non_dia_max_col][non_dia_max_col] = 1;

            ++Iter;
        }

        // 计算特征值
        EigenValue = &DIAGONAL(copy_Mat);
        // 排序
        double _max_value;
        size_t _max_index;
        double tmp;
        std::tie(_mSize_r, _mSize_c) = EigenValue.shape();
        for (size_t i = 0; i < _mSize_r; ++i) {
            _max_index = i;
            _max_value = EigenValue[i][0];
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                if (_max_value < EigenValue[j][0]) {
                    _max_index = j;
                    _max_value = EigenValue[j][0];
                }
                if (abs(EigenVector[i][j]) < PRECISION) {
                    EigenVector[i][j] = 0;
                }
            }
            if (abs(EigenValue[i][0]) < PRECISION) {
                EigenValue[i][0] = 0;
            }
            tmp = EigenValue[i][0];
            std::swap(EigenValue[i][0], EigenValue[_max_index][0]);
            for (size_t k = 0; _max_index != i && k < _mSize_r; ++k) {
                std::swap(EigenVector[k][i], EigenVector[k][_max_index]);
            }
        }
        return std::make_tuple(EigenValue, EigenVector);
    }


    /*
        @ Purpose  : 利用幂法求矩阵特征值
        @ Para :
                    Mat 计算的原矩阵
                    min_delta 控制计算精度
                    max_iter 控制计算轮数
        @ Return : 返回原矩阵的最大特征值和相应的特征向量。
        @ Other : 暂不支持复数结果，矩阵为方阵
        @ Example :
                Matrix A = {{1,2,3},{4,5,6},{7,8,9}};
                Matrix R;
                double Rv;
                std::tie(Rv,R) = Power_Method(Mat);
    */
    std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta,const size_t& max_iter) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r != _mSize_c) {
            std::cerr << "in func Power_Method(), Mat row and col size must equal.\n";
            return std::make_tuple(0, Matrix());
        }


        Matrix X = Matrix(_mSize_r, 1, 1);   // 特征向量
        Matrix Y;
        double m = 0, pre_m = 0;   //特征值
        size_t iter = 0;
        double Delta = INT32_MAX;
        while (iter < max_iter && Delta >min_delta) {
            Y = &(Mat * X);
            m = MAX(Y);
            Delta = abs(m - pre_m);
            pre_m = m;
            X = &(Y / m);
            iter += 1;
        }
        return std::make_tuple(m, X);
    }

    /*
        @ Purpose  : 计算矩阵的幂
        @ Return : 计算结果存放在一个新的矩阵并返回
        @ Other : 暂不支持复数结果
        @ Example :
                Matrix A = {{1,2,3},{4,5,6}};
                Matrix B;
                B = SQRT(A);    // 此时为拷贝赋值，
                B = &SQRT(A);   // 此时为指针； 建议使用本方法
    */
    Matrix SQRT(const Matrix& Mat) {
        Matrix dst(Mat);
        size_t _mSize_r, _mSize_c;
        size_t j, i;

        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4)
            {
                _mm256_store_pd(dst[i] + j,
                    _mm256_sqrt_pd(_mm256_load_pd(Mat[i] + j)));
            }
            while (j < _mSize_c) {
                dst[i][j] = std::sqrt(Mat[i][j]);
                ++j;
            }
        }
        return dst;
    }

    /*
        @ Purpose : 当一个矩阵被多次使用时，且准备同步更新则需要使用此函数
        @ Para :
                    Mat 原矩阵，或者其他引用该矩阵的变量
        @ Return :
                    bool
        @ Other : 此函数不应在其他函数内部进行显示的调用，可能产生潜在的内存泄漏威胁
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  Matrix B;
                  B = &A;  //此时矩阵B绑定了矩阵A
                  B = Matrix(3,3); // 此时矩阵A也被同时更新
                  B.clear();    //释放与矩阵A的绑定
                  B = Matrix(4,4); // 此时仅更新B
    */
    bool Matrix::memAsycEqual(const Matrix& Mat) {
        // 指向同一内存地址
        if (this->Mat == Mat.Mat) {
            return true;
        }
        
        // 若两个矩阵大小一致，则使用函数fast_copy();不重新分配内存
        if (this->shape() == Mat.shape()) {
            fast_copy(*this, Mat);
            return true;
        }

        size_t new_Size_r, new_Size_c;   // 
        std::tie(new_Size_r, new_Size_c) = Mat.shape();
        size_t old_Size_r, old_Size_c;
        std::tie(old_Size_r, old_Size_c) = this->shape();

        // tmp2:Mat
        // tmp :Row
        double** tmp2, * tmp;
        
        size_t j = 0;
        tmp2 = new double* [new_Size_r];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }

        for (size_t i = 0; i < new_Size_r; ++i) {
            tmp = new double[new_Size_c];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }

            for (j = 0; j + 4 <= new_Size_c; j += 4) {
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat[i] + j));
            }
            while (j < new_Size_c) {
                *(tmp + j) = Mat[i][j];
                ++j;
            }
            tmp2[i] = tmp;
        }
        
        auto waitDelMat = this->Mat;
        for (size_t i = 0; i < this->MDesc->memAsyc.size(); ++i) {
            this->MDesc->memAsyc[i]->Mat = tmp2;
        }
        this->MDesc->row_size = new_Size_r;
        this->MDesc->col_size = new_Size_c;
        for (size_t i = 0; i < old_Size_r; ++i) {
            delete []waitDelMat[i];
        }
        delete[]waitDelMat;
        return true;
    }

    /*
        @ Purpose 快速拷贝，必须确保dst的矩阵行列数大于src
    */
    bool fast_copy(Matrix& dst,const Matrix &src) {

        size_t m_Size_r, m_Size_c;
        std::tie(m_Size_r, m_Size_c) = dst.shape();

        for (size_t i = 0; i < m_Size_r; ++i) {
            std::copy(src[i], src[i] + m_Size_c, dst[i]);
        }
        return true;
    }

    /*
        @ Purpose : 计算矩阵的LU分解
        @ Para :
                    Mat 传入的原矩阵
        @ Return :
                    Matrix(2) L矩阵
                    Matrix(3) U矩阵
        @ Other : 矩阵需为方阵
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6},{7,8,9}};
                  std::tie(L,U) = LU(A);
    */
    std::tuple<Matrix, Matrix> LU(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r != _mSize_c) {
            std::cerr << "in func LU(), Mat row and col size must equal.\n";
            return std::make_tuple(Matrix(), Matrix());
        }

        Matrix L = EYE(_mSize_r);
        Matrix U(Mat);
        __m256d m256D;
        size_t j, k;
        double ele;
        for (size_t i = 0; i < _mSize_r; ++i) {
            //U中 i列下方元素变为0；
            for ( j = i + 1; j+2 <= _mSize_r; j +=2) {
                ele = U[j][i] / U[i][i];
                L[j][i] = ele;
                m256D = _mm256_set_pd(ele, ele, ele, ele);
                
                for (k = 0; k + 4 <= _mSize_c; k += 4) {
                    _mm256_store_pd(U[j] + k, _mm256_sub_pd(
                        _mm256_load_pd(U[j] + k), 
                        _mm256_mul_pd(_mm256_load_pd(U[i] + k),m256D)
                    ));
                }
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                }
            }
            //std::cout << i<<"\n";
        }
        return std::make_tuple(L, U);
    }
    

    /*
        @ Purpose  :交换矩阵第i行与第ii行
    */
    void row_swap(Matrix& Mat, size_t i, size_t ii) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (i > _mSize_r || ii > _mSize_r) {
            std::cerr << "in func row_swap, i and ii must in range[0,_row_size).\n";
            return;
        }

        for (size_t j = 0; j < _mSize_c; ++j) {
            std::swap(Mat[i][j], Mat[ii][j]);
        }
    }

    /*
        @ Purpose  :为计算LU分解进行修改的行交换函数
    */
    void row_swap_PLU(Matrix& Mat, size_t i, size_t ii,size_t col_index,bool Left) {
        
        if (Left) {
            for (size_t j = 0; j <= col_index; ++j) {
                std::swap(Mat[i][j], Mat[ii][j]);
            }
        }
        else {
            size_t _mSize_c;
            std::tie(std::ignore, _mSize_c) = Mat.shape();
            for (size_t j = col_index; j <_mSize_c; ++j) {
                std::swap(Mat[i][j], Mat[ii][j]);
            }
        }
    }

    /*
        @ Purpose : 计算矩阵的PLU分解
        @ Para :
                    Mat 传入的原矩阵
        @ Return :
                    Matrix(1) P矩阵
                    Matrix(2) L矩阵
                    Matrix(3) U矩阵
        @ Other :
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  std::tie(P,L,U) = PLU(A);
    */
    std::tuple<Matrix, Matrix, Matrix> PLU(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix P(_mSize_r,_mSize_r);
        std::vector<size_t> P2(_mSize_r,0);
        for (size_t i = 0; i < _mSize_r; ++i) {
            P2[i] = i;
        }
        Matrix L = Matrix(_mSize_r, _mSize_r, 0);
        Matrix U(Mat);
        size_t max_row_index;
        size_t current_col = 0;
        size_t i = 0;
        __m256d m256D;
        double ele;

        for (i = 0; i < _mSize_r && current_col <_mSize_c; ++i) {
            // 寻找该列最大值
            max_row_index = i;

            for (; current_col < _mSize_c;++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (U[max_row_index][current_col] < U[m][current_col]) {
                        max_row_index = m;
                    }
                }

                if (U[max_row_index][current_col] != 0) {
                    break;
                }
            }
            
            if (current_col == _mSize_c || max_row_index==_mSize_r) {
                break;
            }

            // 交换该轮次最大值行
            if (max_row_index != i) {
                std::swap(P2[i], P2[max_row_index]);
                row_swap_PLU(U, i, max_row_index, current_col, false);
                row_swap_PLU(L, i, max_row_index, current_col, true);
            }

            //U中 i列下方元素变为0；
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                L[j][current_col] = ele;
                m256D = _mm256_set_pd(ele, ele, ele, ele);
                size_t k = 0;
                for (k = current_col; k+4 <= _mSize_c; k+=4) {
                   // U[j][k] = U[j][k] - U[i][k] * ele;
                    _mm256_store_pd(U[j] + k, _mm256_sub_pd(
                        _mm256_load_pd(U[j] + k), 
                        _mm256_mul_pd(_mm256_load_pd(U[i] + k),m256D)
                    ));
                }
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                
                }

            }
            current_col += 1;
        }
        
        for (size_t i = 0; i < _mSize_r; ++i) {
            P[i][P2[i]] = 1;
        }

        L += (EYE(_mSize_r));
        return std::make_tuple(P, L, U);
    }

    /*
        @ Matrix function : rk(A)
        @ Purpose : 计算矩阵的秩
        @ Para :
                    Mat 传入的原矩阵
        @ Return :
                    size_t 矩阵的秩，大于等于0且小于等于矩阵的行数 
                    Matrix  一个上三角矩阵
        @ Other : 
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  Matrix R;
                  size_t Rk;
                  std::tie(Rk,R) = Rank(A);
    */
    std::tuple<size_t,Matrix> Rank(const Matrix& Mat) {
        size_t rank = 0;
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix U(Mat);
        size_t max_row_index;
        size_t current_col = 0;
        size_t i = 0;
        __m256d m256D;
        double ele;

        for (i = 0; i < _mSize_r && current_col < _mSize_c; ++i) {
            // 寻找该列最大值
            max_row_index = i;

            for (; current_col < _mSize_c; ++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (U[max_row_index][current_col] < U[m][current_col]) {
                        max_row_index = m;
                    }
                }

                if (U[max_row_index][current_col] != 0) {
                    break;
                }
            }

            if (current_col == _mSize_c || max_row_index == _mSize_r) {
                break;
            }

            // 交换该轮次最大值行
            if (max_row_index != i) {
                row_swap_PLU(U, i, max_row_index, current_col, false);
            }

            //U中 i列下方元素变为0；
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                m256D = _mm256_set_pd(ele, ele, ele, ele);
                size_t k = 0;
                for (k = current_col; k + 4 <= _mSize_c; k += 4) {
                    _mm256_store_pd(U[j] + k, _mm256_sub_pd(
                        _mm256_load_pd(U[j] + k),
                        _mm256_mul_pd(_mm256_load_pd(U[i] + k), m256D)
                    ));
                }
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                }
            }
            current_col += 1;
        }

        current_col = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for (current_col=0; current_col < _mSize_c;++current_col) {
                if (abs(U[i][current_col]) < PRECISION) {
                    U[i][current_col] = 0;
                }
                if (U[i][current_col] != 0) {
                    rank += 1;
                    break;
                }
            }
            current_col += 1;
        }

        return std::make_tuple(rank,U);
    }

    /*
        @ Matrix function : |Mat|2
        @ Purpose : 计算原矩阵的2范数
        @ Para : 
                Mat 传入的原矩阵
        @ Return :
        @ Example :
                Matrix A = {{1,2,3},{4,5,6}};
                Matrix B;
                B = Norm_2(A);
    */
    double Norm_2(const Matrix& Mat) {
        double res = 0;
        Matrix ATA = TRANSPOSITION(Mat) * Mat;
        std::tie(res, std::ignore) = Power_Method(ATA);
        return sqrt(res);
    }

    /*
        @ Matrix function : A = QR
        @ Purpose : 使用格里姆施密特正交法计算QR分解
        @ Para : 
                    Mat 传入的原矩阵
        @ Return :  
                    Q,R 返回最终计算结果
        @ Other : 返回矩阵不支持复数结果
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  Matrix Q,R;
                  std::tie(Q,R) = QR(A);
                    
    */
    std::tuple<Matrix,Matrix> QR(const Matrix& Mat) {
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix Q(_mSize_r, _mSize_c);
        Matrix R(_mSize_c, _mSize_c);
        Matrix q(_mSize_r, 1);      // 临时变量
        Matrix y(_mSize_r, 1);      // 临时中间变量
        Matrix yT;                  // 临时变量
        size_t j, i;
        double norm_2 = 0;
        for (j = 0; j < _mSize_c; ++j) {
            // y = A[][j]
            y = &Mat(-1, j);
            // q(i)
            
            for (i = 0; j>0 && i < j; ++i) {
                q = &Q(-1, i);
                yT = &(TRANSPOSITION(q) * y);
                R[i][j] = yT[0][0];
                y = &(y - DOT(q, R[i][j])); //y -= (q * R[i][j]);
            }
            norm_2 = Norm_2(y);
            R[j][j] = norm_2;
            q = &(y / norm_2);
            Q.part_set(q, 0, -1, j, j + 1);
        }
        return std::make_tuple(Q, R);
    }
    
    /*
        @ Purpose : 将矩阵向上或向下取整，并返回一个新的矩阵
    */
    Matrix Ceil(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t i, j;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j, _mm256_ceil_pd(_mm256_load_pd(Mat[i] + j)));
            }
            while (j < _mSize_c) {
                res[i][j] = std::ceil(Mat[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix Floor(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t i, j;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j, _mm256_floor_pd(_mm256_load_pd(Mat[i] + j)));
            }
            while (j < _mSize_c) {
                res[i][j] = std::floor(Mat[i][j]);
                ++j;
            }
        }
        return res;
    }

    /*
        @ Purpose :当计算多个矩阵乘法时，运算顺序对计算速度影响很大，此函数对计算顺序进行重新排列并进行计算
        @ Return : 返回传入矩阵的乘法运算结果
        @ Other : 矩阵必须满足叉乘规则
        @ Example :
                Matrix A,B,C,D;
                A = RandI_Matrix(3, 3, 1, 10);
                B = RandI_Matrix(3, 10, 1, 10);
                C = RandI_Matrix(10, 20, 1, 10);
                D = RandI_Matrix(20, 3, 1, 10);
                Matrix D = MMul({A,B,C,D});
    */
    Matrix MMul(std::initializer_list <Matrix> Mats) {
        Matrix res;



        return res;
    }


    /*
        @ Purpose : 获取矩阵Mat和Mat2的比较结果
        @ Return : 若Mat[0,0] == Mat2[0,0]，则res[0,0]=1，否则为0
        @          若矩阵大小不一致，则返回一个Matrix()空矩阵
        @ Other : 矩阵大小必须一致
    */
    Matrix operator==(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_EQ_US));
                res[i][j] = (res[i][j] &&0x1);
                res[i][j+1] = (res[i][j+1]&&0x1);
                res[i][j+2] = (res[i][j+2] &&0x1);
                res[i][j+3] = (res[i][j+3] &&0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] == Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix operator!=(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_NEQ_UQ));
                res[i][j] = (res[i][j] && 0x1);
                res[i][j + 1] = (res[i][j + 1] && 0x1);
                res[i][j + 2] = (res[i][j + 2] && 0x1);
                res[i][j + 3] = (res[i][j + 3] && 0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] != Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix operator>(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_GT_OQ));
                res[i][j] = (res[i][j] && 0x1);
                res[i][j + 1] = (res[i][j + 1] && 0x1);
                res[i][j + 2] = (res[i][j + 2] && 0x1);
                res[i][j + 3] = (res[i][j + 3] && 0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] > Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix operator<(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_LT_OS));
                res[i][j] = (res[i][j] && 0x1);
                res[i][j + 1] = (res[i][j + 1] && 0x1);
                res[i][j + 2] = (res[i][j + 2] && 0x1);
                res[i][j + 3] = (res[i][j + 3] && 0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] < Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix operator>=(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_NLT_UQ));
                res[i][j] = (res[i][j] && 0x1);
                res[i][j + 1] = (res[i][j + 1] && 0x1);
                res[i][j + 2] = (res[i][j + 2] && 0x1);
                res[i][j + 3] = (res[i][j + 3] && 0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] >= Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix operator<=(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
        size_t j;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(res[i] + j,
                    _mm256_cmp_pd(_mm256_load_pd(Mat[i] + j), _mm256_load_pd(Mat2[i] + j), _CMP_LE_OQ));    //
                res[i][j] = (res[i][j] && 0x1);
                res[i][j + 1] = (res[i][j + 1] && 0x1);
                res[i][j + 2] = (res[i][j + 2] && 0x1);
                res[i][j + 3] = (res[i][j + 3] && 0x1);
            }
            while (j < _mSize_c)
            {
                res[i][j] = (Mat[i][j] <= Mat2[i][j]);
                ++j;
            }
        }
        return res;
    }
}

#endif //LALIB_MATRIX_H
