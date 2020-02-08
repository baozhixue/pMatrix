//
// created by zhixu on 2020/2/5.
//

#ifndef LALIB_MATRIX_H
#define LALIB_MATRIX_H

#include <iostream>
#include <immintrin.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <ctime>
#include <tuple>
#include <thread>

namespace bzx2 {

    constexpr double PI = 3.1415926;
    constexpr double DMIN = std::numeric_limits<double>::min();  // double min

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
        std::vector<Matrix*> memAsyc;
    };  
    
    class Matrix {
        // double version
        friend std::ostream& operator<<(std::ostream& out, const Matrix& m);
        friend Matrix operator+(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix operator-(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix operator/(const Matrix& Mat, const double& num);
        friend Matrix SQRT(const Matrix& Mat);
        friend double MAX(const Matrix& Mat);
        friend Matrix operator*(const Matrix& Mat, const Matrix& Mat2);   // 叉乘
        friend Matrix DOT(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix DOT(const Matrix& Mat, const double& num);
        friend Matrix INVERSE(const Matrix& Mat);
        friend Matrix ADJOINT_Matrix(const Matrix& Mat);
        friend Matrix DIAGONAL(const Matrix& Mat);
        friend Matrix DIAGONAL(const size_t& _size, std::initializer_list<double> src);
        friend Matrix DIAGONAL(const size_t& rs, const size_t& cs, std::initializer_list<double> src);
        friend Matrix DIAGONAL(const size_t& _size, const Matrix& Mat);
        friend Matrix DIAGONAL(const size_t& rs, const size_t& cs, const Matrix& Mat);
        friend Matrix EYE(const size_t& rs, const size_t& cs);
        friend Matrix EYE(const size_t& _size);
        friend Matrix RAND_Matrix(const size_t& rs, const size_t& cs, const double& low, const double& high);
        friend double DETERMINANT(const Matrix& Mat);
        friend double _DETERMINANT(const Matrix& subMat);
        friend Matrix TRANSPOSITION(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix, Matrix> SVD(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix> JACOBI(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix> EigenDec(const Matrix& Mat);
        friend std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta, const size_t& max_iter);
        friend bool fast_copy(Matrix& dst, const Matrix& src);
        
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
        bool memAsycEqual(const Matrix& Mat);
        int use_count()const;
        bool part_set(const Matrix& Mat, const size_t _Low_r, const size_t& _High_r, const size_t& _Low_c, const size_t& _High_c);


        Matrix& TRANS();
        std::tuple<size_t, size_t> shape() const {
            if (this->MDesc == NULL) {
                return std::make_tuple(0, 0);
            }
            return std::make_tuple(this->MDesc->row_size, this->MDesc->col_size);
        }

        ~Matrix();

    private:
        double** Mat = NULL; // 2d
        Matrixdsec* MDesc = NULL;

    };

    
    class Timer {
    public:
        Timer() = default;
        void SetTimer() { Clock = clock(); Running = true; };
        double Stop() { 
            if (Running) {
                Running = false;
                return 1.0*(clock() - Clock)/CLOCKS_PER_SEC;
            }
            return -1;
        }
        double StopAndAgaing() {
            if (Running) {
                return 1.0 * (clock() - Clock) / CLOCKS_PER_SEC;
            }
            return -1;
        }
    private:
        time_t Clock;
        bool Running = false;
    };


    Matrix::~Matrix() {
        clear();
    }
    

    /*
        返回矩阵内存被使用数量
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

    // 拷贝赋值
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
                    _mm256_store_pd(tmp + j, _mm256_load_pd(Mat.Mat[i] + j));
                }
                while (j < _mSize_c) {
                    *(tmp + j) = Mat.Mat[i][j];
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

    // 指针
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
     * 叉乘
     */
    Matrix operator*(const Matrix& Mat, const Matrix& Mat2) {

        size_t _mSize_r, _mSize_c;
        size_t _m2Size_r, _m2Size_c;

        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        std::tie(_m2Size_r, _m2Size_c) = Mat2.shape();


        assert(_mSize_c == _m2Size_r);

        Matrix dst(_mSize_r, _m2Size_c, 0);
        // version 0.2
        size_t k = 0;
        __m256d dst_m256d = _mm256_set_pd(0, 0, 0, 0);
        __m256d Mat2_m256d;
        double dst_array[4];
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _m2Size_c; ++j) {
                for (k = 0; k + 4 <= _m2Size_r; k += 4) {
                    Mat2_m256d = _mm256_set_pd(Mat2.Mat[k + 3][j], Mat2.Mat[k + 2][j], Mat2.Mat[k + 1][j], Mat2.Mat[k][j]);
                    dst_m256d = _mm256_add_pd(dst_m256d, _mm256_mul_pd(_mm256_load_pd(Mat.Mat[i] + k), Mat2_m256d));
                }
                _mm256_store_pd(dst_array, dst_m256d);
                dst.Mat[i][j] = dst_array[0] + dst_array[1] + dst_array[2] + dst_array[3];
                while (k < _mSize_c) {
                    dst.Mat[i][j] += (Mat.Mat[i][k] * Mat2.Mat[k][j]);
                    ++k;
                }
                dst_m256d = _mm256_set_pd(0, 0, 0, 0);
            }
        }

        return dst;
    }


    Matrix DOT(const Matrix& Mat, const Matrix& Mat2) {
        size_t _mSize_r, _mSize_c;
        size_t _m2Size_r, _m2Size_c;
        
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        std::tie(_m2Size_r, _m2Size_c) = Mat2.shape();

        assert(_m2Size_r == _mSize_r);

        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        if (_m2Size_c == _mSize_c) {
            for (i = 0; i < _mSize_r; ++i) {
                for (j = 0; j + 4 <= _mSize_c; j += 4) {
                    _mm256_store_pd(dst.Mat[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.Mat[i] + j), _mm256_load_pd(Mat2.Mat[i] + j)));
                }
                while (j < _mSize_c) {
                    dst.Mat[i][j] = Mat.Mat[i][j] * Mat2.Mat[i][j];
                    ++j;
                }
            }
        }
        else {
            assert(_m2Size_c == 1);
            for (i = 0; i < _mSize_r; ++i) {
                __m256d m256d = _mm256_set_pd(Mat2.Mat[i][0], Mat2.Mat[i][0], Mat2.Mat[i][0], Mat2.Mat[i][0]);
                for (j = 0; j + 4 <= _mSize_c; j += 4) {
                    _mm256_store_pd(dst.Mat[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.Mat[i] + j), m256d));
                }
                while (j < _mSize_c) {
                    dst.Mat[i][j] = Mat.Mat[i][j] * Mat2.Mat[i][0];
                    ++j;
                }
            }
        }

        return dst;
    }

    Matrix DOT(const Matrix& Mat, const double& num) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst.Mat[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.Mat[i] + j), m256));
            }
            while (j < _mSize_c) {
                dst.Mat[i][j] = Mat.Mat[i][j] * num;
                ++j;
            }
        }
        return dst;
    }



    Matrix operator/(const Matrix& Mat, const double& num) {
        assert(num != 0);
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst.Mat[i] + j, _mm256_div_pd(_mm256_load_pd(Mat.Mat[i] + j), m256));
            }
            while (j < _mSize_c) {
                dst.Mat[i][j] = Mat.Mat[i][j] / num;
                ++j;
            }
        }
        return dst;
    }


    Matrix operator+(const Matrix& Mat, const Matrix& Mat2) {
        assert(Mat.shape() == Mat2.shape());
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst.Mat[i] + j, _mm256_add_pd(_mm256_load_pd(Mat.Mat[i] + j), _mm256_load_pd(Mat2.Mat[i] + j)));
            }
            while (j < _mSize_c) {
                dst.Mat[i][j] = Mat.Mat[i][j] + Mat2.Mat[i][j];
                ++j;
            }
        }
        return dst;
    }

    Matrix operator-(const Matrix& Mat, const Matrix& Mat2) {
        assert(Mat.shape() == Mat2.shape());
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(_mSize_r, _mSize_c);
        size_t i = 0, j = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                _mm256_store_pd(dst.Mat[i] + j, _mm256_sub_pd(_mm256_load_pd(Mat.Mat[i] + j), _mm256_load_pd(Mat2.Mat[i] + j)));
            }
            while (j < _mSize_c) {
                dst.Mat[i][j] = Mat.Mat[i][j] - Mat2.Mat[i][j];
                ++j;
            }
        }
        return dst;
    }

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

    Matrix::Matrix(std::initializer_list<std::initializer_list<double >> src) {
        assert(src.size() > 0);
        assert(src.begin()->size() > 0);

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
        以拷贝赋值的构造函数
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
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat.Mat[i] + j));
            }
            while (j < _mSize_c) {
                *(tmp + j) = Mat.Mat[i][j];
                ++j;
            }
            this->Mat[i] = tmp;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
     * 输出
     */
    std::ostream& operator<<(std::ostream& out, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        out << "[";
        for (size_t i = 0; i < _mSize_r; ++i) {
            out << " ";
            for (size_t j = 0; j < _mSize_c; ++j) {
                out << Mat.Mat[i][j] << " ";
            }
            if (i + 1 == _mSize_r) {
                out << "] , (" << _mSize_r << ", " << _mSize_c << ")";
            }
            out << "\n";
        }
        return out;
    }

    //将矩阵的[Low_r,High_r]->[Low_c,High_c]设置为传入矩阵的值,若High_r或High_c为-1则默认为至相应的矩阵最大值
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

    // 返回某行或某列
    /*
        当使用此函数时，r或c某一个参数必须为-1；
    */
    Matrix Matrix::operator()(const size_t& r, const size_t& c) const{
        assert(r == -1 || c == -1);
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

    double* Matrix::operator[](const size_t& index) {
        return this->Mat[index];
    }
    double* Matrix::operator[](const size_t& index) const {
        return this->Mat[index];
    }
    /*
        返回当前矩阵的子矩阵
        Low_r,Low_c必须小于当前矩阵行数或列数
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
     * 转置矩阵自身
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
                res.Mat[j][i] = Mat.Mat[i][j];
                res.Mat[j + 1][i] = Mat.Mat[i][j + 1];
                res.Mat[j + 2][i] = Mat.Mat[i][j + 2];
                res.Mat[j + 3][i] = Mat.Mat[i][j + 3];
            }
            while (j < _mSize_c) {
                res.Mat[j][i] = Mat.Mat[i][j];
                ++j;
            }
        }
        return res;
    }


    /*
     * 求逆矩阵
     */
    Matrix INVERSE(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        assert(_mSize_r > 0 && _mSize_r == _mSize_c);
        return ADJOINT_Matrix(Mat) / DETERMINANT(Mat);
    }

    /*
     * 行列式
     */
    double DETERMINANT(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == _mSize_c && _mSize_r > 0);
        return _DETERMINANT(Mat);
    }

    /*
     * 返回Mat对角线元素
     */
    Matrix DIAGONAL(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        size_t _size = std::min(_mSize_r, _mSize_c);
        Matrix res(_size, 1);
        for (size_t i = 0; i < _size; ++i) {
            res.Mat[i][0] = Mat.Mat[i][i];
        }
        return res;
    }

    /*
        以src内数值创建一个[_size ,_size]对角矩阵
    */
    Matrix DIAGONAL(const size_t& _size, std::initializer_list<double> src) {
        Matrix res(_size, _size);
        size_t i = 0;
        for (auto el : src) {
            res.Mat[i][i] = el;
            ++i;
            if (i >= _size) {
                break;
            }
        }
        return res;
    }

    /*
        以src内数值创建一个[rs,cs]对角矩阵
    */
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, std::initializer_list<double> src) {
        Matrix res(rs, cs);
        size_t i = 0;
        for (auto el : src) {
            res.Mat[i][i] = el;
            ++i;
            if (i >= rs || i >= cs) {
                break;
            }
        }
        return res;
    }

    /*
        根据Mat内的数值创建一个对角矩阵，Mat必须为[1,n]数组
    */
    Matrix DIAGONAL(const size_t& _size, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == 1);
        Matrix res(_size, _size);
        size_t i = 0;
        for (i = 0; i < _size && i < _mSize_c; ++i) {
            res.Mat[i][i] = Mat.Mat[0][i];
        }
        return res;
    }

    /*
        根据Mat内的数值创建一个对角矩阵，Mat必须为[1,n]数组
    */
    Matrix DIAGONAL(const size_t& rs, const size_t& cs, const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == 1);
        Matrix res(rs, cs);
        size_t i = 0;
        size_t n2 = std::min(rs, cs);
        for (i = 0; i < n2 && i < _mSize_c; ++i) {
            res.Mat[i][i] = Mat.Mat[0][i];
        }
        return res;
    }

    /*
     * 获得一个[_sx_s]单位对角矩阵
     */
    Matrix EYE(const size_t& _size) {
        Matrix res(_size, _size);
        for (size_t i = 0; i < _size; ++i) {
            res.Mat[i][i] = 1;
        }
        return res;
    }



    Matrix EYE(const size_t& rs, const size_t& cs) {
        Matrix res(rs, cs);
        for (size_t i = 0; i < rs && i < cs; ++i) {
            res.Mat[i][i] = 1;
        }

        return res;
    }

    Matrix RAND_Matrix(const size_t& rs, const size_t& cs, const double& low, const double& high) {
        srand(unsigned(time(0)));
        size_t j = 0;
        Matrix res(rs, cs);
        __m256d m256;
        for (size_t i = 0; i < rs; ++i) {
            for (j = 0; j + 4 <= cs; j += 4) {
                m256 = _mm256_set_pd(
                    rand() % int(high - low + 1) + low,
                    rand() % int(high - low + 1) + low,
                    rand() % int(high - low + 1) + low,
                    rand() % int(high - low + 1) + low);
                _mm256_store_pd(res.Mat[i] + j, m256);
            }
            while (j < cs) {
                res.Mat[i][j] = rand() % int(high - low + 1) + low;
                ++j;
            }
        }
        return res;
    }

    /*
     *  行列式
     *  i,j 控制子矩阵位置
     *  sub_i,sub_j 记录删除行列位置
     */

    double _DETERMINANT(const Matrix& subMat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = subMat.shape();
        assert(_mSize_r == _mSize_c);  // 方阵
        double res = 0.0;
        if (_mSize_r == 1) {
            res = subMat.Mat[0][0];
            return res;
        }
        if (_mSize_r == 2) {
            res = subMat.Mat[0][0] * subMat.Mat[1][1] - subMat.Mat[0][1] * subMat.Mat[1][0];
            return res;
        }

        Matrix new_subMat(_mSize_r - 1, _mSize_c - 1,0);
        for (size_t j = 0; j < _mSize_c; ++j) {
            for (size_t r = 0; r < _mSize_r; ++r) {
                for (size_t c = 0; c < _mSize_c; ++c) {
                    if (r != 0 && c != j) {
                        new_subMat.Mat[r > 0 ? r - 1 : r][c > j ? c - 1 : c] = subMat.Mat[r][c];
                    }
                }
            }
            res += (subMat.Mat[0][j] * (pow(-1, j) * _DETERMINANT(new_subMat)));
        }
        return abs(res - (1e-5)) > 0 ? res : 0;
    }

    /*
     * 伴随矩阵
     */
    Matrix ADJOINT_Matrix(const Matrix& Mat) {
        std::size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        assert(_mSize_r == _mSize_c && _mSize_r > 0);
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
                            sub_Mat.Mat[r > i ? r - 1 : r][c > j ? c - 1 : c] = Mat.Mat[r][c];
                        }
                    }
                }
                res.Mat[i][j] = (pow(-1, i + j) * _DETERMINANT(sub_Mat));
            }
        }
        res.TRANS();
        return res;
    }


    // 特征值分解
    std::tuple<Matrix, Matrix> EigenDec(const Matrix& Mat) {
        return JACOBI(Mat);
    }

    double MAX(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        double tmp[4] = { DMIN,DMIN,DMIN,DMIN };
        __m256d max_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        double res = DMIN;
        size_t j = 0;
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4) {
                max_256 = _mm256_max_pd(_mm256_load_pd(Mat.Mat[i] + j), max_256);
            }
            _mm256_store_pd(tmp, max_256);
            res = std::max(res, std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3])));
            while (j < _mSize_c) {
                res = std::max(res, Mat.Mat[i][j]);
                ++j;
            }
            max_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        }
        return res;
    }

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
     * 雅克比法计算特征值与特征向量
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
        size_t MAX_Iter = size_t(1e4);     // 控制迭代次数
        size_t Iter = 0;
        double dbangle, sintheta, costheta;
        double precision = 1e-10;  // 控制精度

        while (Iter < MAX_Iter)
        {
            // 寻找非对角线元素最大值，及位置
            double non_dia_max_value_abs = abs(copy_Mat[0][1]);
            size_t non_dia_max_row = 0, non_dia_max_col = 1;
            for (size_t i = 0; i < _mSize_r; ++i) {
                for (size_t j = 0; j < _mSize_c; ++j) {
                    if (i != j && abs(copy_Mat.Mat[i][j]) > non_dia_max_value_abs)
                    {
                        non_dia_max_row = i;
                        non_dia_max_col = j;
                        non_dia_max_value_abs = abs(copy_Mat.Mat[i][j]);
                    }
                }
            }

            // 检车是否需要退出循环
            if (non_dia_max_value_abs < precision) {
                break;
            }

            // 计算旋转矩阵
            if (copy_Mat.Mat[non_dia_max_col][non_dia_max_col] == copy_Mat.Mat[non_dia_max_row][non_dia_max_row]) {
                dbangle = PI / 4;
            }
            else {
                dbangle = 0.5 * atan2(2 * copy_Mat[non_dia_max_row][non_dia_max_col],
                    copy_Mat.Mat[non_dia_max_col][non_dia_max_col] - copy_Mat.Mat[non_dia_max_row][non_dia_max_row]);
            }

            sintheta = sin(dbangle);
            costheta = cos(dbangle);

            // 计算特征向量 ,givens rotation Matrix
            U.Mat[non_dia_max_row][non_dia_max_row] = costheta;
            U.Mat[non_dia_max_row][non_dia_max_col] = -sintheta;
            U.Mat[non_dia_max_col][non_dia_max_row] = sintheta;
            U.Mat[non_dia_max_col][non_dia_max_col] = costheta;
            UT = &TRANSPOSITION(U);
            copy_Mat = &(U * copy_Mat * UT);

            EigenVector = &(EigenVector * (UT));
            fast_copy(EigenVector, EigenVector * (UT));
            U.Mat[non_dia_max_row][non_dia_max_row] = 1;
            U.Mat[non_dia_max_row][non_dia_max_col] = 0;
            U.Mat[non_dia_max_col][non_dia_max_row] = 0;
            U.Mat[non_dia_max_col][non_dia_max_col] = 1;

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
            _max_value = EigenValue.Mat[i][0];
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                if (_max_value < EigenValue.Mat[j][0]) {
                    _max_index = j;
                    _max_value = EigenValue.Mat[j][0];
                }
                if (abs(EigenVector.Mat[i][j]) < precision) {
                    EigenVector.Mat[i][j] = 0;
                }
            }
            if (abs(EigenValue.Mat[i][0]) < precision) {
                EigenValue.Mat[i][0] = 0;
            }
            tmp = EigenValue.Mat[i][0];
            std::swap(EigenValue.Mat[i][0], EigenValue.Mat[_max_index][0]);
            for (size_t k = 0; _max_index != i && k < _mSize_r; ++k) {
                std::swap(EigenVector.Mat[k][i], EigenVector.Mat[k][_max_index]);
            }
        }
        return std::make_tuple(EigenValue, EigenVector);
    }


    /*
     *  返回一对特征值与特征向量
     */
    std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta = 1e-5, const size_t& max_iter = 1e3) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == _mSize_c); // row_size,col_size

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

    Matrix SQRT(const Matrix& Mat) {
        Matrix dst(Mat);
        size_t _mSize_r, _mSize_c;
        size_t j, i;

        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        for (i = 0; i < _mSize_r; ++i) {
            for (j = 0; j + 4 <= _mSize_c; j += 4)
            {
                _mm256_store_pd(dst.Mat[i] + j,
                    _mm256_sqrt_pd(_mm256_load_pd(Mat.Mat[i] + j)));
            }
            while (j < _mSize_c) {
                dst.Mat[i][j] = std::sqrt(Mat.Mat[i][j]);
                ++j;
            }
        }
        return dst;
    }

    // 拷贝赋值
    /*
        此函数功能类似与share_ptr中的reset(ELE)函数。
    */
    bool Matrix::memAsycEqual(const Matrix& Mat) {
        // 指向同一内存地址
        if (this->Mat == Mat.Mat) {
            return true;
        }
        
        // 若两个矩阵大小一致，则使用函数fast_copy();
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
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat.Mat[i] + j));
            }
            while (j < new_Size_c) {
                *(tmp + j) = Mat.Mat[i][j];
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
        快速拷贝，必须确保dst的矩阵行列数大于src
    */
    bool fast_copy(Matrix& dst,const Matrix &src) {

        size_t m_Size_r, m_Size_c;
        std::tie(m_Size_r, m_Size_c) = dst.shape();

        for (size_t i = 0; i < m_Size_r; ++i) {
            std::copy(src.Mat[i], src.Mat[i] + m_Size_c, dst.Mat[i]);
            
        }
        return true;
    }


    // LU分解
    std::tuple<Matrix,Matrix> LU(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r > 0 && _mSize_r == _mSize_c);

        Matrix L = EYE(_mSize_r);
        Matrix U(Mat);
        
        for (size_t i = 0; i < _mSize_r; ++i) {
            //U中 i列下方元素变为0；
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                L[j][i] = U[j][i] / U[i][i];
                double ele = U[j][i] / U[i][i];
                for (size_t k = 0; k < _mSize_c; ++k) {
                    U[j][k] = U[j][k] - U[i][k] * ele;
                }
                
            }
        }
        return std::make_tuple(L,U);
    }

    std::tuple<Matrix, Matrix> LU2(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r > 0 && _mSize_r == _mSize_c);

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

    

    void row_swap(Matrix& Mat, size_t i, size_t ii) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        for (size_t j = 0; j < _mSize_c; ++j) {
            std::swap(Mat[i][j], Mat[ii][j]);
        }
    }

    void row_swap_PLU(Matrix& Mat, size_t i, size_t ii,size_t col_index,bool Left=true) {
        
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

    std::tuple<Matrix, Matrix, Matrix> PLU(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
       // assert(_mSize_r > 0 && _mSize_r == _mSize_c);

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


    // 2范数,[1,n]
    double Norm_2(const Matrix& Mat) {
        double res = 0;
        Matrix ATA = TRANSPOSITION(Mat) * Mat;
        std::tie(res, std::ignore) = Power_Method(ATA);
        return sqrt(res);
    }

   
    /*
        格里姆施密特正交法
    */
    std::tuple<Matrix,Matrix> QR(const Matrix& Mat) {
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix Q(_mSize_r, _mSize_c);
        Matrix R(_mSize_c, _mSize_c);
        Matrix q(_mSize_r, 1);
        Matrix y(_mSize_r, 1);
        Matrix yT;
        size_t j, i;
        double norm_2 = 0;
        for (j = 0; j < _mSize_c; ++j) {
            // y = A[][j]
            y = Mat(-1, j);
            // q(i)
            
            for (i = 0; j>0 && i < j; ++i) {
                q = Q(-1, i);
                yT = &(TRANSPOSITION(q) * y);
                R[i][j] = yT[0][0];
                y -= DOT(q, R[i][j]); //y -= (q * R[i][j]);
            }
            norm_2 = Norm_2(y);
            R[j][j] = norm_2;
            q = y / norm_2;
            Q.part_set(q, 0, -1, j, j + 1);
        }
        return std::make_tuple(Q, R);
    }
    
}

#endif //LALIB_MATRIX_H
