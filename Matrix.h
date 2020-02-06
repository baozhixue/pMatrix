//
// Created by zhixu on 2020/2/5.
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


namespace bzx {

    constexpr double PI = 3.1415926;
    constexpr double DMIN = std::numeric_limits<double>::min();

    // 记录当前矩阵描述信息
    class MatrixDsec {
    public:
        MatrixDsec(size_t rs, size_t cs, size_t uC) {
            row_size = rs;
            col_size = cs;
            use_Counter = uC;
        }
        size_t row_size = 0;
        size_t col_size = 0;
        size_t use_Counter = 0;   //当作为指针传递时，记录被引用数量
    };


    class Matrix {
        // double version
        friend std::ostream& operator<<(std::ostream& out, const Matrix& M);
        friend Matrix operator+(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix operator-(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix operator/(const Matrix& Mat, const double& num);
        friend Matrix SQRT(const Matrix& Mat);
        friend double MAX(const Matrix& Mat);
        friend Matrix operator*(const Matrix& Mat, const Matrix& Mat2);   // 叉乘
        friend Matrix dot(const Matrix& Mat, const Matrix& Mat2);
        friend Matrix dot(const Matrix& Mat, const double& num);
        friend Matrix inverse(const Matrix& Mat);
        friend Matrix adjoint_matrix(const Matrix& Mat);
        friend Matrix diagonal(const Matrix& Mat);
        friend Matrix diagonal(const size_t& _size, std::initializer_list<double> src);
        friend Matrix diagonal(const size_t& rs, const size_t& cs, std::initializer_list<double> src);
        friend Matrix diagonal(const size_t& _size, const Matrix& Mat);
        friend Matrix diagonal(const size_t& rs, const size_t& cs, const Matrix& Mat);
        friend Matrix eye(const size_t& _size);
        friend Matrix eye(const size_t& rs, const size_t& cs);
        friend Matrix rand_matrix(const size_t& rs, const size_t& cs, const double& Low, const double& high);
        friend double Determinant(const Matrix& Mat);
        friend double _Determinant(const Matrix& subMat);
        friend Matrix transposition(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix, Matrix> SVD(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix> Jacobi(const Matrix& Mat);
        friend std::tuple<Matrix, Matrix> Eigendecomposition(const Matrix& Mat);
        friend std::tuple<double, Matrix> Power_method(const Matrix& Mat, const double& min_delta, const size_t& max_iter);

    public:
        Matrix(const int _row_size, const int _col_size, const double init_num = 0);
        Matrix(std::initializer_list<std::initializer_list<double >> src);
        Matrix() = default;
        Matrix(const Matrix& Mat);
        Matrix(const Matrix* Mat);
        Matrix& operator=(const Matrix& Mat);
        Matrix& operator=(Matrix* Mat);
        void operator+=(const Matrix& Mat);
        void operator-=(const Matrix& Mat);
        void operator*=(const Matrix& Mat);
        double* operator[](const size_t& index);
        void Clear();
        void pClear();

        Matrix& Trans();
        std::tuple<size_t, size_t> shape() const {
            if (this->MDesc == NULL) {
                return std::make_tuple(0, 0);
            }
            return std::make_tuple(this->MDesc->row_size, this->MDesc->col_size);
        }

        ~Matrix() {
            if (this->MDesc != NULL) {
                --MDesc->use_Counter;
                //std::cout <<MDesc->use_Counter <<" delete!\n";
                if (MDesc->use_Counter == 0) {
                    //std::cout << " auto clear!******\n";
                    for (int i = 0; i < MDesc->row_size; ++i) {
                        delete[]this->M[i];
                        this->M[i] = NULL;
                    }
                    delete[]this->M;
                    delete this->MDesc;
                }
                this->M = NULL;
                this->MDesc = NULL;
            }
        }

        //private:
        double** M = NULL; // 2d
        MatrixDsec* MDesc = NULL;

    };

    /*
        若useCounter计数为0，则释放所属内存;否则，仅将指针置为NULL
    */
    void Matrix::Clear() {
        if (this->MDesc != NULL) {
            --MDesc->use_Counter;
            if (MDesc->use_Counter == 0) {
                //std::cout << "self clear\n";
                for (int i = 0; i < MDesc->row_size; ++i) {
                    delete[]this->M[i];
                    this->M[i] = NULL;
                }
                delete[]this->M;
                delete this->MDesc;
            }
            this->M = NULL;
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

        if (this->M == Mat.M) {
            return *this;
        }
        Clear();
        // 内存为NULL
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        this->MDesc = new MatrixDsec(R, C, 1);

        int i = 0, j = 0;
        double** tmp2, * tmp;
        tmp2 = new double* [R];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }

        for (i = 0; i < R; ++i) {
            tmp = new double[C];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }

            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat.M[i] + j));
            }
            while (j < C) {
                *(tmp + j) = Mat.M[i][j];
                ++j;
            }
            tmp2[i] = tmp;
        }
        this->M = tmp2;
        return *this;
    }

    // 指针
    Matrix& Matrix::operator=(Matrix* Mat) {
        if (this->M == Mat->M) {
            return *this;
        }
        Mat->MDesc->use_Counter += 1;
        Clear(); // 更新useCounter
        this->M = Mat->M;
        this->MDesc = Mat->MDesc;
        return *this;
    }


    /*
     * 叉乘
     */
    Matrix operator*(const Matrix& Mat, const Matrix& Mat2) {
        size_t R, C, R2, C2;
        std::tie(R, C) = Mat.shape();
        std::tie(R2, C2) = Mat2.shape();
        assert(C == R2);

        Matrix Dst(R, C2, 0);
        // version 0.2
        int k = 0;
        __m256d dst_m256d = _mm256_set_pd(0, 0, 0, 0);
        __m256d mat2_m256d;
        double dst_array[4];
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C2; ++j) {
                for (k = 0; k + 4 <= R2; k += 4) {
                    mat2_m256d = _mm256_set_pd(Mat2.M[k + 3][j], Mat2.M[k + 2][j], Mat2.M[k + 1][j], Mat2.M[k][j]);
                    dst_m256d = _mm256_add_pd(dst_m256d, _mm256_mul_pd(_mm256_load_pd(Mat.M[i] + k), mat2_m256d));
                }
                _mm256_store_pd(dst_array, dst_m256d);
                Dst.M[i][j] = dst_array[0] + dst_array[1] + dst_array[2] + dst_array[3];
                while (k < C) {
                    Dst.M[i][j] += (Mat.M[i][k] * Mat2.M[k][j]);
                    ++k;
                }
                dst_m256d = _mm256_set_pd(0, 0, 0, 0);
            }
        }

        return Dst;
    }


    Matrix dot(const Matrix& Mat, const Matrix& Mat2) {
        size_t R, C, R2, C2;
        std::tie(R, C) = Mat.shape();
        std::tie(R2, C2) = Mat2.shape();
        assert(C == R2);
        assert(R == R2);

        Matrix dst(R, C);
        int i = 0, j = 0;
        if (C == C2) {
            for (i = 0; i < R; ++i) {
                for (j = 0; j + 4 <= C; j += 4) {
                    _mm256_store_pd(dst.M[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.M[i] + j), _mm256_load_pd(Mat2.M[i] + j)));
                }
                while (j < C) {
                    dst.M[i][j] = Mat.M[i][j] * Mat2.M[i][j];
                    ++j;
                }
            }
        }
        else {
            assert(C2 == 1);
            for (i = 0; i < R; ++i) {
                __m256d m256d = _mm256_set_pd(Mat2.M[i][0], Mat2.M[i][0], Mat2.M[i][0], Mat2.M[i][0]);
                for (j = 0; j + 4 <= C; j += 4) {
                    _mm256_store_pd(dst.M[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.M[i] + j), m256d));
                }
                while (j < C) {
                    dst.M[i][j] = Mat.M[i][j] * Mat2.M[i][0];
                    ++j;
                }
            }
        }

        return dst;
    }

    Matrix dot(const Matrix& Mat, const double& num) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        Matrix dst(R, C);
        int i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(dst.M[i] + j, _mm256_mul_pd(_mm256_load_pd(Mat.M[i] + j), m256));
            }
            while (j < C) {
                dst.M[i][j] = Mat.M[i][j] * num;
                ++j;
            }
        }
        return dst;
    }



    Matrix operator/(const Matrix& Mat, const double& num) {
        assert(num != 0);
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        Matrix dst(R, C);
        int i = 0, j = 0;
        __m256d m256 = _mm256_set_pd(num, num, num, num);
        for (i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(dst.M[i] + j, _mm256_div_pd(_mm256_load_pd(Mat.M[i] + j), m256));
            }
            while (j < C) {
                dst.M[i][j] = Mat.M[i][j] / num;
                ++j;
            }
        }
        return dst;
    }


    Matrix operator+(const Matrix& Mat, const Matrix& Mat2) {
        assert(Mat.shape() == Mat2.shape());

        size_t R, C;
        std::tie(R, C) = Mat.shape();

        Matrix dst(R, C);

        int i = 0, j = 0;
        for (i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(dst.M[i] + j, _mm256_add_pd(_mm256_load_pd(Mat.M[i] + j), _mm256_load_pd(Mat2.M[i] + j)));
            }
            while (j < C) {
                dst.M[i][j] = Mat.M[i][j] + Mat2.M[i][j];
                ++j;
            }
        }
        return dst;
    }

    Matrix operator-(const Matrix& Mat, const Matrix& Mat2) {
        assert(Mat.shape() == Mat2.shape());
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        Matrix dst(R, C);
        int i = 0, j = 0;
        for (i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(dst.M[i] + j, _mm256_sub_pd(_mm256_load_pd(Mat.M[i] + j), _mm256_load_pd(Mat2.M[i] + j)));
            }
            while (j < C) {
                dst.M[i][j] = Mat.M[i][j] - Mat2.M[i][j];
                ++j;
            }
        }
        return dst;
    }

    Matrix::Matrix(const int _row_size, const int _col_size, const double init_num) {
        Clear();
        this->MDesc = new MatrixDsec(_row_size, _col_size, 1);
        int  i = 0, j = 0;

        // 分配内存并初始化
        double** tmp2 = new double* [_row_size];
        double* tmp;
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->M = tmp2;

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
            this->M[i] = tmp;
        }
    }

    Matrix::Matrix(std::initializer_list<std::initializer_list<double >> src) {
        assert(src.size() > 0);
        assert(src.begin()->size() > 0);

        ////////////////////
        Clear();

        size_t R = src.size(), C = src.begin()->size();
        this->MDesc = new MatrixDsec(R, C, 1);

        int i = 0, j = 0;
        double** tmp2, * tmp;
        tmp2 = new double* [R];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->M = tmp2;

        for (auto row = src.begin(); row != src.end(); ++row) {
            j = 0;
            tmp = new double[C];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }
            for (auto ele = row->begin(); ele != row->end(); ++ele) {
                *(tmp + j) = *ele;
                ++j;
            }
            this->M[i] = tmp;
            ++i;
        }
    }

    /*
        以拷贝赋值
    */
    Matrix::Matrix(const Matrix& Mat) {

        Clear();
        size_t R = Mat.MDesc->row_size;
        size_t C = Mat.MDesc->col_size;
        this->MDesc = new MatrixDsec(R, C, 1);

        int i = 0, j = 0;
        double** tmp2, * tmp;
        tmp2 = new double* [R];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }
        this->M = tmp2;

        for (i = 0; i < R; ++i) {
            tmp = new double[C];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }

            for (j = 0; j + 4 <= C; j += 4) {
                _mm256_store_pd(tmp + j, _mm256_load_pd(Mat.M[i] + j));
            }
            while (j < C) {
                *(tmp + j) = Mat.M[i][j];
                ++j;
            }
            this->M[i] = tmp;
        }
    }

    /*
     * 输出
     */
    std::ostream& operator<<(std::ostream& out, const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                out << Mat.M[i][j] << " ";
            }
            out << "\n";
        }
        out << "\r- size ( " << R << " ," << C << " )\n";
        return out;
    }


    double* Matrix::operator[](const size_t& index) {
        return this->M[index];
    }

    /*
     * 转置矩阵自身
     */
    Matrix& Matrix::Trans() {
        *this = &transposition(*this);
        return *this;
    }

    Matrix transposition(const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        Matrix res(C, R);
        int j = 0;
        for (int i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                res.M[j][i] = Mat.M[i][j];
                res.M[j + 1][i] = Mat.M[i][j + 1];
                res.M[j + 2][i] = Mat.M[i][j + 2];
                res.M[j + 3][i] = Mat.M[i][j + 3];
            }
            while (j < C) {
                res.M[j][i] = Mat.M[i][j];
                ++j;
            }
        }
        return res;
    }


    /*
     * 求逆矩阵
     */
    Matrix inverse(const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();

        assert(R > 0 && R == C);
        return adjoint_matrix(Mat) / Determinant(Mat);
    }

    /*
     * 行列式
     */
    double Determinant(const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        assert(R == C && R > 0);
        return _Determinant(Mat);
    }

    /*
     * 返回Mat对角线元素
     */
    Matrix diagonal(const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        size_t _size = std::min(R, C);
        Matrix res(_size, 1);
        for (int i = 0; i < _size; ++i) {
            res.M[i][0] = Mat.M[i][i];
        }
        return res;
    }

    /*
        以src内数值创建一个[_size ,_size]对角矩阵
    */
    Matrix diagonal(const size_t& _size, std::initializer_list<double> src) {
        Matrix res(_size, _size);
        int i = 0;
        for (auto el : src) {
            res.M[i][i] = el;
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
    Matrix diagonal(const size_t& rs, const size_t& cs, std::initializer_list<double> src) {
        Matrix res(rs, cs);
        int i = 0;
        for (auto el : src) {
            res.M[i][i] = el;
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
    Matrix diagonal(const size_t& _size, const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        assert(R == 1);
        Matrix res(_size, _size);
        int i = 0;
        for (int i = 0; i < _size && i < C; ++i) {
            res.M[i][i] = Mat.M[0][i];
        }
        return res;
    }

    /*
        根据Mat内的数值创建一个对角矩阵，Mat必须为[1,n]数组
    */
    Matrix diagonal(const size_t& rs, const size_t& cs, const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        assert(R == 1);
        Matrix res(rs, cs);
        int i = 0;
        int N2 = std::min(rs, cs);
        for (int i = 0; i < N2 && i < C; ++i) {
            res.M[i][i] = Mat.M[0][i];
        }
        return res;
    }

    /*
     * 获得一个[_sx_s]单位对角矩阵
     */
    Matrix eye(const size_t& _size) {
        Matrix res(_size, _size);
        for (int i = 0; i < _size; ++i) {
            res.M[i][i] = 1;
        }
        return res;
    }

    Matrix eye(const size_t& rs, const size_t& cs) {
        Matrix res(rs, cs);
        for (int i = 0; i < rs && i < cs; ++i) {
            res.M[i][i] = 1;
        }

        return res;
    }

    Matrix rand_matrix(const size_t& rs, const size_t& cs, const double& Low, const double& high) {
        srand(time(0));
        int j = 0;
        Matrix res(rs, cs);
        __m256d m256;
        for (int i = 0; i < rs; ++i) {
            for (j = 0; j + 4 <= cs; j += 4) {
                m256 = _mm256_set_pd(
                    rand() % int(high) + Low,
                    rand() % int(high) + Low,
                    rand() % int(high) + Low,
                    rand() % int(high) + Low);
                _mm256_store_pd(res.M[i] + j, m256);
            }
            while (j < cs) {
                res.M[i][j] = rand() % int(high) + Low;
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

    double _Determinant(const Matrix& subMat) {
        size_t R, C;
        std::tie(R, C) = subMat.shape();
        assert(R == C);  // 方阵
        double res = 0.0;
        if (R == 1) {
            res = subMat.M[0][0];
            return res;
        }
        if (R == 2) {
            res = subMat.M[0][0] * subMat.M[1][1] - subMat.M[0][1] * subMat.M[1][0];
            return res;
        }

        Matrix new_subMat(R - 1, C - 1);
        for (int j = 0; j < C; ++j) {
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < C; ++c) {
                    if (r != 0 && c != j) {
                        new_subMat.M[r > 0 ? r - 1 : r][c > j ? c - 1 : c] = subMat.M[r][c];
                    }
                }
            }
            res += (subMat.M[0][j] * (pow(-1, j) * _Determinant(new_subMat)));
        }
        return res;
    }

    /*
     * 伴随矩阵
     */
    Matrix adjoint_matrix(const Matrix& Mat) {
        std::size_t R, C;
        std::tie(R, C) = Mat.shape();

        assert(R == C && R > 0);
        Matrix res(R, C);
        Matrix sub_mat(R - 1, C - 1);

        if (R == 1) {
            res = { {1} };
            return res;
        }
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {

                for (int r = 0; r < R; ++r) {
                    for (int c = 0; c < C; ++c) {
                        if (r != i && c != j) {
                            sub_mat.M[r > i ? r - 1 : r][c > j ? c - 1 : c] = Mat.M[r][c];
                        }
                    }
                }
                res.M[i][j] = (pow(-1, i + j) * _Determinant(sub_mat));
            }
        }
        res.Trans();
        return res;
    }


    // 特征值分解
    std::tuple<Matrix, Matrix> Eigendecomposition(const Matrix& Mat) {
        return Jacobi(Mat);
    }

    double MAX(const Matrix& Mat) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        double tmp[4] = { DMIN,DMIN,DMIN,DMIN };
        __m256d MAX_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        double res = DMIN;
        int j = 0;
        for (int i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4) {
                MAX_256 = _mm256_max_pd(_mm256_load_pd(Mat.M[i] + j), MAX_256);
            }
            _mm256_store_pd(tmp, MAX_256);
            res = std::max(res, std::max(std::max(tmp[0], tmp[1]), std::max(tmp[2], tmp[3])));
            while (j < C) {
                res = std::max(res, Mat.M[i][j]);
                ++j;
            }
            MAX_256 = _mm256_set_pd(DMIN, DMIN, DMIN, DMIN);
        }
        return res;
    }

    std::tuple<Matrix, Matrix, Matrix> SVD(const Matrix& Mat) {

        Matrix ATA, AAT, tmp, U, Sigma, V;
        tmp = &transposition(Mat);
        ATA = &(tmp * Mat);
        AAT = &(Mat * tmp);
        std::tie(std::ignore, V) = Jacobi(ATA);

        std::tie(Sigma, U) = Jacobi(AAT);

        size_t R, C;
        std::tie(R, C) = Mat.shape();
        Sigma.Trans();
        Sigma = &diagonal(R, C, Sigma);
        Sigma = &SQRT(Sigma);
        //std::cout << *U << *Sigma2 << *V;

        return std::make_tuple(U, Sigma, V);
    }

    /*
     * 雅克比法计算特征值与特征向量
     */
    std::tuple<Matrix, Matrix> Jacobi(const Matrix& Mat) {
        std::size_t R, C;
        std::tie(R, C) = Mat.shape();

        Matrix EigenVector;
        EigenVector = &eye(R, C);
        Matrix U;
        U = &eye(R, C);
        Matrix UT;
        Matrix EigenValue(R, 1);
        Matrix copy_Mat = Mat;
        size_t max_iter = 1e4;
        size_t iter = 0;
        double dbAngle, sinTheta, cosTheta;
        double precision = 1e-10;

        while (iter < max_iter)
        {
            // 寻找非对角线元素最大值，及位置
            double Non_Dia_Max_Value_abs = abs(copy_Mat[0][1]);
            size_t Non_Dia_Max_row = 0, Non_Dia_Max_col = 1;
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    if (i != j && abs(copy_Mat.M[i][j]) > Non_Dia_Max_Value_abs)
                    {
                        Non_Dia_Max_row = i;
                        Non_Dia_Max_col = j;
                        Non_Dia_Max_Value_abs = abs(copy_Mat.M[i][j]);
                    }
                }
            }

            // 检车是否需要退出循环
            if (Non_Dia_Max_Value_abs < precision) {
                break;
            }

            // 计算旋转矩阵
            if (copy_Mat.M[Non_Dia_Max_col][Non_Dia_Max_col] == copy_Mat.M[Non_Dia_Max_row][Non_Dia_Max_row]) {
                dbAngle = PI / 4;
            }
            else {
                dbAngle = 0.5 * atan2(2 * copy_Mat[Non_Dia_Max_row][Non_Dia_Max_col],
                    copy_Mat.M[Non_Dia_Max_col][Non_Dia_Max_col] - copy_Mat.M[Non_Dia_Max_row][Non_Dia_Max_row]);
            }

            sinTheta = sin(dbAngle);
            cosTheta = cos(dbAngle);

            // 计算特征向量 ,Givens rotation matrix
            U.M[Non_Dia_Max_row][Non_Dia_Max_row] = cosTheta;
            U.M[Non_Dia_Max_row][Non_Dia_Max_col] = -sinTheta;
            U.M[Non_Dia_Max_col][Non_Dia_Max_row] = sinTheta;
            U.M[Non_Dia_Max_col][Non_Dia_Max_col] = cosTheta;
            UT = &transposition(U);
            copy_Mat = &(U * copy_Mat * UT);

            EigenVector = &(EigenVector * (UT));
            U.M[Non_Dia_Max_row][Non_Dia_Max_row] = 1;
            U.M[Non_Dia_Max_row][Non_Dia_Max_col] = 0;
            U.M[Non_Dia_Max_col][Non_Dia_Max_row] = 0;
            U.M[Non_Dia_Max_col][Non_Dia_Max_col] = 1;

            ++iter;
        }

        // 计算特征值
        EigenValue = &diagonal(copy_Mat);
        // 排序
        double _MAX_VALUE;
        size_t _MAX_index;
        double tmp;
        std::tie(R, C) = EigenValue.shape();
        for (int i = 0; i < R; ++i) {
            _MAX_index = i;
            _MAX_VALUE = EigenValue.M[i][0];
            for (int j = i + 1; j < R; ++j) {
                if (_MAX_VALUE < EigenValue.M[j][0]) {
                    _MAX_index = j;
                    _MAX_VALUE = EigenValue.M[j][0];
                }
                if (abs(EigenVector.M[i][j]) < precision) {
                    EigenVector.M[i][j] = 0;
                }
            }
            if (abs(EigenValue.M[i][0]) < precision) {
                EigenValue.M[i][0] = 0;
            }
            tmp = EigenValue.M[i][0];
            std::swap(EigenValue.M[i][0], EigenValue.M[_MAX_index][0]);
            for (int k = 0; _MAX_index != i && k < R; ++k) {
                std::swap(EigenVector.M[k][i], EigenVector.M[k][_MAX_index]);
            }
        }
        return std::make_tuple(EigenValue, EigenVector);
    }


    /*
     *  返回一对特征值与特征向量
     */
    std::tuple<double, Matrix> Power_method(const Matrix& Mat, const double& min_delta = 1e-5, const size_t& max_iter = 1e3) {
        size_t R, C;
        std::tie(R, C) = Mat.shape();
        assert(R == C); // ROW_SIZE,COL_SIZE

        Matrix X = Matrix(R, 1, 1);
        Matrix Y;
        double M = 0, pre_M = 0;
        size_t iter = 0;
        double delta = INT32_MAX;
        while (iter < max_iter && delta >min_delta) {
            Y = &(Mat * (X));
            M = MAX(Y);
            delta = abs(M - pre_M);
            pre_M = M;
            X = &((Y) / M);
            iter += 1;
        }
        return std::make_tuple(M, X);
    }

    Matrix SQRT(const Matrix& Mat) {
        Matrix dst(Mat);
        size_t R, C, j, i;

        std::tie(R, C) = Mat.shape();
        for (i = 0; i < R; ++i) {
            for (j = 0; j + 4 <= C; j += 4)
            {
                _mm256_store_pd(dst.M[i] + j,
                    _mm256_sqrt_pd(_mm256_load_pd(Mat.M[i] + j)));
            }
            while (j < C) {
                dst.M[i][j] = std::sqrt(Mat.M[i][j]);
                ++j;
            }
        }
        return dst;
    }
}

#endif //LALIB_MATRIX_H
