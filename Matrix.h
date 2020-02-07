//
// created by zhixu on 2020/2/5.
//

#ifndef lalib_Matrix_h
#define lalib_Matrix_h

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
    constexpr double DMIN = std::numeric_limits<double>::min();  // double min

    class Matrix;
    // ��¼��ǰ����������Ϣ
    class Matrixdsec {
    public:
        Matrixdsec(size_t rs, size_t cs, size_t uc) {
            row_size = rs;
            col_size = cs;
            use_counter = uc;
        }
        
        size_t row_size = 0;
        size_t col_size = 0;
        size_t use_counter = 0;   //����Ϊָ�봫��ʱ����¼����������
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
        friend Matrix operator*(const Matrix& Mat, const Matrix& Mat2);   // ���
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
        Matrix() { this->MDesc = new Matrixdsec(0, 0, 1); this->MDesc->memAsyc.push_back(this); };
        Matrix(const Matrix& Mat);
        Matrix& operator=(const Matrix& Mat);
        Matrix& operator=(Matrix* Mat);
        void operator+=(const Matrix& Mat);
        void operator-=(const Matrix& Mat);
        void operator*=(const Matrix& Mat);
        double* operator[](const size_t& index);
        void clear();
        bool memAsycEqual(const Matrix& Mat);
        int use_count()const;

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

    Matrix::~Matrix() {
        clear();
    }
    

    /*
        ���ؾ����ڴ汻ʹ������
    */
    int Matrix::use_count()const {
        if (this->MDesc == NULL) {
            return -1;
        }
        return this->MDesc->use_counter;
    }


    /*
        ��usecounter����Ϊ0�����ͷ������ڴ�;���򣬽���ָ����ΪNULL
    */
    void Matrix::clear() {
        //std::cout << "ready clear!\n";
        if (this->MDesc != NULL) {
            std::cout << "clear all!\n";
            --MDesc->use_counter;
            if (MDesc->use_counter == 0) {
                //std::cout << "self clear\n";
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

    // ������ֵ
    Matrix& Matrix::operator=(const Matrix& Mat) {

        if (this->Mat == Mat.Mat) {
            return *this;
        }

        if (this->MDesc==NULL || this->MDesc->use_counter <=1) {
            clear();
            // �ڴ�ΪNULL
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
            this->memAsycEqual(Mat);
        }
        
        return *this;
    }

    // ָ��
    Matrix& Matrix::operator=(Matrix* Mat) {
        if (this->Mat == Mat->Mat) {
            return *this;
        }
        Mat->MDesc->use_counter += 1;
        clear(); // ����usecounter
        this->Mat = Mat->Mat;
        this->MDesc = Mat->MDesc;
        this->MDesc->memAsyc.push_back(this);
       
        return *this;
    }


    /*
     * ���
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
                dst.Mat[i][j] = Mat.Mat[i][j] - Mat2.Mat[i][j];
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

        // �����ڴ沢��ʼ��
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
        �Կ�����ֵ�Ĺ��캯��
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
     * ���
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


    double* Matrix::operator[](const size_t& index) {
        return this->Mat[index];
    }

    /*
     * ת�þ�������
     */
    Matrix& Matrix::TRANS() {
        *this = &TRANSPOSITION(*this);
        return *this;
    }

    Matrix TRANSPOSITION(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(_mSize_r, _mSize_c);
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
     * �������
     */
    Matrix INVERSE(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        assert(_mSize_r > 0 && _mSize_r == _mSize_c);
        return ADJOINT_Matrix(Mat) / DETERMINANT(Mat);
    }

    /*
     * ����ʽ
     */
    double DETERMINANT(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == _mSize_c && _mSize_r > 0);
        return _DETERMINANT(Mat);
    }

    /*
     * ����Mat�Խ���Ԫ��
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
        ��src����ֵ����һ��[_size ,_size]�ԽǾ���
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
        ��src����ֵ����һ��[rs,cs]�ԽǾ���
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
        ����Mat�ڵ���ֵ����һ���ԽǾ���Mat����Ϊ[1,n]����
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
        ����Mat�ڵ���ֵ����һ���ԽǾ���Mat����Ϊ[1,n]����
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
     * ���һ��[_sx_s]��λ�ԽǾ���
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
     *  ����ʽ
     *  i,j �����Ӿ���λ��
     *  sub_i,sub_j ��¼ɾ������λ��
     */

    double _DETERMINANT(const Matrix& subMat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = subMat.shape();
        assert(_mSize_r == _mSize_c);  // ����
        double res = 0.0;
        if (_mSize_r == 1) {
            res = subMat.Mat[0][0];
            return res;
        }
        if (_mSize_r == 2) {
            res = subMat.Mat[0][0] * subMat.Mat[1][1] - subMat.Mat[0][1] * subMat.Mat[1][0];
            return res;
        }

        Matrix new_subMat(_mSize_r - 1, _mSize_c - 1);
        for (size_t j = 0; j < _mSize_c; ++j) {
            for (size_t r = 0; r < _mSize_r; ++r) {
                for (size_t c = 0; c < _mSize_c; ++c) {
                    if (_mSize_r != 0 && _mSize_r != _mSize_c) {
                        new_subMat.Mat[r > 0 ? r - 1 : r][c > j ? c - 1 : c] = subMat.Mat[r][c];
                    }
                }
            }
            res += (subMat.Mat[0][j] * (pow(-1, j) * _DETERMINANT(new_subMat)));
        }
        return res;
    }

    /*
     * �������
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


    // ����ֵ�ֽ�
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
     * �ſ˱ȷ���������ֵ����������
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
        size_t MAX_Iter = size_t(1e4);     // ���Ƶ�������
        size_t Iter = 0;
        double dbangle, sintheta, costheta;
        double precision = 1e-10;  // ���ƾ���

        while (Iter < MAX_Iter)
        {
            // Ѱ�ҷǶԽ���Ԫ�����ֵ����λ��
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

            // �쳵�Ƿ���Ҫ�˳�ѭ��
            if (non_dia_max_value_abs < precision) {
                break;
            }

            // ������ת����
            if (copy_Mat.Mat[non_dia_max_col][non_dia_max_col] == copy_Mat.Mat[non_dia_max_row][non_dia_max_row]) {
                dbangle = PI / 4;
            }
            else {
                dbangle = 0.5 * atan2(2 * copy_Mat[non_dia_max_row][non_dia_max_col],
                    copy_Mat.Mat[non_dia_max_col][non_dia_max_col] - copy_Mat.Mat[non_dia_max_row][non_dia_max_row]);
            }

            sintheta = sin(dbangle);
            costheta = cos(dbangle);

            // ������������ ,givens rotation Matrix
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

        // ��������ֵ
        EigenValue = &DIAGONAL(copy_Mat);
        // ����
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
     *  ����һ������ֵ����������
     */
    std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta = 1e-5, const size_t& max_iter = 1e3) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        assert(_mSize_r == _mSize_c); // row_size,col_size

        Matrix X = Matrix(_mSize_r, 1, 1);   // ��������
        Matrix Y;
        double m = 0, pre_m = 0;   //����ֵ
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

    // ������ֵ
    /*
        �˺�������������share_ptr�е�reset(ELE)������
    */
    bool Matrix::memAsycEqual(const Matrix& Mat) {
        // ָ��ͬһ�ڴ��ַ
        if (this->Mat == Mat.Mat) {
            return true;
        }
        
        // �����������Сһ�£���ʹ�ú���fast_copy();
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
        ���ٿ���������ȷ��dst�ľ�������������src
    */
    bool fast_copy(Matrix& dst,const Matrix &src) {

        size_t m_Size_r, m_Size_c;
        std::tie(m_Size_r, m_Size_c) = dst.shape();

        for (size_t i = 0; i < m_Size_r; ++i) {
            std::copy(src.Mat[i], src.Mat[i] + m_Size_c, dst.Mat[i]);
            
        }
        return true;
    }

}

#endif //lalib_Matrix_h
