/*
    created by baozhixue
    email: baozhixue@hnu.edu.cn
*/

#ifndef LALIB_UMATRIX_H
#define LALIB_UMATRIX_H

#include <iostream>
#include <memory>
#include <random>
#include <ctime>
#include <tuple>
#include <string>
#include <list>
#include <cmath>

namespace ubzx {

    constexpr double PI = 3.1415926;
    constexpr double DMIN = std::numeric_limits<double>::min();  // double min
    constexpr double DMAX = std::numeric_limits<double>::max();
    constexpr double PRECISION = 1e-15;   // JACOBI
    constexpr double DETERMINANT_PRECISION = 1e-8;
    constexpr double MIN_DELTA = 1e-7;  // Power_Method
    constexpr size_t MAX_ITER = int(1e6);   // JACOBI,Power_Method
    constexpr double OUT_PRECISION = 1e-8;  // ���ʱ������ֵС�ڴ�ֵ�������Ϊ0��

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
        std::vector<Matrix*> memAsyc;  //ȫ�־���ĸ���
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
    Matrix operator||(const Matrix& Mat, const Matrix& Mat2);
    Matrix operator&&(const Matrix& Mat, const Matrix& Mat2);

    std::ostream& operator<<(std::ostream& out, const Matrix& m); //���
    Matrix operator*(const Matrix& Mat, const Matrix& Mat2);   // ���

    std::tuple<double ,double >Givens(const double &a, const double &b);
    Matrix SQRT(const Matrix& Mat);  
    double MAX(const Matrix& Mat);
    double MIN(const Matrix& Mat);
    Matrix DOT(const Matrix& Mat, const Matrix& Mat2);
    Matrix DOT(const Matrix& Mat, const double& num);
    Matrix Inverse(const Matrix& Mat);
    Matrix Adjoint(const Matrix& Mat);
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
    double Det(const Matrix& Mat);
    std::tuple<size_t, Matrix> Rank(const Matrix& Mat);
    std::tuple<size_t,Matrix> EleTrans(const Matrix& Mat);
    Matrix Trans(const Matrix& Mat);
    std::tuple<double, Matrix> Power_Method(const Matrix& Mat, const double& min_delta = MIN_DELTA, const size_t& max_iter= MAX_ITER);
    std::tuple<Matrix, Matrix> QR(const Matrix& Mat);
    double Norm_2(const Matrix& Mat);
    std::tuple<Matrix, Matrix, Matrix> PLU(const Matrix& Mat);
    std::tuple<Matrix, Matrix> LU(const Matrix& Mat);
    Matrix Ceil(const Matrix& Mat);
    Matrix Floor(const Matrix& Mat);
    Matrix MMul(std::initializer_list <Matrix> Mats);
    // ���ܺ���
    bool fast_copy(Matrix& dst, const Matrix& src);
    void row_swap_PLU(Matrix& Mat, size_t i, size_t ii, size_t col_index, bool Left = true);
    void row_swap(Matrix& Mat, size_t i, size_t ii);


    /*
        ˫����double����       
    */
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
        Matrix& trans();
        std::tuple<size_t, size_t> shape() const;
        void CEIL() { *this = Ceil(*this); };
        void FLOOR() { *this = Floor(*this); };
        bool Resize(const size_t &nrs,const size_t &ncs);
        bool Equal(const Matrix &Mat);
        ~Matrix();
    private:
        double** Mat = NULL; // 2d
        Matrixdsec* MDesc = NULL;

    };

    /*
        ���ؾ�����к���
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

    bool Matrix::Resize(const size_t &nrs,const size_t &ncs){
        size_t olR,olC;
        std::tie(olR,olC) = this->shape();
        size_t copy_length = std::min(ncs,olC);
        double **tmp2 = new double*[nrs];
        for(size_t i = 0;i<nrs;++i){
            tmp2[i] = new double [ncs]{0};
            if(i<olR){
                std::copy(this->Mat[i],this->Mat[i]+ copy_length,tmp2[i]);
                delete this->Mat[i];
            }
        }
        this->Mat = tmp2;
        this->MDesc->row_size = nrs;
        this->MDesc->col_size = ncs;
        for (size_t i = 0; i < this->MDesc->memAsyc.size(); ++i) {
            this->MDesc->memAsyc[i]->Mat = tmp2;
        }
        return true;
    }

    bool Matrix::Equal(const Matrix &Mat){
        if(this->shape() != Mat.shape()){
            return false;
        }
        size_t rs,cs;
        std::tie(rs,cs) = this->shape();
        for(size_t i=0;i<rs;++i){
            for(size_t j=0;j<cs;++j){
                if(abs(this->Mat[i][j] != Mat[i][j]) > 1e7){
                    return false;
                }
            }
        }

        return true;
    }


    /*
        ���ؾ����ڴ汻ʹ������,
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
        *this = (*this + Mat);
    }
    void Matrix::operator-=(const Matrix& Mat) {
        *this = (*this - Mat);
    }
    void Matrix::operator*=(const Matrix& Mat) {
        *this = (*this * Mat);
    }

    /*
     * @ Purpose  ��������ֵ
     */
    Matrix& Matrix::operator=(const Matrix& Mat) {

        if (this->Mat == Mat.Mat) {
            return *this;
        }

        //����ָ���µ��ڴ�
        if (this->MDesc==NULL || this->MDesc->use_counter <=1) {
            if (this->shape() == Mat.shape()) {
                fast_copy(*this, Mat);
                return *this;
            }
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

//                j = 0;
//                while (j < _mSize_c) {
//                    *(tmp + j) = Mat[i][j];
//                    ++j;
//                }
                std::copy(Mat[i],Mat[i]+_mSize_c,tmp);
                tmp2[i] = tmp;
            }
            this->Mat = tmp2;
            this->MDesc->memAsyc.push_back(this);
        }
        else {
            // ��������ָ����ָ�ĵ�ַ
            this->memAsycEqual(Mat);
        }
        
        return *this;
    }

    /*
     * @ Purpose  ָ����¸�ֵ
     */
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
     * @ Purpose :����Mat��Mat2�ľ�����
     * @ Return : �������  
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
        Matrix Mat2T(Mat2);
        Mat2T.trans();
#pragma omp parallel for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _m2Size_c; ++j) {
                for(size_t k = 0;k< _mSize_c;++k){
                    dst[i][j] += (Mat[i][k] * Mat2T[j][k]);
                }
            }
        }

        return dst;
    }

    /*
        @ Purpose : ������������ĵ��
        @ Return  �� ����
        @ Other ��Mat2��Mat��������һ�£���Mat2��������Mat����һ��������Ϊ1
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

        Matrix dst(Mat);
        size_t i = 0, j = 0;
        if (_m2Size_c == _mSize_c) {
            for (size_t j = 0; j < _mSize_c; ++j){
                dst[i][j] *= Mat2[i][j];
            }
        }
        else {
            if (_m2Size_c != 1) {
                std::cerr << "in Dot,if col_size is not equal,then second Mat.col_size need equal 1.\n";
                return Matrix();
            }
            for (i = 0; i < _mSize_r; ++i) {
                for (size_t j = 0; j < _mSize_c; ++j){
                    dst[i][j] *= Mat2[i][0];
                }
            }
        }

        return dst;
    }

    /*
        @ Purpose : �����������ֵ�ĵ��
        @ Return  �� ����
    */
    Matrix DOT(const Matrix& Mat, const double& num) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(Mat);
        size_t i = 0, j = 0;
        for (i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j<_mSize_c;++j){
                dst[i][j] *= num;
            }
        }
        return dst;
    }


    /*
        @ Purpose :�������һ����ֵ
        @ Return :����
    */
    Matrix operator/(const Matrix& Mat, const double& num) {
        if (num == 0) {
            std::cerr << "in div(/),every ele must not equal 0.\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(Mat);
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _mSize_c; ++j){
                dst[i][j] /=num;
            }
        }
        return dst;
    }

    /*
        @ Purpose :����ӷ�
    */
    Matrix operator+(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape())
        {
            std::cerr << "in add(+) two Mat's shape must equal!\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(Mat);
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _mSize_c; ++j){
                dst[i][j] += Mat2[i][j];
            }
        }
        return dst;
    }

    /*
        @ Purpose :�������
    */
    Matrix operator-(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape())
        {
            std::cerr << "in sub(-) two Mat's shape must equal!\n";
            return Matrix();
        }
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix dst(Mat);
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j = 0; j < _mSize_c; ++j){
                dst[i][j] -= Mat2[i][j];
            }
        }
        return dst;
    }

    /*
        @ Purpose :����һ��[_row_size,_col_size]����init_numΪ��ʼֵ�ľ���
    */
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

        for (i = 0; i < _row_size; ++i) {
            tmp = new double[_col_size]{init_num};
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }
            j = 0;
            while (j < _col_size) {
                *(tmp + j) = init_num;
                ++j;
            }
            this->Mat[i] = tmp;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
        @ Purpose : ����src����һ���µľ���
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
        @ Purpose :�Կ�����ֵ�Ĺ��캯��
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
//            j = 0;
//            while (j < _mSize_c) {
//                *(tmp + j) = Mat[i][j];
//                ++j;
//            }
            std::copy(Mat[i],Mat[i]+_mSize_c,tmp);
            this->Mat[i] = tmp;
        }
        this->MDesc->memAsyc.push_back(this);
    }

    /*
     * @ Purpose : ��׼���
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
                out << Mat[i][j] << " ";
            }
            if (i + 1 == _mSize_r) {
                out << "] , (" << _mSize_r << ", " << _mSize_c << ")";
            }
            out << "\n";
        }
        return out;
    }

    /*
        @ Purpose : ���ݴ������Mat����[Low_r,High_r)->[Low_c,High_c)λ�õ���ֵ
        @ Para  :
                    Mat 
                    [Low_r,High_r]->[Low_c,High_c] ׼�����µ�λ��
        @ Example :
                    Matrix A = {{1,2}};
                    Matrix B = {{5,5,3},{4,5,6}};
                    B.part_set(A,0,1,0,2);  // ��ʱB����Ϊ{{1,2,3},{4,5,6}}
    */
    bool Matrix::part_set(const Matrix& Mat, const size_t _Low_r, const size_t &_High_r, const size_t &_Low_c, const size_t &_High_c) {
        size_t High_c, High_r;

        std::tie(High_r, High_c) = this->shape();
        if (_High_c != -1) {
            High_c = _High_c;
        }
        if (_High_r != -1) {
            High_r = _High_r;
        }
        size_t copy_range_r,copy_range_c;
        std::tie(copy_range_r,copy_range_c) = Mat.shape();
        copy_range_c = std::min(copy_range_c,High_c-_Low_c);
        for(size_t i = _Low_r,i2=0; i < High_r && i2<copy_range_r;++i,++i2){
            std::copy(Mat.Mat[i2],Mat.Mat[i2]+copy_range_c,this->Mat[i]+_Low_c);
        }

        return true;
    }

    /*
        @ Purpose : ��ȡ�����ĳ�л�ĳ��
        @ Para  :
                r, ���˲���Ϊ-1����ȡ��Ӧ��c��
                c�����˲���Ϊ-1����ȡ��Ӧ��r��
        @ Other �� r��c����һ����������Ϊ-1
        @ Return ��ĳ�л�ĳ�еĿ���
        @ Example��
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
//            for (size_t i = 0; i < _mSize_c; ++i) {
//                child[0][i] = this->Mat[r][i];
//            }
            std::copy(this->Mat[r],this->Mat[r]+_mSize_c,child.Mat[0]);
        }

        return child;
    }

    /*
        @ Purpose ��������תΪstring����
        @ Return �� string
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
       @ Purpose  : ��������������ֵ
    */
    double* Matrix::operator[](const size_t& index) {
        return this->Mat[index];
    }
    /*
        @ Purpose  : ��������������ֵ
        @ Other : ��������const����ʱ��Ҫʹ�ô˺���
    */
    double* Matrix::operator[](const size_t& index) const {
        return this->Mat[index];
    }


    /*
        @ Purpose  : ���ص�ǰ������Ӿ���
        @ Other    ��Low_r,Low_c����С�ڵ�ǰ��������������
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
     * @ Purpose  :ת�þ�������
     */
    Matrix& Matrix::trans() {
        *this = Trans(*this);
        return *this;
    }

    Matrix Trans(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix res(Mat);
        double range = double (_mSize_r) - double (_mSize_c);
        range = abs(range);
        if(range/double (_mSize_r + _mSize_c) < 0.9){
            size_t ns = std::max(_mSize_r, _mSize_c);
            res.Resize(ns,ns);
#pragma omp parallel for
            for (size_t i = 0; i < ns; ++i) {
                for (size_t  j = i; j < ns; ++j) {
                    std::swap(res[i][j],res[j][i]);
                }
            }
            res.Resize(_mSize_c,_mSize_r);
        }
        else{
            res = Matrix(_mSize_c,_mSize_r);
            for (size_t i = 0; i < _mSize_r; ++i) {
                for (size_t  j = 0; j < _mSize_c; ++j) {
                    res[j][i] = Mat[i][j];
                }
            }
        }
        return res;
    }


    /*
     * @ Purpose  : �������
     */
    Matrix Inverse(const Matrix &Mat){
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix tmp(Mat);
        Matrix E = EYE(_mSize_r, _mSize_c);
        tmp.Resize(_mSize_r,2*_mSize_c);
        tmp.part_set(E,0,-1,_mSize_c,-1);
        std::tie(std::ignore,tmp) = EleTrans(tmp);
        size_t ns = std::min(_mSize_c,_mSize_r);
        double det = 1.0;
        for(size_t i = 0;i<ns;++i){
            det *= tmp[i][i];
        }
        if(abs(det) <= 1e-20){
            std::cerr<<"mat's det = 0!\n";
        }
        tmp = tmp(0,-1,_mSize_c,-1);
        return tmp;
    }


    /*
     * @ Purpose  :����Mat�Խ���Ԫ��
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
        @ Purpose  :��src����ֵ����һ��[_size ,_size]�ԽǾ���
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
       @ Purpose  : ��src����ֵ����һ��[rs,cs]�ԽǾ���
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
        @ Purpose  :����Mat�ڵ���ֵ����һ���ԽǾ���Mat����Ϊ[1,n]����
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
        @ Purpose  :����Mat�ڵ���ֵ����һ���ԽǾ���Mat����Ϊ[1,n]����
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
     * @ Purpose  :���һ��[_size,_size]��С�ĵ�λ�ԽǾ���
     */
    Matrix EYE(const size_t& _size) {
        Matrix res(_size, _size);
        for (size_t i = 0; i < _size; ++i) {
            res[i][i] = 1;
        }
        return res;
    }

    /*
     * @ Purpose  :���һ��[rs,cs]��С�ĵ�λ�ԽǾ���
     */
    Matrix EYE(const size_t& rs, const size_t& cs) {
        Matrix res(rs, cs);
        for (size_t i = 0; i < rs && i < cs; ++i) {
            res[i][i] = 1;
        }

        return res;
    }
    
    /*
        @ Purpose  :����һ��rs,cs��С��[low,high)���������������
    */
    Matrix RandI_Matrix(const size_t& rs, const size_t& cs, const int& low, const int& high) {

        Matrix res(rs, cs);
        std::random_device rd;
        std::uniform_int_distribution<int> dist(low, high);

        for (size_t i = 0; i < rs; ++i) {
            for(size_t j = 0;j < cs;++j)
            {
                res[i][j] = dist(rd);
            }
        }
        return res;
    }

    /*
        @ Purpose  :����һ��rs,cs��С��[low,high)��˫�������������
    */
    Matrix RandD_Matrix(const size_t& rs, const size_t& cs, const double& low, const double& high)
    {
        std::random_device rd;
        std::uniform_real_distribution<double> dist(low,high);
        Matrix res(rs, cs);
        for (size_t i = 0; i < rs; ++i) {
            for(size_t j = 0;j < cs;++j)
            {
                res[i][j]  = dist(rd);
            }
        }

        return res;
    }

    /*
        @ Purpose  :����һ��rs,cs��С�ģ�M,S2���ֲ�����̬����
    */
    Matrix RandN_Matrix(const size_t& rs, const size_t& cs, const double& M, const double& S2)
    {
        std::random_device rd;
        std::normal_distribution<double> dist(M,S2);
        Matrix res(rs, cs);
        for (size_t i = 0; i < rs; ++i) {
            for (size_t i = 0; i < rs; ++i) {
                for(size_t j = 0;j < cs;++j)
                {
                    res[i][j] = dist(rd);
                }
            }
        }

        return res;
    }


    /*
     *  @ Purpose  :������������ʽ
     *  @ Iner_Para:
     *          i,j �����Ӿ���λ��
     *          sub_i,sub_j ��¼ɾ������λ��
     */
    double Det(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r == 0 || _mSize_c == 0) {
            std::cerr << "in func DETERMINANT(), Mat row and col size must big than 0;\n";
            return 0;
        }

        Matrix U(Mat);
        size_t max_row_index;
        size_t current_col = 0;
        size_t i = 0;
        double ele;
        double swap_count = 0;
        for (i = 0; i < _mSize_r && current_col < _mSize_c; ++i) {
            // Ѱ�Ҹ������ֵ
            max_row_index = i;

            for (; current_col < _mSize_c; ++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (abs(U[max_row_index][current_col]) < abs(U[m][current_col])) {
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

            // �������ִ����ֵ��
            if (max_row_index != i) {
                swap_count +=1;
                row_swap_PLU(U, i, max_row_index, current_col, false);
            }

            //U�� i���·�Ԫ�ر�Ϊ0��
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                size_t k = 0;
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                }
            }
            current_col += 1;
        }

        double res = 1.0*pow(-1,swap_count);
        size_t ns = std::min(_mSize_c,_mSize_r);
        for(size_t i = 0;i<ns;++i){
            res *= U[i][i];
        }
        return res;
    }


    /*
        @ Purpose  : ����ԭ����İ������
    */
    Matrix Adjoint(const Matrix& Mat) {
        std::size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        if (_mSize_r != _mSize_c) {
            std::cerr << "in func Adjoint_Matrix(),Mat must row==col!\n";
            return Matrix();
        }
        if (_mSize_r == 0 || _mSize_c == 0) {
            std::cerr << "in func Adjoint_Matrix(), Mat row and col size must big than 0;\n";
            return Matrix();
        }
        Matrix res(_mSize_r, _mSize_c);

        // step 1
        Matrix E = EYE(_mSize_r, _mSize_c);
        Matrix inMat = Inverse(Mat);
        double det = Det(Mat);
        res = inMat*DOT(E,det);

        return res;
    }


    /*
       @ Purpose  : ���ؾ������ֵ
    */
    double MAX(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        double res = Mat[0][0];
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j=0;j < _mSize_c;++j){
                res = std::max(res, Mat[i][j]);
            }
        }
        return res;
    }

    /*
        @ Purpose  :���ؾ�����Сֵ
    */  
    double MIN(const Matrix& Mat)
    {
        size_t _mSize_r, _mSize_c;;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        double res = Mat[0][0];
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j=0;j < _mSize_c;++j){
                res = std::min(res, Mat[i][j]);
            }
        }
        return res;
    }


    /*
        @ Purpose  : �����ݷ����������ֵ
        @ Para :
                    Mat �����ԭ����
                    min_delta ���Ƽ��㾫��
                    max_iter ���Ƽ�������
        @ Return : ����ԭ������������ֵ����Ӧ������������
        @ Other : �ݲ�֧�ָ������������Ϊ����
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

        Matrix X = Matrix(_mSize_r, 1, 1);   // ��������
        Matrix Y;
        double m = 0, pre_m = 0;   //����ֵ
        size_t iter = 0;
        double Delta = INT32_MAX;
        while (iter < max_iter && Delta >min_delta) {
            Y = (Mat * X);
            m = MAX(Y);
            Delta = abs(m - pre_m);
            pre_m = m;
            X = (Y / m);
            iter += 1;
        }
        return std::make_tuple(m, X);
    }

    /*
        @ Purpose  : ����������
        @ Return : �����������һ���µľ��󲢷���
        @ Other : �ݲ�֧�ָ������
        @ Example :
                Matrix A = {{1,2,3},{4,5,6}};
                Matrix B;
                B = SQRT(A);    // ��ʱΪ������ֵ��
                B = &SQRT(A);   // ��ʱΪָ�룻 ����ʹ�ñ�����
    */
    Matrix SQRT(const Matrix& Mat) {
        Matrix dst(Mat);
        size_t _mSize_r, _mSize_c;

        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j)
            {
                dst[i][j] = std::sqrt(dst[i][j]);
            }
        }
        return dst;
    }

    /*
        @ Purpose : ��һ�����󱻶��ʹ��ʱ����׼��ͬ����������Ҫʹ�ô˺���
        @ Para :
                    Mat ԭ���󣬻����������øþ���ı���
        @ Return :
                    bool
        @ Other : �˺�����Ӧ�����������ڲ�������ʾ�ĵ��ã����ܲ���Ǳ�ڵ��ڴ�й©��в
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  Matrix B;
                  B = &A;  //��ʱ����B���˾���A
                  B = Matrix(3,3); // ��ʱ����AҲ��ͬʱ����
                  B.clear();    //�ͷ������A�İ�
                  B = Matrix(4,4); // ��ʱ������B
    */
    bool Matrix::memAsycEqual(const Matrix& Mat) {
        // ָ��ͬһ�ڴ��ַ
        if (this->Mat == Mat.Mat) {
            return true;
        }
        
        // �����������Сһ�£���ʹ�ú���fast_copy();�����·����ڴ�
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
        tmp2 = new double* [new_Size_r];
        if (tmp2 == NULL) {
            throw "failed to alloc new memory!\n";
        }

        for (size_t i = 0; i < new_Size_r; ++i) {
            tmp = new double[new_Size_c];
            if (tmp == NULL) {
                throw "failed to alloc new memory!\n";
            }
//            for(size_t j = 0;j < new_Size_c;++j){
//                *(tmp + j) = Mat[i][j];
//            }
            std::copy(Mat[i],Mat[i]+new_Size_c,tmp);
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
        @ Purpose ���ٿ���������ȷ��dst�ľ�������������src
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
        @ Purpose : ��������LU�ֽ�
        @ Para :
                    Mat �����ԭ����
        @ Return :
                    Matrix(2) L����
                    Matrix(3) U����
        @ Other : ������Ϊ����
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
        size_t j, k;
        double ele;
        for (size_t i = 0; i < _mSize_r; ++i) {
            //U�� i���·�Ԫ�ر�Ϊ0��
            for ( j = i + 1; j+2 <= _mSize_r; j +=2) {
                ele = U[j][i] / U[i][i];
                L[j][i] = ele;
                k = 0;
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
        @ Purpose  :���������i�����ii��
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
        @ Purpose  :Ϊ����LU�ֽ�����޸ĵ��н�������
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
        @ Purpose : ��������PLU�ֽ�
        @ Para :
                    Mat �����ԭ����
        @ Return :
                    Matrix(1) P����
                    Matrix(2) L����
                    Matrix(3) U����
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
        double ele;

        for (i = 0; i < _mSize_r && current_col <_mSize_c; ++i) {
            // Ѱ�Ҹ������ֵ
            max_row_index = i;

            for (; current_col < _mSize_c;++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (abs(U[max_row_index][current_col]) < abs(U[m][current_col])) {
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

            // �������ִ����ֵ��
            if (max_row_index != i) {
                std::swap(P2[i], P2[max_row_index]);
                row_swap_PLU(U, i, max_row_index, current_col, false);
                row_swap_PLU(L, i, max_row_index, current_col, true);
            }

            //U�� i���·�Ԫ�ر�Ϊ0��
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                L[j][current_col] = ele;
                size_t k = 0;
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
        @ Purpose : ����������
        @ Para :
                    Mat �����ԭ����
        @ Return :
                    size_t ������ȣ����ڵ���0��С�ڵ��ھ�������� 
                    Matrix  һ�������Ǿ���
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
        double ele;

        for (i = 0; i < _mSize_r && current_col < _mSize_c; ++i) {
            // Ѱ�Ҹ������ֵ
            max_row_index = i;

            for (; current_col < _mSize_c; ++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (abs(U[max_row_index][current_col]) < abs(U[m][current_col])) {
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

            // �������ִ����ֵ��
            if (max_row_index != i) {
                row_swap_PLU(U, i, max_row_index, current_col, false);
            }

            //U�� i���·�Ԫ�ر�Ϊ0��
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                size_t k = 0;
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
     * ���ȱ任��elementary transformation��
     */
    std::tuple<size_t,Matrix> EleTrans(const Matrix& Mat) {
        size_t rank = 0;

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix U(Mat);
        size_t max_row_index;
        size_t current_col = 0;
        size_t i = 0;
        double ele,ele_s;

        for (i = 0; i < _mSize_r && current_col < _mSize_c; ++i) {
            // Ѱ�Ҹ������ֵ
            max_row_index = i;

            for (; current_col < _mSize_c; ++current_col) {
                for (size_t m = i; m < _mSize_r; ++m) {
                    if (abs(U[max_row_index][current_col]) < abs(U[m][current_col])) {
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

            // �������ִ����ֵ��
            if (max_row_index != i) {
                row_swap_PLU(U, i, max_row_index, current_col, false);
            }

            //U�� i���·�Ԫ�ر�Ϊ0��
            for (size_t j = i + 1; j < _mSize_r; ++j) {
                ele = U[j][current_col] / U[i][current_col];
                size_t k = current_col;
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                }
            }

            for(size_t j = 0; j < i;++j){
                ele = U[j][current_col] / U[i][current_col];
                size_t k = current_col;
                while (k < _mSize_c) {
                    U[j][k] -= (U[i][k] * ele);
                    ++k;
                }
            }

            ele_s = 1.0/U[i][current_col];
            for(size_t j =current_col;j<_mSize_c;++j){
                U[i][j] *=ele_s;
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
        @ Purpose : ����ԭ�����2����
        @ Para : 
                Mat �����ԭ����
        @ Return :
        @ Example :
                Matrix A = {{1,2,3},{4,5,6}};
                Matrix B;
                B = Norm_2(A);
    */
    double Norm_2(const Matrix& Mat) {
        double res = 0;
        size_t m_Size_r,m_Size_c;
        std::tie(m_Size_r,m_Size_c) = Mat.shape();
        Matrix ATA = Trans(Mat) * Mat;
        std::tie(res, std::ignore) = Power_Method(ATA);
        return sqrt(res);
    }

    /*
        @ Matrix function : A = QR
        @ Purpose : ʹ�ø���ķʩ��������������QR�ֽ�
        @ Para : 
                    Mat �����ԭ����
        @ Return :  
                    Q,R �������ռ�����
        @ Other : ���ؾ���֧�ָ������
        @ Example :
                  Matrix A = {{1,2,3},{4,5,6}};
                  Matrix Q,R;
                  std::tie(Q,R) = QR(A);
                    
    */
    // Q (m,n) , R(n,n)
    std::tuple<Matrix,Matrix> QR(const Matrix& Mat) {

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();

        Matrix Q = EYE(_mSize_r,_mSize_r);
        Matrix R(Mat);
        double s,c;
        double a1,a2;
        for(size_t k=0;k<_mSize_c;++k){
            for(size_t j = _mSize_r-1;j>k;--j){
                // G
                std::tie(c,s) = Givens(R[j-1][k],R[j][k]);

                //R = G*R,change  j-1 and j  row ele
#pragma omp paralle for
                for(size_t j2 = 0;j2<_mSize_c;++j2){
                    a1 = R[j-1][j2]*c + R[j][j2]*s;
                    R[j][j2] = R[j-1][j2]*(-s) + R[j][j2]*c;
                    R[j-1][j2] = a1;
                }

                // Q = Q*GT
#pragma omp paralle for
                for(size_t j2 = 0;j2<_mSize_r;++j2){
                    a2 = Q[j-1][j2]*c + Q[j][j2]*s;
                    Q[j][j2] = -s*Q[j-1][j2] + c*Q[j][j2];
                    Q[j-1][j2] = a2;
                }

            }
        }
        Q.trans();
        return std::make_tuple(Q, R);
    }

    /*
     * return cosTheta,sinTheta,
     */
    std::tuple<double ,double >Givens(const double &a, const double &b){
        double c,s;
        if(b==0){
            c=1;
            s=0;
        }
        else{
            if(abs(b)>abs(a)){
                double t = a/b;
                s = 1/sqrt(1+pow(t,2));
                c = s*t;
            }
            else{
                double t = b/a;
                c = 1/sqrt(1+pow(t,2));
                s = c*t;
            }
        }
        return std::make_tuple(c,s);
    }

    /*
        @ Purpose : ���������ϻ�����ȡ����������һ���µľ���
    */
    Matrix Ceil(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(Mat);
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j= 0; j < _mSize_c; ++j)  {
                res[i][j] = std::ceil(res[i][j]);
                ++j;
            }
        }
        return res;
    }
    Matrix Floor(const Matrix& Mat) {
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(Mat);
        for (size_t i = 0; i < _mSize_r; ++i) {
            for (size_t j= 0; j < _mSize_c; ++j)  {
                res[i][j] = std::floor(res[i][j]);
                ++j;
            }
        }
        return res;
    }

    /*
        @ Purpose :������������˷�ʱ������˳��Լ����ٶ�Ӱ��ܴ󣬴˺����Լ���˳������������в����м���
        @ Return : ���ش������ĳ˷�������
        @ Other : ������������˹���
        @ Example :
                Matrix A,B,C,D;
                A = RandI_Matrix(3, 3, 1, 10);
                B = RandI_Matrix(3, 10, 1, 10);
                C = RandI_Matrix(10, 20, 1, 10);
                D = RandI_Matrix(20, 3, 1, 10);
                Matrix D = MMul({A,B,C,D});
    */
    Matrix MMul(std::initializer_list <Matrix> Mats) {
        std::vector<Matrix> vec(Mats.begin(),Mats.end());
        size_t _mSize_r, _mSize_c;
        size_t max_Size_c = 0;
        size_t MaxCol_MaxIndex = 0;
        size_t i = 0;
        size_t Size = vec.size();
        while (Size>1)
        {
            MaxCol_MaxIndex = 0;
            std::tie(std::ignore, max_Size_c) = vec[0].shape();
            for (i = 1; i < Size-1; ++i) {
                std::tie(std::ignore, _mSize_c) = vec[i].shape();
                if (max_Size_c < _mSize_c) {
                    MaxCol_MaxIndex = i;
                    max_Size_c = _mSize_c;
                }
            }
            vec[MaxCol_MaxIndex] = (vec[MaxCol_MaxIndex]*vec[MaxCol_MaxIndex + 1]);
            for (i = MaxCol_MaxIndex+1; i < Size-1; ++i) {
                vec[i] = &vec[i + 1];
            }
            --Size;
        }
        return vec[0];
    }


    /*
        @ Purpose : ��ȡ����Mat��Mat2�ıȽϽ��
        @ Return : ��Mat[0,0] == Mat2[0,0]����res[0,0]=1������Ϊ0
        @          �������С��һ�£��򷵻�һ��Matrix()�վ���
        @ Other : �����С����һ��
    */
    Matrix operator==(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }
        
        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (abs(res[i][j]-Mat2[i][j]) < 1e-10);
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
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] != Mat2[i][j]);
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
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] > Mat2[i][j]);
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
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] < Mat2[i][j]);
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
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] >= Mat2[i][j]);
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
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] <= Mat2[i][j]);
            }
        }
        return res;
    }
    Matrix operator||(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] || Mat2[i][j]);
            }
        }
        return res;
    }
    Matrix operator&&(const Matrix& Mat, const Matrix& Mat2) {
        if (Mat.shape() != Mat2.shape()) {
            std::cerr << "Mat's shape must equal Mat2's shape\n";
            return Matrix();
        }

        size_t _mSize_r, _mSize_c;
        std::tie(_mSize_r, _mSize_c) = Mat.shape();
        Matrix res(Mat);
#pragma omp paralle for
        for (size_t i = 0; i < _mSize_r; ++i) {
            for(size_t j = 0;j < _mSize_c;++j){
                res[i][j] = (res[i][j] && Mat2[i][j]);
            }
        }
        return res;
    }


}

#endif //LALIB_UMATRIX_H
