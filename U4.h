#ifndef U4_H
#define U4_H

#include <iostream>
#include <memory>
#include <tuple>
#include <random>
#include <immintrin.h>
#include <iomanip>

namespace u4 {

	template <typename T>
	class Matrix;

	template <typename T>
	Matrix<T> RandI_Matrix(const size_t& _row, const size_t& _col, const T& Low, const T& High);
	template <typename T>
	std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat);
	template <typename T>
	Matrix<T> Trans(const Matrix<T>& mat);
	template <typename T>
	Matrix<T> operator/(const Matrix<T>& mat, const T& num);
	template <typename T>
	Matrix<T> operator*(const Matrix<T>& mat, const Matrix<T>& mat2);
	template <typename T>
	T Max(const Matrix<T>& mat);
	template <typename T>
	T Min(const Matrix<T>& mat);
	template <typename T>
	std::tuple<T, Matrix<T>> Power_Method(const Matrix<T>& mat, const double& min_delta = 1e-5, const size_t& max_iter = 1e5);
	template <typename T>
	Matrix<T> Dot(const Matrix<T>& mat, const Matrix<T>& mat2);
	template <typename T>
	Matrix<T> Dot(const Matrix<T>& mat, const T& num);
	template <typename T>
	Matrix<T> Inverse(const Matrix<T>& Mat);
	template <typename T>
	Matrix<T> Eye(const size_t& _row, const size_t& _col);
	template <typename T>
	std::tuple<size_t, Matrix<T>> EleTrans(const Matrix<T>& mat);
	template <typename T>
	Matrix<T> Inverse(const Matrix<T>& mat);
	template <typename T>
	Matrix<T> Adjoint(const Matrix<T>& mat);
	template<typename T>
	T Det(const Matrix<T>& mat);
	template<typename T>
	std::tuple<size_t, Matrix<T>> Rank(const Matrix<T>& mat);
	template <typename T>
	Matrix<T> Eye(const size_t& _size);
	template <typename T>
	Matrix<T> Eye(const Matrix<T>& mat);
	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>> QR(const Matrix<T>& mat);
	template<typename T>
	T Norm_2(const Matrix<T>& mat);
	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> PLU(const Matrix<T>& mat);
	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>> LU(const Matrix<T>& mat);
	template<typename T>
	Matrix<T> Ceil(const Matrix<T>& Mat);
	template<typename T>
	Matrix<T> Floor(const Matrix<T>& Mat);
	template<typename T>
	std::tuple<T, T >Givens(const T& a, const T& b);
	template<typename T>
	Matrix<T> operator+(const Matrix<T>& mat, const Matrix<T>& mat2);
	template<typename T>
	Matrix<T> operator-(const Matrix<T>& Mat, const Matrix<T>& Mat2);


	void Test(int argc, char* argv[]);


	template <typename T>
	class Matrix
	{
		template <typename Ty>
		friend Ty Min(const Matrix<Ty>& mat);
		template <typename Ty>
		friend Ty Max(const Matrix<Ty>& mat);


		template <typename Ty>
		friend Matrix<Ty> RandI_Matrix(const size_t& _row, const size_t& _col, const Ty& Low, const Ty& High);
		template <typename Ty>
		friend Matrix<Ty> Trans(const Matrix<Ty>& mat);
		template<typename Ty>
		friend std::tuple<Matrix<Ty>, Matrix<Ty>, Matrix<Ty>> PLU(const Matrix<Ty>& mat);
		template<typename Ty>
		friend std::tuple< Matrix<Ty>, Matrix<Ty>> LU(const Matrix<Ty>& mat);
		template <typename Ty>
		friend std::tuple<Ty, Matrix<Ty>> Power_Method(const Matrix<Ty>& mat, const double& min_delta, const size_t& max_iter);
		template <typename Ty>
		friend Matrix<Ty> Dot(const Matrix<Ty>& mat, const Matrix<Ty>& mat2);
		template <typename Ty>
		friend Matrix<Ty> Dot(const Matrix<Ty>& mat, const Ty& num);
		template <typename Ty>
		friend Matrix<Ty> Inverse(const Matrix<Ty>& Mat);
		template <typename Ty>
		friend Matrix<Ty> Eye(const size_t& _row, const size_t& _col);
		template <typename Ty>
		friend std::tuple<size_t, Matrix<Ty>> EleTrans(const Matrix<Ty>& mat);
		template<typename Ty>
		friend Ty Det(const Matrix<Ty>& mat);
		template<typename Ty>
		friend Matrix<Ty> Adjoint(const Matrix<Ty>& mat);
		template<typename Ty>
		friend Matrix<Ty> Ceil(const Matrix<Ty>& Mat);
		template<typename Ty>
		friend Matrix<Ty> Floor(const Matrix<Ty>& Mat);
		template<typename Ty>
		friend std::tuple<size_t, Matrix<Ty>> Rank(const Matrix<Ty>& mat);
		template<typename Ty>
		friend std::tuple<Matrix<Ty>, Matrix<Ty>> QR(const Matrix<Ty>& mat);

		friend void _DQR(double *mat,const size_t &_row,const size_t &_col,const size_t &i);
		friend void _DEleTrans(double* U, const size_t& current_row, const size_t& current_col, const size_t& _row, const size_t& _col);
		friend void _FEleTrans(float* U, const size_t& current_row, const size_t& current_col, const size_t& _row, const size_t& _col);

		template <typename Ty>
		friend std::ostream& operator<<(std::ostream& out, const Matrix<Ty>& mat);
		template <typename Ty>
		friend Matrix<Ty> operator/(const Matrix<Ty>& mat, const Ty& num);
		template <typename Ty>
		friend Matrix<Ty> operator*(const Matrix<Ty>& mat, const Matrix<Ty>& mat2);
		friend Matrix<double> operator*(const Matrix<double>& dmat, const Matrix<double>& dmat2);
		friend Matrix<float> operator*(const Matrix<float>& dmat, const Matrix<float>& dmat2);
		template<typename Ty>
		friend Matrix<Ty> operator+(const Matrix<Ty>& mat, const Matrix<Ty>& mat2);
		template<typename Ty>
		friend Matrix<Ty> operator-(const Matrix<Ty>& mat, const Matrix<Ty>& mat2);
		

	public:
		Matrix() = default;
		Matrix(const size_t& _row, const size_t& _col, const T& init_num = 0);
		Matrix(const Matrix<T>& mat);
		T* operator[](const size_t& index);
		T* operator[](const size_t& index) const;
		size_t size()const noexcept { return this->row * this->col; }
		std::tuple<size_t, size_t> shape()const noexcept;
		Matrix<T>& operator=(const Matrix<T>& mat) noexcept;
		void resize(const size_t& _nrow, const size_t& _ncol) noexcept;
		void part_set(const Matrix<T>& mat, const size_t& r, const size_t& c, const size_t& r2 = -1, const size_t& c2 = -1);
		Matrix<T> operator()(const size_t& r, const size_t& c, const size_t& r2 = 0, const size_t& c2 = 0);
		void trans();
		~Matrix();
	private:
		T* Mat = NULL;
		size_t row = 0, col = 0;
	};


	template <typename T>
	Matrix<T>::~Matrix()
	{
		this->row = this->col = 0;
		delete[]Mat;
		this->Mat = NULL;
	}

	template<typename T>
	inline Matrix<T>::Matrix(const size_t& _row, const size_t& _col, const T& init_num)
	{
		

		if (_row == 0 || _col == 0) {
			std::cerr << "_row and _col is big than 0.\n";
			return;
		}
		this->row = _row;
		this->col = _col;
		this->Mat = new T[this->size()]{ 0 };
		
		size_t mat_size = this->size();
		if (init_num != 0) {
			for (size_t i = 0; i < mat_size; ++i) {
				this->Mat[i] = init_num;
			}
		}
	}

	template<typename T>
	void Matrix<T>::trans() {
		if (this->row == 1 || this->col == 1) {
			std::swap(this->row, this->col);
		}
		else {
			*this = Trans(*this);
		}
	}



	template<typename T>
	void Matrix<T>::resize(const size_t& _nrow, const size_t& _ncol) noexcept {
		T* nmat = new T[_nrow * _ncol]{ 0 };
		size_t i = 0;
		size_t copy_length = std::min(_ncol, this->col);
		size_t copy_high = std::min(_nrow, this->row);
		while (i < copy_high) {
			std::copy(this->Mat + i * col, this->Mat + i * col + copy_length, nmat + i * _ncol);
			++i;
		}
		T* wait_del_mat = this->Mat;
		this->Mat = nmat;
		delete[]wait_del_mat;
		this->row = _nrow;
		this->col = _ncol;
	}

	/*
			-  (r,c)  -    -       -   -    -
			-    -    -    -       -	  -    -
			-    -    -  (r2,c2)   -   -  (-1,-1)
	*/
	template<typename T>
	inline void Matrix<T>::part_set(const Matrix<T>& mat, const size_t& r, const size_t& c, const size_t& r2, const size_t& c2)
	{
		size_t i = r;
		while (i < mat.row)
		{
			std::copy(mat[i - r], mat[i - r] + mat.col, (*this)[i] + c);
			++i;
		}
	}

	template<typename T>
	Matrix<T> Matrix<T>::operator()(const size_t& r, const size_t& c, const size_t& r2, const size_t& c2)
	{
		size_t pmat_row = this->row - r;
		size_t pmat_col = this->col - c;
		if (r2 != 0 && r2 > r) {
			pmat_row = r2 - r;
		}
		if (c2 != 0 && c2 > c) {
			pmat_col = c2 - c;
		}
		Matrix<T> pmat(pmat_row, pmat_col);
		size_t i = 0;
		while (i < pmat_row) {
			std::copy(this->Mat + (i + r) * this->col + c, this->Mat + (i + r) * this->col + c + pmat_col, pmat[i]);
			++i;
		}

		return pmat;
	}

	template<typename T>
	Matrix<T>::Matrix(const Matrix<T>& mat) {
		this->Mat = new T[mat.size()];
		this->row = mat.row;
		this->col = mat.col;
		std::copy(mat.Mat, mat.Mat + mat.size(), this->Mat);
	}

	template<typename T>
	std::tuple<size_t, size_t> Matrix<T>::shape()const noexcept {
		return std::make_tuple(this->row, this->col);
	}

	template <typename T>
	T* Matrix<T>::operator[](const size_t& index) {
		if (this->Mat == NULL) {
			std::cerr << "Mat is empty.\n";
			return NULL;
		}
		return this->Mat + index * this->col;
	}
	template <typename T>
	T* Matrix<T>::operator[](const size_t& index) const {
		if (this->Mat == NULL) {
			std::cerr << "Mat is empty.\n";
			return NULL;
		}
		return this->Mat + index * this->col;
	}

	template <typename T>
	Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat) noexcept {
		if (this->size() != mat.size()) {
			T* wait_del_mem = this->Mat;
			this->Mat = new T[mat.size()];
			if (wait_del_mem != NULL) {
				delete[]wait_del_mem;
			}
		}

		std::tie(this->row, this->col) = mat.shape();
		std::copy(mat.Mat, mat.Mat + mat.size(), this->Mat);
		return *this;
	}


	template<typename T>
	T Min(const Matrix<T>& mat)
	{
		T min_val = mat[0][0];

		size_t mat_size = mat.size();
		size_t i = 0;
		while (i<mat_size)
		{
			min_val = std::min(min_val, mat.Mat[i]);
			++i;
		}
		return min_val;
	}

	template <typename T>
	std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat)
	{
		if (mat.size() == 0) {
			out << "--- " << 0 << ", " << 0 << " ---\n";
			return out;
		}
		for (size_t i = 0; i < mat.row; ++i) {
			out << "  ";
			for (size_t j = 0; j < mat.col; ++j) {
				if (fabs(mat[i][j]) < 1e-10) {
					out << 0 << " ";
				}
				else {
					out << mat[i][j] << " ";
				}

			}
			out << "\n";
		}
		out << "--- " << mat.row << ", " << mat.col << " ---\n";
		return out;
	}

	template<typename T>
	Matrix<T> RandI_Matrix(const size_t& _row, const size_t& _col, const T& Low, const T& High)
	{
		if (_row == 0 || _col == 0) {
			std::cerr << "_row and _col is big than 0.\n";
			return Matrix<T>();
		}
		Matrix<T> res(_row, _col);
		std::random_device rd;
		std::uniform_int_distribution<int> dist((int)Low, (int)High);
		for (size_t i = 0; i < _row; ++i) {
			for (size_t j = 0; j < _col; ++j) {
				res[i][j] = dist(rd);
			}
		}

		return res;
	}


	template <typename T>
	Matrix<T> Trans(const Matrix<T>& mat) {
		if (mat.size() == 0) {
			return mat;
		}

		Matrix<T> matT;
		if (mat.row == 1 || mat.col == 1) {
			matT = mat;
			std::swap(matT.row, matT.col);
			return matT;
		}

		matT = Matrix<T>(mat.col, mat.row);

		for (size_t i = 0; i < mat.row; ++i) {
			for (size_t j = 0; j < mat.col; ++j) {
				matT[j][i] = mat[i][j];
			}
		}

		return matT;
	}

	template <typename T>
	Matrix<T> operator/(const Matrix<T>& mat, const T& num) {
		T div_num = 1.0 / num;
		Matrix<T> res(mat);
		size_t mat_size = mat.size();
		// #pragma omp parallel for
		for (size_t i = 0; i < mat_size; ++i) {
			res.Mat[i] *= div_num;
		}
		return res;
	}

	Matrix<double> operator*(const Matrix<double>& dmat, const Matrix<double>& dmat2) {
		size_t r, c;
		std::tie(r, c) = dmat.shape();
		size_t r2, c2;
		std::tie(r2, c2) = dmat2.shape();

		Matrix<double> dst(r, c2);
		Matrix<double> mat2T = Trans(dmat2);
		double TMP[4];
		double TMP2[4];
		__m256d m256D;
		__m256d m256D2;
		__m256d m256D3;

		for (size_t i = 0; i < r; ++i) {
			for (size_t j = 0; j < c2; j += 2) {
				m256D = _mm256_setzero_pd();
				m256D2 = _mm256_setzero_pd();
				size_t k = 0;
				while (k + 4 <= c) {
					m256D3 = _mm256_loadu_pd(dmat[i] + k);
					m256D = _mm256_add_pd(m256D,
						_mm256_mul_pd(m256D3, _mm256_loadu_pd(mat2T[j] + k)));
					if (j + 1 < c2) {
						m256D2 = _mm256_add_pd(m256D2,
							_mm256_mul_pd(m256D3, _mm256_loadu_pd(mat2T[j + 1] + k)));
					}

					k += 4;
				}
				_mm256_storeu_pd(TMP, m256D);
				dst[i][j] += (TMP[0] + TMP[1] + TMP[2] + TMP[3]);
				if (j + 1 < c2) {
					_mm256_storeu_pd(TMP2, m256D2);
					dst[i][j + 1] += (TMP2[0] + TMP2[1] + TMP2[2] + TMP2[3]);
				}

				while (k < c) {
					dst[i][j] += (dmat[i][k] * mat2T[j][k]);
					if (j + 1 < c2) {
						dst[i][j + 1] += (dmat[i][k] * mat2T[j + 1][k]);
					}

					++k;
				}
			}
		}
		return dst;
	}


	Matrix<float> operator*(const Matrix<float>& dmat, const Matrix<float>& dmat2) {
		size_t r, c;
		std::tie(r, c) = dmat.shape();
		size_t r2, c2;
		std::tie(r2, c2) = dmat2.shape();

		Matrix<float> dst(r, c2);
		Matrix<float> mat2T = Trans(dmat2);
		float TMP[8];
		float TMP2[8];
		__m256 m256I;
		__m256 m256I2;
		__m256 m256I3;

		for (size_t i = 0; i < r; ++i) {
			for (size_t j = 0; j < c2; j += 2) {
				m256I = _mm256_setzero_ps();
				m256I2 = _mm256_setzero_ps();
				m256I3 = _mm256_setzero_ps();
				size_t k = 0;
				while (k + 8 <= c) {
					m256I3 = _mm256_loadu_ps(dmat[i] + k);
					m256I = _mm256_add_ps(m256I, _mm256_mul_ps(m256I3, _mm256_loadu_ps(mat2T[j] + k)));
					if (j + 1 < c2) {
						m256I2 = _mm256_add_ps(m256I2, _mm256_mul_ps(m256I3, _mm256_loadu_ps(mat2T[j + 1] + k)));
					}
					k += 8;
				}
				_mm256_storeu_ps(TMP, m256I);
				dst[i][j] += (TMP[0] + TMP[1] + TMP[2] + TMP[3] + TMP[4] + TMP[5] + TMP[6] + TMP[7]);
				if (j + 1 < c2) {
					_mm256_storeu_ps(TMP2, m256I2);
					dst[i][j + 1] += (TMP2[0] + TMP2[1] + TMP2[2] + TMP2[3] + TMP2[4] + TMP2[5] + TMP2[6] + TMP2[7]);
				}

				while (k < c) {
					dst[i][j] += (dmat[i][k] * mat2T[j][k]);
					if (j + 1 < c2) {
						dst[i][j + 1] += (dmat[i][k] * mat2T[j + 1][k]);
					}

					++k;
				}
			}
		}
		return dst;
	}


	template <typename T>
	Matrix<T> operator*(const Matrix<T>& mat, const Matrix<T>& mat2) {

		if (mat.col != mat2.row) {
			std::cerr << "row_size must equal col_size in operator*() !\n";
		}

		Matrix<T> dst(mat.row, mat2.col);
		Matrix<T> mat2T = Trans(mat2);
		T a = 0, b = 0, c = 0, d = 0;
		for (size_t i = 0; i < mat.row; ++i) {
			for (size_t j = 0; j < mat2.col; ++j) {
				size_t k = 0;
				while (k + 4 <= mat.col) {
					a += (mat[i][k] * mat2T[j][k]);
					b += (mat[i][k + 1] * mat2T[j][k + 1]);
					c += (mat[i][k + 2] * mat2T[j][k + 2]);
					d += (mat[i][k + 3] * mat2T[j][k + 3]);
					k += 4;
				}
				dst[i][j] += (a + b + c + d);
				while (k < mat.col) {
					dst[i][j] += (mat[i][k] * mat2T[j][k]);
					++k;
				}
				// for (size_t k = 0; k < mat.col; ++k) {
				//     dst[i][j] += (mat[i][k] * mat2T[j][k]);
				// }
				a = b = c = d = 0;
			}
		}

		return dst;
	}

	template <typename T>
	T Max(const Matrix<T>& mat) {
		T max_val = mat[0][0];
		size_t mat_size = mat.size();

		for (size_t i = 0; i < mat_size; ++i) {
			max_val = std::max(mat.Mat[i], max_val);
		}

		return max_val;
	}

	template <typename T>
	std::tuple<T, Matrix<T>> Power_Method(const Matrix<T>& mat, const double& min_delta, const size_t& max_iter) {

		if (mat.row != mat.col) {
			std::cerr << "in func Power_Method(), Mat row and col size must equal.\n";
			return std::make_tuple(0, Matrix<T>());
		}

		Matrix<T> X = Matrix<T>(mat.row, 1, (T)1);   // 特征向量
		Matrix<T> Y;
		T m = 0, pre_m = 0;   //特征值
		size_t iter = 0;
		T Delta = T(min_delta + 1.0);
		T Min_delta = T(min_delta);
		while (iter < max_iter && Delta > Min_delta) {
			Y = (mat * X);
			m = Max(Y);
			Delta = fabs(m - pre_m);
			pre_m = m;
			X = (Y / m);
			iter += 1;
		}
		return std::make_tuple(m, X);
	}


	template <typename T>
	Matrix<T> Dot(const Matrix<T>& mat, const Matrix<T>& mat2) {

		if (mat.row != mat2.row) {
			std::cerr << "Dot() error.\n";
			return Matrix<T>();
		}

		Matrix<T> dst(mat);

		if (mat.size() == mat2.size()) {
			size_t mat_size = mat.size();
			for (size_t i = 0; i < mat_size; ++i) {
				dst.Mat[i] *= mat2.Mat[i];
			}
		}
		else if (mat.row == mat2.row) {
			for (size_t i = 0; i < mat.row; ++i) {
				T tmp = mat2[i][0];
				for (size_t j = 0; j < mat.col; ++j) {
					dst[i][j] *= tmp;
				}
			}
		}

		return dst;
	}

	template <typename T>
	Matrix<T> Dot(const Matrix<T>& mat, const T& num) {
		Matrix<T> dst(mat);

		size_t mat_size = mat.size();
		size_t i = 0;
		while (i < mat_size) {
			mat.Mat[i] *= num;
			++i;
		}
		return dst;
	}

	template <typename T>
	Matrix<T> Eye(const size_t& _row, const size_t& _col) {
		Matrix<T> eye(_row, _col);
		size_t i = 0;
		size_t Length = std::min(_row, _col);
		while (i < Length) {
			eye[i][i] = 1;
			++i;
		}
		return eye;
	}

	void _DEleTrans(double* U, const size_t& current_row, const size_t& current_col, const size_t& _row, const size_t& _col)
	{
		__m256d m256D;
		__m256d m256D2;
		double ele;
		size_t i = 0, j = 0;
		
		while (i<_row)
		{
			j = current_col;
			ele = U[i * _col + j]/U[current_row*_col+j];
			m256D = _mm256_set_pd(ele, ele, ele,ele);

			while (i != current_row && j+4 <= _col) {
				m256D2 = _mm256_mul_pd(_mm256_loadu_pd(U + current_row * _col + j), m256D);
				m256D2 = _mm256_sub_pd(_mm256_loadu_pd(U + i * _col + j),m256D2);
				_mm256_storeu_pd(U + i * _col + j,m256D2);
				j += 4;
			}

			while (i!=current_row && j<_col)
			{
				U[i * _col + j] -= (U[current_row * _col + j]*ele);
				++j;
			}
			++i;
		}
	}

	void _FEleTrans(float* U, const size_t& current_row, const size_t& current_col, const size_t& _row, const size_t& _col)
	{
		__m256 m256I;
		__m256 m256I2;
		float ele;
		size_t i = 0, j = 0;

		while (i < _row)
		{
			j = current_col;
			ele = U[i * _col + j] / U[current_row * _col + j];
			m256I = _mm256_set_ps(ele, ele, ele, ele, ele, ele, ele, ele);

			while (i != current_row && j + 8 <= _col) {
				m256I2 = _mm256_mul_ps(_mm256_loadu_ps(U + current_row * _col + j), m256I);
				m256I2 = _mm256_sub_ps(_mm256_loadu_ps(U + i * _col + j), m256I2);
				_mm256_storeu_ps(U + i * _col + j, m256I2);
				j += 8;
			}

			while (i != current_row && j < _col)
			{
				U[i * _col + j] -= (U[current_row * _col + j] * ele);
				++j;
			}
			++i;
		}
	}


	template <typename T>
	std::tuple<size_t, Matrix<T>> EleTrans(const Matrix<T>& mat) {
		Matrix<T> U(mat);
		size_t max_row_index;
		size_t current_col = 0;
		size_t i = 0;
		T ele, ele_s;
		T* tmp_row = new T[mat.col];

		for (i = 0; i < mat.row && current_col < mat.col; ++i) {
			// 寻找该列最大值
			max_row_index = i;

			for (; current_col < mat.col; ++current_col) {
//#pragma omp parallel for
				for (size_t m = i; m < mat.row; ++m) {
					if (abs(U[max_row_index][current_col]) < abs(U[m][current_col])) {
						max_row_index = m;
					}
				}

				if (U[max_row_index][current_col] != 0) {
					break;
				}
			}

			if (current_col == mat.col || max_row_index == mat.row) {
				break;
			}

			// 交换该轮次最大值行
			if (max_row_index != i) {
				//row_swap_PLU(U, i, max_row_index, current_col, false);
				std::copy(U[i], U[i] + mat.col, tmp_row);
				std::copy(U[max_row_index], U[max_row_index] + mat.col, U[i]);
				std::copy(tmp_row, tmp_row + mat.col, U[max_row_index]);
			}

			//U中 i列下方元素变为0；
			if (typeid(T).name() == typeid(double).name()) {
				_DEleTrans((double*)U.Mat, i, current_col, U.row, U.col);
			}
			else if (typeid(T).name() == typeid(float).name()) {
				_FEleTrans((float*)U.Mat, i, current_col, U.row, U.col);
			}
			else {
				for (size_t j = 0; j < mat.row; ++j) {
					if (i != j) {
						size_t k = current_col;
						ele = U[j][current_col] / U[i][current_col];
						while (k < mat.col) {
							U[j][k] -= (U[i][k] * ele);
							++k;
						}
					}
				}
			}


			ele_s = 1.0 / U[i][current_col];
//#pragma omp parallel for
			for (size_t j = current_col; j < mat.col; ++j) {
				U[i][j] *= ele_s;
			}
			current_col += 1;

		}

		size_t rank = 0;
		current_col = 0;
		for (i = 0; i < mat.row; ++i) {
			for (current_col = 0; current_col < mat.col; ++current_col) {
				if (fabs(U[i][current_col]) < 1e-10) {
					U[i][current_col] = 0;
				}
				if (U[i][current_col] != 0) {
					rank += 1;
					break;
				}
			}
			current_col += 1;
		}

		return std::make_tuple(rank, U);
	}


	/*
	 * @ Purpose  : 求逆矩阵
	 */
	template <typename T>
	Matrix<T> Inverse(const Matrix<T>& Mat) {
		size_t _mSize_r, _mSize_c;
		std::tie(_mSize_r, _mSize_c) = Mat.shape();
		Matrix<T> tmp(Mat);
		Matrix<T> E = Eye<T>(_mSize_r, _mSize_c);
		tmp.resize(_mSize_r, 2 * _mSize_c);
		tmp.part_set(E, 0, _mSize_c);
		std::tie(std::ignore, tmp) = EleTrans<T>(tmp);
		size_t ns = std::min(_mSize_c, _mSize_r);
		T det = (T)1.0;
		for (size_t i = 0; i < ns; ++i) {
			det *= tmp[i][i];
		}
		if (fabs(det) <= 1e-20) {
			std::cerr << "mat's det = 0!\n";
		}
		tmp = tmp(0, _mSize_c, 0, 0);

		return tmp;
	}

	template<typename T>
	Matrix<T> Adjoint(const Matrix<T>& mat)
	{
		Matrix<T> eye = Eye<T>(mat.row, mat.col);
		Matrix<T> inMat = Inverse<T>(mat);
		T det_val = Det<T>(mat);
		return inMat*Dot(eye,det_val);
	}

	template<typename T>
	T Det(const Matrix<T>& mat) {
		Matrix<T> U(mat);
		size_t max_row_index;
		size_t current_col = 0;
		size_t i = 0;
		T ele, ele_s;
		T* tmp_row = new T[mat.col];
		size_t swap_count = 0;
		for (i = 0; i < mat.row && current_col < mat.col; ++i) {
			// 寻找该列最大值
			max_row_index = i;

			for (; current_col < mat.col; ++current_col) {
//#pragma omp parallel for
				for (size_t m = i; m < mat.row; ++m) {
					if (fabs(U[max_row_index][current_col]) <fabs(U[m][current_col])) {
						max_row_index = m;
					}
				}

				if (U[max_row_index][current_col] != 0) {
					break;
				}
			}

			if (current_col == mat.col || max_row_index == mat.row) {
				break;
			}

			// 交换该轮次最大值行
			if (max_row_index != i) {
				swap_count += 1;
				//row_swap_PLU(U, i, max_row_index, current_col, false);
				std::copy(U[i], U[i] + mat.col, tmp_row);
				std::copy(U[max_row_index], U[max_row_index] + mat.col, U[i]);
				std::copy(tmp_row, tmp_row + mat.col, U[max_row_index]);
			}

			//U中 i列下方元素变为0；
			if (typeid(T).name() == typeid(double).name()) {
				_DEleTrans((double*)U.Mat, i, current_col, U.row, U.col);
			}
			else if (typeid(T).name() == typeid(float).name()) {
				_FEleTrans((float*)U.Mat, i, current_col, U.row, U.col);
			}
			else {
				//std::cout << "I";
				for (size_t j = 0; j < mat.row; ++j) {
					if (i != j) {
						size_t k = current_col;
						ele = U[j][current_col] / U[i][current_col];
						while (k < mat.col) {
							U[j][k] -= (U[i][k] * ele);
							++k;
						}
					}
				}
			}
			current_col += 1;
		}

		T res = (T)1.0 * pow(-1, swap_count);
		size_t ns = std::min(mat.row,mat.col);
		for (size_t i = 0; i < ns; ++i) {
			res *= U[i][i];
		}
		return res;
	}

	template<typename T>
	std::tuple<size_t, Matrix<T>> Rank(const Matrix<T>& mat)
	{
		Matrix<T> U(mat);
		size_t rank=0;
		size_t max_row_index;
		size_t current_col = 0;
		size_t i = 0;
		T ele, ele_s;
		T* tmp_row = new T[mat.col];
		size_t swap_count = 0;
		for (i = 0; i < mat.row && current_col < mat.col; ++i) {
			// 寻找该列最大值
			max_row_index = i;

			for (; current_col < mat.col; ++current_col) {
//#pragma omp parallel for
				for (size_t m = i; m < mat.row; ++m) {
					if (fabs(U[max_row_index][current_col]) < fabs(U[m][current_col])) {
						max_row_index = m;
					}
				}

				if (U[max_row_index][current_col] != 0) {
					break;
				}
			}

			if (current_col == mat.col || max_row_index == mat.row) {
				break;
			}

			// 交换该轮次最大值行
			if (max_row_index != i) {
				swap_count += 1;
				//row_swap_PLU(U, i, max_row_index, current_col, false);
				std::copy(U[i], U[i] + mat.col, tmp_row);
				std::copy(U[max_row_index], U[max_row_index] + mat.col, U[i]);
				std::copy(tmp_row, tmp_row + mat.col, U[max_row_index]);
			}

			//U中 i列下方元素变为0；
			if (typeid(T).name() == typeid(double).name()) {
				_DEleTrans((double*)U.Mat, i, current_col, U.row, U.col);
			}
			else if (typeid(T).name() == typeid(float).name()) {
				_FEleTrans((float*)U.Mat, i, current_col, U.row, U.col);
			}
			else {
				//std::cout << "I";
				for (size_t j = 0; j < mat.row; ++j) {
					if (i != j) {
						size_t k = current_col;
						ele = U[j][current_col] / U[i][current_col];
						while (k < mat.col) {
							U[j][k] -= (U[i][k] * ele);
							++k;
						}
					}
				}
			}
			current_col += 1;
		}

		current_col = 0;
		for (i = 0; i < mat.row; ++i) {
			for (current_col = 0; current_col < mat.col; ++current_col) {
				if (fabs(U[i][current_col]) > 1e-5) {
					rank += 1;
					break;
				}
			}
			current_col += 1;
		}

		//std::cout << U << rank;
		return std::tuple<size_t, Matrix<T>>(rank,U);
	}

	template<typename T>
	Matrix<T> Eye(const size_t& _size)
	{
		Matrix<T> eye(_size, _size);
		size_t i = 0;
		while (i<_size)
		{
			eye[i][i] = 1;
			++i;
		}
		return eye;
	}

	template<typename T>
	Matrix<T> Eye(const Matrix<T>& mat)
	{
		size_t _size = std::min(mat.row, mat.col);
		size_t i = 0;
		Matrix<T> eye_val(_size, 1);
		while (i<_size)
		{
			eye_val[i][0] = mat[i][i];
			++i;
		}
		return eye_val;
	}


	void _DQR(double *mat,const size_t &_row,const size_t &_col,const size_t &i,double c, double s){
		
		size_t i2 = 0;
		size_t j2 = 0;

		size_t range = i*_col;

		__m256d m256A1;

		__m256d m256C = _mm256_set_pd(c,c,c,c);
		__m256d m256S = _mm256_set_pd(s,s,s,s);
		__m256d m256S2 = _mm256_set_pd(-s,-s,-s,-s);

		__m256d m256D;
		__m256d m256D2;
		while(j2+4<=_col)
		{
			m256D = _mm256_loadu_pd(mat+range-_col+j2);
			m256D2 = _mm256_loadu_pd(mat+range+j2);
			m256A1 = _mm256_add_pd(_mm256_mul_pd(m256D,m256C),_mm256_mul_pd(m256D2,m256S));
			_mm256_storeu_pd(mat+range+j2,_mm256_add_pd(_mm256_mul_pd(m256D,m256S2),_mm256_mul_pd(m256D2,m256C)));
			_mm256_storeu_pd(mat+range-_col+j2,m256A1);
			j2+=4;
		}

		double a1;
		while (j2 <_col)
		{
			a1 = mat[range-_col +j2] * c + mat[range+j2] * s;
			mat[range+j2] = mat[range-_col +j2] * (-s) + mat[range+j2] * c;
			mat[range-_col +j2] = a1;
			++j2;
		}
	}

	void _FQR(float *mat,const size_t &_row,const size_t &_col,const size_t &i,float c, float s){
		
		size_t i2 = 0;
		size_t j2 = 0;

		size_t range = i*_col;

		__m256 m256A1;

		__m256 m256C = _mm256_set_ps(c,c,c,c,c,c,c,c);
		__m256 m256S = _mm256_set_ps(s,s,s,s,s,s,s,s);
		__m256 m256S2 = _mm256_set_ps(-s,-s,-s,-s,-s,-s,-s,-s);

		__m256 m256D;
		__m256 m256D2;
		while(j2+8<=_col)
		{
			m256D = _mm256_loadu_ps(mat+range-_col+j2);
			m256D2 = _mm256_loadu_ps(mat+range+j2);
			m256A1 = _mm256_add_ps(_mm256_mul_ps(m256D,m256C),_mm256_mul_ps(m256D2,m256S));
			_mm256_storeu_ps(mat+range+j2,_mm256_add_ps(_mm256_mul_ps(m256D,m256S2),_mm256_mul_ps(m256D2,m256C)));
			_mm256_storeu_ps(mat+range-_col+j2,m256A1);
			j2+=8;
		}

		float a1;
		while (j2 <_col)
		{
			a1 = mat[range-_col +j2] * c + mat[range+j2] * s;
			mat[range+j2] = mat[range-_col +j2] * (-s) + mat[range+j2] * c;
			mat[range-_col +j2] = a1;
			++j2;
		}
		

	}


	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>> QR(const Matrix<T>& mat)
	{
		size_t _mSize_r, _mSize_c;
		std::tie(_mSize_r, _mSize_c) = mat.shape();

		Matrix<T> Q = Eye<T>(_mSize_r, _mSize_r);
		Matrix<T> R(mat);
		T s, c;
		T a1, a2;
		for (size_t k = 0; k < _mSize_c; ++k) {
			for (size_t j = _mSize_r - 1; j > k; --j) {
				// G
				std::tie(c, s) = Givens(R[j - 1][k], R[j][k]);

				if(typeid(T).name() == typeid(double).name()){
					_DQR((double*) R.Mat,R.row,R.col,j,double(c),double(s));
					_DQR((double*)Q.Mat,Q.row,Q.col,j,double(c),double(s));
				}
				else if(typeid(T).name() == typeid(float).name()){
					_FQR((float*) R.Mat,R.row,R.col,j,float(c),float(s));
					_FQR((float*)Q.Mat,Q.row,Q.col,j,float(c),float(s));
				}
				else{
				//R = G*R,change  j-1 and j  row ele
//#pragma omp paralle for
					for (size_t j2 = 0; j2 < _mSize_c; ++j2) {
						a1 = R[j - 1][j2] * c + R[j][j2] * s;
						R[j][j2] = R[j - 1][j2] * (-s) + R[j][j2] * c;
						R[j - 1][j2] = a1;
					}

					// Q = Q*GT
//#pragma omp paralle for
					for (size_t j2 = 0; j2 < _mSize_r; ++j2) {
						a2 = Q[j - 1][j2] * c + Q[j][j2] * s;
						Q[j][j2] = -s * Q[j - 1][j2] + c * Q[j][j2];
						Q[j - 1][j2] = a2;
 					}
				}
			}
		}
		Q.trans();
		return std::make_tuple(Q, R);
	}


	template<typename T>
	T Norm_2(const Matrix<T>& mat)
	{
		T res = 0;
		Matrix<T> ATA = Trans<T>(mat) * mat;
		std::tie(res, std::ignore) = Power_Method<T>(ATA);
		return sqrt(res);
	}

	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> PLU(const Matrix<T>& mat)
	{
		size_t _mSize_r, _mSize_c;
		std::tie(_mSize_r, _mSize_c) = mat.shape();

		Matrix<T> P(_mSize_r, _mSize_r);
		std::vector<size_t> P2(_mSize_r, 0);
		for (size_t i = 0; i < _mSize_r; ++i) {
			P2[i] = i;
		}
		Matrix<T> L(_mSize_r, _mSize_r, 0);
		Matrix<T> U(mat);
		size_t max_row_index;
		size_t current_col = 0;
		size_t i = 0;
		T ele;
		T* tmp_row = new T[_mSize_c];
		for (i = 0; i < _mSize_r && current_col < _mSize_c; ++i) {
			// 寻找该列最大值
			max_row_index = i;

			for (; current_col < _mSize_c; ++current_col) {
				for (size_t m = i; m < _mSize_r; ++m) {
					if (fabs(U[max_row_index][current_col]) < fabs(U[m][current_col])) {
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
				std::swap(P2[i], P2[max_row_index]);
				std::copy(U[i], U[i] + mat.col, tmp_row);
				std::copy(U[max_row_index], U[max_row_index] + mat.col, U[i]);
				std::copy(tmp_row, tmp_row + mat.col, U[max_row_index]);

				std::copy(L[i], L[i] + mat.col, tmp_row);
				std::copy(L[max_row_index], L[max_row_index] + mat.col, L[i]);
				std::copy(tmp_row, tmp_row + mat.col, L[max_row_index]);

			}

			//U中 i列下方元素变为0；
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

		//std::cout << L;
		L = L + Eye<T>(_mSize_r);
		
		//std::cout << P << L << U;

		return std::make_tuple(P, L, U);
	}

	template<typename T>
	std::tuple<Matrix<T>, Matrix<T>> LU(const Matrix<T>& mat)
	{
		if (mat.row != mat.col) {
			std::cerr << "in func LU(), Mat row and col size must equal.\n";
			return std::make_tuple(Matrix<T>(), Matrix<T>());
		}

		Matrix<T> L = Eye<T>(mat.row);
		Matrix<T> U(mat);
		size_t j, k;
		T ele;
		for (size_t i = 0; i < mat.row; ++i) {
			//U中 i列下方元素变为0；
			for (j = i + 1; j + 2 <= mat.row; j += 2) {
				ele = U[j][i] / U[i][i];
				L[j][i] = ele;
				k = 0;
				while (k < mat.col) {
					U[j][k] -= (U[i][k] * ele);
					++k;
				}
			}
			//std::cout << i<<"\n";
		}
		return std::make_tuple(L, U);
	}

	template<typename T>
	Matrix<T> Ceil(const Matrix<T>& mat)
	{
		Matrix<T> res(mat);
		size_t mat_size = res.size();
		size_t i = 0;
		while (i<mat_size)
		{
			res.Mat[i] = std::ceil(res.Mat[i]);
			++i;
		}
		return res;
	}

	template<typename T>
	Matrix<T> Floor(const Matrix<T>& mat)
	{
		Matrix<T> res(mat);
		size_t mat_size = res.size();
		size_t i = 0;
		while (i < mat_size)
		{
			res.Mat[i] = std::floor(res.Mat[i]);
			++i;
		}
		return res;
	}
	
	template<typename T>
	std::tuple<T, T >Givens(const T& a, const T& b) {
		T c, s;
		if (b == 0) {
			c = 1;
			s = 0;
		}
		else {
			if (fabs(b) > fabs(a)) {
				T t = a / b;
				s = 1.0 / sqrt(1 + pow(t, 2));
				c = s * t;
			}
			else {
				T t = b / a;
				c = 1.0 / sqrt(1 + pow(t, 2));
				s = c * t;
			}
		}
		return std::make_tuple(c, s);
	}

	template<typename T>
	Matrix<T> operator+(const Matrix<T>& mat, const Matrix<T>& mat2)
	{
		Matrix<T> res(mat);

		size_t mat_size = mat.size();
		size_t i = 0;

		while (i<mat_size)
		{
			res.Mat[i] += mat2.Mat[i];
			++i;
		}
		return res;
	}

	template<typename T>
	Matrix<T> operator-(const Matrix<T>& mat, const Matrix<T>& mat2)
	{
		Matrix<T> res(mat);

		size_t mat_size = mat.size();
		size_t i = 0;

		while (i < mat_size)
		{
			res.Mat[i] += mat2.Mat[i];
			++i;
		}
		return res;
	}


	void Test(int argc, char* argv[]) {

		Matrix<double> Da, Db, Dc;
		Matrix<float> Fa, Fb, Fc;
		for (int i = 1; i < argc;++i) {

			int ns = atoi(argv[i]);
			printf("\n");
			printf("--------------%d x %d ------------------",ns,ns);

			Da = RandI_Matrix<double>(ns, ns, 1.0, 10.0);
			Db = Da;
			Fa = RandI_Matrix<float>(ns, ns, 1.0f, 10.0f);
			Fb = Fa;

			printf("\n                        %20s   %20s\n","Float Matrix","Double Matrix");
			std::cout << "....\n";
		
			printf("%20s","cross product");
			auto bt = clock();
			Fc =  Fa * Fb;
			auto fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			Dc = Da * Db;
			auto det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));

			
			printf("\n%20s","Inverse");
			bt = clock();
			Fc =Inverse(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			Dc =Inverse(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","Dot");
			bt = clock();
			Fc =Dot(Fa,1.0f);
			Fc =Dot(Fa,Fb);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			Dc =Dot(Da,1.0);
			Dc =Dot(Da,Db);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));

			printf("\n%20s","Adjoint");
			bt = clock();
			Fc = Adjoint(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			Dc = Adjoint(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","PLU");
			bt = clock();
			std::tie(Fb,Fc,std::ignore) =PLU(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			std::tie(Db,Dc,std::ignore) =PLU(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","QR");
			bt = clock();
			std::tie(Fb,Fc) =QR<float>(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			std::tie(Db,Dc) =QR<double>(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","Det");
			bt = clock();
			float ff =Det<float>(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			double dd =Det<double>(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","Trans");
			bt = clock();
			Fc =Trans<float>(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			Dc =Trans<double>(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","Norm 2");
			bt = clock();
			ff =Norm_2<float>(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			dd =Norm_2<double>(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));


			printf("\n%20s","Power Method");
			bt = clock();
			std::tie(ff,Fc) = Power_Method<float>(Fa);
			fet = clock() - bt;
			printf("%20f s",1.0 * (1.0*fet / CLOCKS_PER_SEC));
			bt = clock();
			std::tie(dd,Dc) = Power_Method<double>(Da);
			det = clock() - bt;
			printf("%20f s",1.0 * (1.0*det / CLOCKS_PER_SEC));



			std::cout<<"\n";
		}	

	}



}





#endif // !U4_H
#define U4_H