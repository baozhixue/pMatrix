#ifndef MYWEB
#define MYWEB

#include "Matrix.h"
#include <WinSock2.h>
#pragma comment(lib, "ws2_32.lib")
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <regex>



namespace bzx {

    constexpr int WEB_PORT = 80;
	constexpr size_t Buffer_Size= 4096;
	enum class ProcessMode { PM_GET, PM_POST, PM_PUT, PM_DELETE };   // 
	static struct MyStruct
	{
		std::string read_content;
		std::string response_content;
	}MyStruct;


    bool run_server();
	std::string response_header_succes(const std::size_t& Content_Length, const std::string& Content_Type = "text/html;");
	std::string response_head_failed(const std::size_t& Content_Length, const std::string& Content_Type = "text/html;");
	std::tuple<ProcessMode, std::string> process(std::string& content);
	bool process_get(std::string& command, SOCKET& sockConn, std::string& root_path);
	bool process_post(std::string& command, std::string &recv_str,SOCKET& sockConn, std::string& root_path);
	std::string responseMat_and_Html(std::string& command, const bzx::Matrix& Mat);



    bool run_server(){
		char recv_buff[Buffer_Size];
		memset(recv_buff, '\0', Buffer_Size);
		SOCKET sockSrv, sockConn;
		SOCKADDR_IN addrSrv, addrClient;
		WSADATA wsaData;
		
		if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
		{
			std::cout << "Failed to load WinSock!\n";
			return false;
		}

		addrSrv.sin_family = AF_INET;
		addrSrv.sin_port = htons(WEB_PORT);
		addrSrv.sin_addr.S_un.S_addr = htonl(INADDR_ANY);

		sockSrv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

		int retVal = bind(sockSrv, (LPSOCKADDR)&addrSrv, sizeof(SOCKADDR));
		if (retVal == SOCKET_ERROR) {
			std::cout << "Failed bind " << WSAGetLastError() << "\n";
			return false;
		}
		if (listen(sockSrv, 5) == SOCKET_ERROR)
		{
			std::cout << "Listen failed: " << WSAGetLastError() << "\n";
			return false;
		}

		std::ifstream inFile;

		std::string root_path = ".//Server//";
		int len = sizeof(SOCKADDR);

		ProcessMode PM;
		std::string tmp_content;
		std::string recv_str;

		

		while (true)
		{
			std::cout << "ready recv!\n";
			sockConn = accept(sockSrv, (SOCKADDR*)&addrClient, &len);
			if (sockConn == SOCKET_ERROR) {
				printf("Accept failed:%d\n", WSAGetLastError());
				continue;
			}
			memset(recv_buff, '\0', Buffer_Size);
			int Len = recv(sockConn, recv_buff, Buffer_Size,0);
			recv_str = recv_buff;
			std::tie(PM,tmp_content) = process(recv_str);
			std::string str = "/";

			switch (PM)
			{
			case bzx::ProcessMode::PM_GET:
				process_get(tmp_content, sockConn, root_path);
				break;
			case bzx::ProcessMode::PM_POST:
				process_post(tmp_content, recv_str, sockConn, root_path);
				break;
			case bzx::ProcessMode::PM_PUT:
				break;
			case bzx::ProcessMode::PM_DELETE:
				break;
			default:
				break;
			}

			std::cout << "doen an server!\n";
			closesocket(sockConn);
		}

		return true;
    }


	std::string response_header_succes(const std::size_t &Content_Length, const std::string &Content_Type) {
		
		std::string H = "HTTP/1.1 200 OK\r\n";
		//std::string	D = "Date: " + formatdate(None, usegmt = True) + "\r\n";
		std::string	CT = "Content-Type: " + Content_Type + "\r\n";
		std::string	CL = "Content-Length: " + std::to_string(Content_Length) + "\r\n\r\n";
		//std::string res = H + D + CT + CL;
		std::string res = H + CT + CL;
		return res;
	}


	std::string response_head_failed(const std::size_t& Content_Length, const std::string& Content_Type) {
		std::string H = "HTTP/1.1 404 NOT FOUND\r\n";
		//std::string	D = "Date: " + formatdate(None, usegmt = True) + "\r\n";
		std::string	CT = "Content-Type: " + Content_Type + "\r\n";
		std::string	CL = "Content-Length: " + std::to_string(Content_Length) + "\r\n\r\n";
		//std::string res = H + D + CT + CL;
		std::string res = H + CT + CL;

		return res;
	}

	/*
		处理接收到的内容
	*/
	std::tuple<ProcessMode, std::string> process(std::string& content)
	{
		std::istringstream read_buffer(content);
		std::string read_content;
		read_buffer >> read_content;

		ProcessMode PM;
		if (read_content=="GET") // Get
		{
			PM = ProcessMode::PM_GET;
		}
		else if (read_content == "DELETE") {
			PM = ProcessMode::PM_DELETE;
		}
		else if (read_content == "PUT") {
			PM = ProcessMode::PM_PUT;
		}
		else if (read_content == "POST") {
			PM = ProcessMode::PM_POST;
		}
		read_buffer >> read_content;
		return std::tuple<ProcessMode, std::string>(PM, read_content);
	}

	/*
		若为GET，则使用此函数发送相应网页
	*/
	bool process_get(std::string& command,SOCKET &sockConn,std::string &root_path) {
		std::ifstream inFile;
		std::string read_content;
		std::string response_content;
		std::cout << command<<"\n";
		//if (command == "/") {
			inFile.open(root_path + "hello.html",std::ios::in);
			std::stringstream read_buf;
			read_buf << inFile.rdbuf();
			read_content = read_buf.str();
			size_t read_content_size = read_content.size();
			response_content = response_header_succes(read_content_size);
		//}
		send(sockConn, response_content.c_str(), response_content.size(), 0);
		send(sockConn, read_content.c_str(), read_content.size(), 0);
		return true;
	}

	/*
		根据POST内容选择Matrix函数处理相应数据，并返回结果
	*/
	bool process_post(std::string& command, std::string& recv_str, SOCKET& sockConn, std::string& root_path)
	{
		bzx::Matrix Mat(3, 3);

		std::string num = "0";
		std::string str = recv_str.substr(recv_str.find_last_of("\n"));
		str = str.substr(0, str.find_last_of('&'));
		command = recv_str.substr(recv_str.find_last_of('=')+1);
		size_t step = 4;
		size_t index = 0;

		for (size_t i = 0; i <= str.length(); ++i) {

			if (i < str.length() && str[i] >= '0' && str[i] <= '9') {
				num += str[i];
			}
			else if (i == str.length() ||str[i] == '&') {
				Mat[index / 3][index % 3] = std::strtod(num.c_str(),0);
				num = "0";
				++index;
			}
		}
		
		std::ifstream inFile;
		std::string read_content;
		std::string response_content;
		inFile.open(root_path + "hello.html",std::ios::in);
		std::stringstream read_buf;
		read_buf << inFile.rdbuf();
		read_content = read_buf.str();
		read_content = (read_content + "\n" + responseMat_and_Html(command,Mat));
		size_t read_content_size = read_content.size();
		response_content = response_header_succes(read_content_size);
		send(sockConn, response_content.c_str(), response_content.size(), 0);
		send(sockConn, read_content.c_str(), read_content.size(), 0);

		return true;
	}

	/*
		根据函数返回内容生成相应字符串
	*/
	std::string responseMat_and_Html(std::string& command ,const bzx::Matrix &Mat) {
		std::string ans = "";
		if (command == "LU") {
			Matrix L, U;
			std::tie(L, U) = bzx::LU(Mat);
			ans = command + "\n原矩阵\n" + Mat.to_str() + "\n" + "\nL:\n" + L.to_str() + "\nU:\n" + U.to_str();
		}
		else if (command == "Adjoint")
		{
			Matrix A;
			A = bzx::ADJOINT_Matrix(Mat);
			ans = command + "\n原矩阵\n" + Mat.to_str() + "\n" +  "\nAdjoint Matrix:\n" + A.to_str();
		}
		else if (command == "QR") {
			Matrix Q, R;
			std::tie(Q, R) = bzx::JACOBI(Mat);
			ans = command + "\n原矩阵\n" + Mat.to_str() + "\n" +  "\nQ:\n" + Q.to_str() + "\nR:\n" + R.to_str();
		}
		else if (command == "PLU") {
			Matrix P,L, U;
			std::tie(P,L, U) = bzx::PLU(Mat);
			ans = command + "\n原矩阵\n" + Mat.to_str() + "\n" +"\nP:\n" + P.to_str()+ "\nL:\n" + L.to_str() + "\nU:\n" + U.to_str();
		}
		else if (command == "SVD") {
			Matrix U,S,V;
			std::tie(U, S, V) = bzx::SVD(Mat);

			ans = command + "\n原矩阵\n" + Mat.to_str() + "\n" + "\nU: \n" + U.to_str() + "S: \n" + S.to_str() + "V: \n" + V.to_str();
		}
		return ans;
	}

}

#endif // !MYWEB
#pragma once
