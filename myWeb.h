#include <tuple>
#include <regex>


#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include "UMatrix.h"


namespace ubzx {

    constexpr int WEB_PORT = 80;
        constexpr size_t Buffer_Size= 4096;
        enum ProcessMode { PM_GET, PM_POST, PM_PUT, PM_DELETE };   // 

    bool run_server();
        std::string response_header_succes(const std::size_t& Content_Length, const std::string& Content_Type = "text/html;");
        std::string response_head_failed(const std::size_t& Content_Length, const std::string& Content_Type = "text/html;");
        std::tuple<ProcessMode, std::string> process(std::string& content);
        bool process_get(std::string& command, int& sockConn, std::string& root_path);
        bool process_post(std::string& command, std::string &recv_str, int& sockConn, std::string& root_path);
        std::string responseMat_and_Html(std::string& command, const uubzx::Matrix& Mat);

        // ubuntu
        bool run_server() {
                char recv_buff[Buffer_Size];
                int serv_sock, clnt_sock;
                struct sockaddr_in serv_addr;
                struct sockaddr_in clnt_addr;
                socklen_t clnt_addr_size;
                ProcessMode PM;
                std::string tmp_content;
                std::string recv_str;
                std::string root_path = ".//Server//";

                serv_sock = socket(AF_INET, SOCK_STREAM, 0);
                if (serv_sock == -1) {
                        std::cerr << "socket() eror!\n";
                        return false;
                }

                memset(&serv_addr, 0, sizeof(serv_addr));
                serv_addr.sin_family = AF_INET;
                serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
                serv_addr.sin_port = htons(80);

                if (bind(serv_sock, (struct sockaddr*) & serv_addr, sizeof(serv_addr)) == -1) {
                        std::cerr << "bind() error!\n";
                        return false;
                }
                if (listen(serv_sock, 10) == -1) {
                        std::cerr << "Listen() error!\n";
                        return false;
                }

                clnt_addr_size = sizeof(clnt_addr);

                while (true)
                {
                        std::cout << "ready recv!\n";
                        clnt_sock = accept(serv_sock, (struct sockaddr*) & clnt_addr, &clnt_addr_size);
                        if (clnt_sock == -1) {
                                continue;
                        }

                        memset(recv_buff, '\0', Buffer_Size);
                        read(clnt_sock, recv_buff, Buffer_Size);

                        recv_str = recv_buff;
                        std::tie(PM, tmp_content) = process(recv_str);

                        switch (PM)
                        {
                        case ubzx::PM_GET:
                                process_get(tmp_content, clnt_sock, root_path);
                                break;
                        case ubzx::PM_POST:
                                process_post(tmp_content, recv_str, clnt_sock, root_path);
                                break;
                        case ubzx::PM_PUT:
                                break;
                        case ubzx::PM_DELETE:
                                break;
                        default:
                                break;
                        }

                        close(clnt_sock);
                }

                close(serv_sock);
                return true;
        }


        std::string response_header_succes(const std::size_t &Content_Length, const std::string &Content_Type) {

                std::string H = "HTTP/1.1 200 OK\r\n";
                //std::string   D = "Date: " + formatdate(None, usegmt = True) + "\r\n";
                std::string     CT = "Content-Type: " + Content_Type + "\r\n";
                std::string     CL = "Content-Length: " + std::to_string(Content_Length) + "\r\n\r\n";
                //std::string res = H + D + CT + CL;
                std::string res = H + CT + CL;
                return res;
        }


        std::string response_head_failed(const std::size_t& Content_Length, const std::string& Content_Type) {
                std::string H = "HTTP/1.1 404 NOT FOUND\r\n";
                //std::string   D = "Date: " + formatdate(None, usegmt = True) + "\r\n";
                std::string     CT = "Content-Type: " + Content_Type + "\r\n";
                std::string     CL = "Content-Length: " + std::to_string(Content_Length) + "\r\n\r\n";
                //std::string res = H + D + CT + CL;
                std::string res = H + CT + CL;

                return res;
        }

        /*
                ???¨ª????????????
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
                ????GET???¨°??????????¡¤????¨¤??????
        */
        bool process_get(std::string& command,int &sockConn,std::string &root_path) {
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
                write(sockConn, response_content.c_str(), response_content.size());
                write(sockConn, read_content.c_str(), read_content.size());
                return true;
        }

        /*
                ?¨´??POST????????Matrix???????¨ª?¨¤??????????¡¤????¨¢??
        */
        bool process_post(std::string& command, std::string& recv_str, int& sockConn, std::string& root_path)
        {
                ubzx::Matrix Mat(3, 3);

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
                read_content = (read_content + "<br>" + responseMat_and_Html(command,Mat));
                size_t read_content_size = read_content.size();
                response_content = response_header_succes(read_content_size);
                write(sockConn, response_content.c_str(), response_content.size());
                write(sockConn, read_content.c_str(), read_content.size());

                return true;
        }

        /*
                ?¨´??????¡¤????????¨²???¨¤??¡Á?¡¤???
        */
        std::string responseMat_and_Html(std::string& command ,const ubzx::Matrix &Mat) {
                std::string ans = "";
                if (command == "Inverse") {
                        Matrix  res = ubzx::Inverse(Mat);
                        ans = command +"<br>"+ res.to_str();
                }
                else if (command == "LU") {
                        Matrix L, U;
                        std::tie(L, U) = ubzx::LU(Mat);
                        ans = command + "<br>" + "L:<br>" + L.to_str() + "<br>U:<br>" + U.to_str();
                }
                else if (command == "Adjoint")
                {
                        Matrix A;
                        A = ubzx::Adjoint(Mat);
                        ans = command + "<br>" + "Adjoint Matrix:<br>" + A.to_str();
                }
                else if (command == "QR") {
                        Matrix Q, R;
                        std::tie(Q, R) = ubzx::QR(Mat);
                        ans = command + "<br>" + "Q:<br>" + Q.to_str() + "<br>R:<br>" + R.to_str();
                }
                else if (command == "PLU") {
                        Matrix P,L, U;
                        std::tie(P,L, U) = ubzx::PLU(Mat);
                        ans = command + "<br>" + "P:<br>" + P.to_str()+ "L:<br>" + L.to_str() + "<br>U:<br>" + U.to_str();
                }
                return ans;
        }

}

#endif // !MYWEB