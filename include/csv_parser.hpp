// csv_parser.hpp
// Simple CSV parser / 简单CSV解析器
#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

namespace biot {

    class CSVParser {
    public:
        // Parse a CSV file / 解析CSV文件
        static bool parse_file(
            const std::string& filepath,
            std::vector<std::vector<std::string>>& rows,
            bool skip_header = true
        ) {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                return false;
            }

            std::string line;
            bool first_line = true;

            while (std::getline(file, line)) {
                // Skip header if requested / 跳过表头
                if (first_line && skip_header) {
                    first_line = false;
                    continue;
                }

                std::vector<std::string> row = parse_line(line);
                if (!row.empty()) {
                    rows.push_back(row);
                }
            }

            file.close();
            return true;
        }

        // Parse a single CSV line / 解析单行CSV
        static std::vector<std::string> parse_line(const std::string& line) {
            std::vector<std::string> tokens;
            std::string token;
            bool in_quotes = false;

            for (size_t i = 0; i < line.size(); ++i) {
                char c = line[i];

                if (c == '"') {
                    in_quotes = !in_quotes;
                }
                else if (c == ',' && !in_quotes) {
                    tokens.push_back(trim(token));
                    token.clear();
                }
                else {
                    token += c;
                }
            }

            // Add last token / 添加最后一个token
            tokens.push_back(trim(token));

            return tokens;
        }

        // Trim whitespace / 去除空白字符
        static std::string trim(const std::string& str) {
            size_t first = str.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";

            size_t last = str.find_last_not_of(" \t\r\n");
            return str.substr(first, last - first + 1);
        }

        // Convert string to double safely / 安全地转换字符串到double
        static bool to_double(const std::string& str, double& value) {
            try {
                value = std::stod(str);
                return true;
            }
            catch (...) {
                return false;
            }
        }
    };

} // namespace biot#pragma once
