#pragma once

#include <cstring>
#include <sstream>
#include <cstdint>
#include <iomanip>
#include <vector>

namespace hetu {

inline bool parse_bool_slow_but_safe(const std::string& str, bool& r) {
  if (str == "True" || str == "true" || str == "1") {
    r = true;
    return true;
  } else if (str == "False" || str == "false" || str == "0") {
    r = false;
    return true;
  } else {
    return false;
  }
}

inline bool parse_int32_slow_but_safe(const std::string& str, int32_t& r) {
  std::istringstream iss(str);
  iss >> std::noskipws >> r;
  return iss.eof();
}

inline bool parse_int64_slow_but_safe(const std::string& str, int64_t& r) {
  std::istringstream iss(str);
  iss >> std::noskipws >> r;
  return iss.eof();
}

inline bool parse_float64_slow_but_safe(const std::string& str, double& r) {
  std::istringstream iss(str);
  iss >> std::noskipws >> r;
  return iss.eof();
}

inline std::string parse_int32_list_slow_but_safe(const std::string& str,
                                                  std::vector<int32_t>& r) {
  if (str.length() < 2 || str.front() != '[' || str.back() != ']')
    return "String \'" + str + "\' is not surrounded by square brackets";
  std::istringstream iss(str.substr(1, str.length() - 2));
  std::string token;
  int32_t parsed;
  while (std::getline(iss, token, ',')) {
    bool ok = parse_int32_slow_but_safe(token, parsed);
    if (!ok)
      return "Substring \'" + token + "\' cannot be parsed as an int64";
    r.push_back(parsed);
  }
  return "";
}

inline std::string parse_int64_list_slow_but_safe(const std::string& str,
                                                  std::vector<int64_t>& r) {
  if (str.length() < 2 || str.front() != '[' || str.back() != ']')
    return "String \'" + str + "\' is not surrounded by square brackets";
  std::istringstream iss(str.substr(1, str.length() - 2));
  std::string token;
  int64_t parsed;
  while (std::getline(iss, token, ',')) {
    bool ok = parse_int64_slow_but_safe(token, parsed);
    if (!ok)
      return "Substring \'" + token + "\' cannot be parsed as an int64";
    r.push_back(parsed);
  }
  return "";
}

inline std::string parse_float64_list_slow_but_safe(const std::string& str,
                                                    std::vector<double>& r) {
  if (str.length() < 2 || str.front() != '[' || str.back() != ']')
    return "String \'" + str + "\' is not surrounded by square brackets";
  std::istringstream iss(str.substr(1, str.length() - 2));
  std::string token;
  double parsed;
  while (std::getline(iss, token, ',')) {
    bool ok = parse_float64_slow_but_safe(token, parsed);
    if (!ok)
      return "Substring \'" + token + "\' cannot be parsed as a float64";
    r.push_back(parsed);
  }
  return "";
}

inline std::string parse_bool_list_slow_but_safe(const std::string& str,
                                                 std::vector<bool>& r) {
  if (str.length() < 2 || str.front() != '[' || str.back() != ']')
    return "String \'" + str + "\' is not surrounded by square brackets";
  std::istringstream iss(str.substr(1, str.length() - 2));
  std::string token;
  bool parsed;
  while (std::getline(iss, token, ',')) {
    bool ok = parse_bool_slow_but_safe(token, parsed);
    if (!ok)
      return "Substring \'" + token + "\' cannot be parsed as a bool";
    r.push_back(parsed);
  }
  return "";
}

inline std::string parse_bool_list_list_slow_but_safe(const std::string& str,
                                                 std::vector<std::vector<bool>>& r) {
  if (str.length() < 2 || str.front() != '[' || str.back() != ']')
    return "String \'" + str + "\' is not surrounded by square brackets";
  std::istringstream iss(str.substr(1, str.length() - 2));
  std::string token;
  std::vector<bool> parsed;
  while (std::getline(iss, token, ',')) {
    std::string ok = parse_bool_list_slow_but_safe(token, parsed);
    if (ok != "")
      return "Substring \'" + token + "\' cannot be parsed as a bool list";
    r.push_back(parsed);
  }
  return "";
}

inline std::string parse_string_literal(const std::string& str, std::string& r,
                                        bool quoted = true) {
  size_t begin = 0, end = str.size();
  if (quoted) {
    if (str.length() < 2 ||
        (!(str.front() == '"' && str.back() == '"') &&
         !(str.front() == '\'' && str.back() == '\'')))
      return "String \'" + str + "\' is not quoted";
    begin = 1;
    end = str.size();
  }

  r.reserve(end - begin);
  for (size_t i = begin; i < end - 1; i++) {
    if (str.at(i) != '\\') {
      r.push_back(str.at(i));
    } else {
      // https://en.cppreference.com/w/cpp/language/escape
      if (i + 1 == end - 1)
        return "String " + str + " ends with unfinished escape";
      switch (str.at(i + 1)) {
        case '\\': break;
        case 'a': r.push_back('\a'); break;
        case 'b': r.push_back('\b'); break;
        case 'f': r.push_back('\f'); break;
        case 'n': r.push_back('\n'); break;
        case 'v': r.push_back('\v'); break;
        case 't': r.push_back('\t'); break;
        case '\'': r.push_back('\''); break;
        case '"': r.push_back('"'); break;
        default:
          return "String " + str + " contains unsupported escape at position " +
            std::to_string(i);
      }
      i++;
    }
  }
  return "";
}

inline void ltrim(std::string& str, const char* t = " \t\n\r\f\v") {
  str.erase(0, str.find_first_not_of(t));
}

inline void rtrim(std::string& str, const char* t = " \t\n\r\f\v") {
  str.erase(str.find_last_not_of(t) + 1);
}

inline void trim(std::string& str, const char* t = " \t\n\r\f\v") {
  ltrim(str, t);
  rtrim(str, t);
}

} // namespace hetu
