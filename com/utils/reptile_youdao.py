import requests
from bs4 import BeautifulSoup

format_url = "http://dict.youdao.com/search?q=%s&keyfrom=new-fanyi.smartResult"
param = {
    "Host" : "dict.youdao.com",
    "Upgrade-Insecure-Requests" : 1,
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"

}
read_file_path = "E:\\Alls\\code\\python\\words.txt"
write_file_path = "E:\Alls\code\python\\result.txt"

def word_request(word):
    url = format_url % (word)
    response = requests.get(url, params=param)
    if response.status_code == 200:
        html = response.text
        return parse_html(html)
    else:
        print("%s word request fail" % (word))

def parse_html(html):
    result = []
    soup = BeautifulSoup(html, "lxml")
    #获取word
    word = soup.find(name="span", attrs="keyword").text
    result.append(word)

    #获取所有的语音信息span
    voices = soup.find(attrs="baav").find_all(name="span", attrs="pronounce")
    voice_results = []
    for voice in voices:
        voice_result = voice.text.split(" ")[0].strip() + " " + voice.find(attrs="phonetic").text
        voice_results.append(voice_result)
    result.append(voice_results)
    #获取所有的中文翻译li
    values = soup.find(attrs="trans-wrapper clearfix").find(name="ul").find_all(name="li")
    value_result = []
    for value in values:
        value_result.append(char_deal_with(value.text))
    result.append(value_result)
    return result

def read_file():
    with open(read_file_path, "r") as f:
        for line in f.readlines():
            try:
                write_result(word_request(line))
            except Exception:
                print(line.strip(), "fail!")

def write_result(value):
    with open(write_file_path, "a", encoding="utf-8") as f:
        f.writelines(value[0] + "  " + "、".join(value[1]) + " " + "、".join(value[2]) + "\n")

def char_deal_with(value):
    chars = ["adj.", "n.", "v.", "vt.", "vi."]
    for char in chars:
        if value.find(char) != -1:
            return value.replace(char, "")
    return value

if __name__ == "__main__":
    read_file()