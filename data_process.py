import ujson, os
import pyarrow as pa
import pyarrow.parquet as pq
from unicodedata import normalize
from collections import defaultdict
import pandas as pd
import random
from transformers import Qwen2Tokenizer
from tqdm import tqdm, trange


def process_none(s: str) -> str:
    if s: return s
    return ""


def gen_baike(origin_file, output_dir, eos_token):
    if not eos_token:
        assert "It should have an eos_token defined by Tokenizer."
    
    baike_items, max_len, batch_size, batch_cnt = [], 1500, 200000, 0  # 仅保留长度小于 max_len 的
    with open(origin_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line: break
            item = ujson.loads(line)
            cur_txt, cur_len = [], 0
            if not item["title"]:  continue
            temp_txt = f"{item['title']}：{process_none(item['summary'])}"
            cur_len += len(temp_txt)
            cur_txt.append(temp_txt)
            
            for section in item["sections"]:
                # 太长的截断不要了
                if cur_len > max_len: break
                title = f"{section['title']}：" if section["title"] else ""
                temp_txt = f"{title}{process_none(section['content'])}"
                cur_len += len(temp_txt)
                cur_txt.append(temp_txt)
            temp_txt = normalize("NFKC", "".join(cur_txt))
            
            if len(temp_txt) > max_len:
                # 从 max_len 开始找第一个句号，叹号
                n, i = len(temp_txt), max_len
                while i < n and temp_txt[i] not in ("。", "！"):
                    i += 1
                temp_txt = "".join(temp_txt[0: i + 1])
                
                # 添加 eos token
            temp_txt = f"{temp_txt}{eos_token}"
            baike_items.append(temp_txt)
            
            if len(baike_items) % batch_size == 0:
                chunk_data = split_txt_cropus_to_chunk_data(baike_items)
                tb = pa.Table.from_arrays([chunk_data], names=["text"])
                file_name = os.path.join(output_dir, f"baike_chunk_1500_5.6M_{batch_cnt}.parquet")
                pq.write_table(
                    table=tb,
                    where=file_name,
                    row_group_size=50000,
                )
                print(f"save to {file_name}")
                batch_cnt += 1
                baike_items = []


def split_txt_cropus_to_chunk_data(
        texts: list, batch_size: int = 512 ** 2, max_len: int = 1024, window_size: int = 2
) -> list:
    buffer, buffer_len = [], 0
    chunk_data = []
    
    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)
        
        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)
            
            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):
                chunk_data.append("".join(buffer_txt[i: i + max_len]))
            
            buffer, buffer_len = [], 0
    
    return chunk_data


def gen_wiki_filter(origin_file, eos_token, test_length=-1,
                    output_file="../datasets/wiki_fi.parquet"):
    if not eos_token:
        assert "It should have an eos_token defined by Tokenizer."
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)
        for i, item in tqdm(enumerate(items)):
            if test_length != -1 and i >= test_length: break
            lines.append(item["completion"] + eos_token)
    chunk_data = split_txt_cropus_to_chunk_data(lines)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )


def get_length_dist(input_folder):
    import matplotlib.pyplot as plt
    # 初始化长度统计字典
    length_dict = {}
    max_length = -1
    
    # 遍历 JSON Lines 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            ppath = os.path.join(input_folder, filename)
            with open(ppath, "r", encoding="utf-8") as f:
                for line in tqdm(f, f"Processing {ppath}"):
                    # 解析 JSON 行数据
                    data = ujson.loads(line.strip())
                    # 计算样本长度
                    sample_length = len(data["text"])
                    # if sample_length > 2000:
                    #     print(data["text"])
                    max_length = max(max_length, sample_length)
                    # 将长度按照 10 为单位取整
                    rounded_length = sample_length // 100 * 100
                    # 更新长度统计字典
                    length_dict[rounded_length] = length_dict.get(rounded_length, 0) + 1
    
    # 获取最大长度和最大频率
    max_length = max(length_dict.keys())
    max_frequency = max(length_dict.values())
    
    # 绘制长度分布图
    plt.bar(length_dict.keys(), length_dict.values())
    plt.xlabel("Sample Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sample Length")
    plt.xlim(0, max_length)
    plt.ylim(0, max_frequency)
    plt.show()


def gen_sky(input_folder, output_folder, eos_token):
    if not eos_token:
        assert "It should have an eos_token defined by Tokenizer."
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):  # 修改为处理JSON Lines文件
            origin_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, filename.replace(".jsonl", ".parquet")
            )
            
            lines = []
            with open(origin_file, "r", encoding="utf-8") as f:
                for line in tqdm(f, f"Processing {origin_file}"):
                    item = ujson.loads(line)
                    lines.append(item["text"] + eos_token)  # 确保每行都是一个有效的JSON对象
            
            if lines:  # 确保文件中有内容
                chunk_data = split_txt_cropus_to_chunk_data(lines)
                tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
                pq.write_table(
                    table=tb,
                    where=output_file,
                    row_group_size=50000,
                    data_page_size=50000,
                )
                print(f"Processed {origin_file} to {output_file}")
            else:
                print(f"No content in {origin_file}. Skipping.")


def shuffle(dirs, chunk_size, output_dir):
    class data_iterator:
        def __init__(self, directory):
            self.directory = directory
        
        def get_data(self):
            for root, _, files in tqdm(os.walk(self.directory), f"Loop through {self.directory}"):
                for file in files:
                    if file.endswith('.parquet') and not file.startswith("dev"):
                        # print("Read file", file)
                        for i in pq.read_table(os.path.join(root, file))["text"]:
                            yield i
    
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    
    weights = {'data/BaiKe': 0.09753958865159912,
               'data/skypile/parquet': 0.6198822346945129,
               'data/wiki_chinese': 0.282578176653888}
    
    bake_data_iterater = data_iterator('data/BaiKe')
    skypile_data_iterater = data_iterator('data/skypile/parquet')
    wiki_data_iterater = data_iterator('data/wiki_chinese')
    
    # 将样本划分为每个 chunk_size 大小的新的 parquet 文件
    chunk_counter = 1
    current_chunk_size = 0
    current_chunk_data = []
    
    # 如果当前 chunk 达到指定大小，则保存为一个 parquet 文件
    for _ in trange(12013064):
        selected_key = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        
        if selected_key == 'data/BaiKe':
            sample = next(bake_data_iterater.get_data(), None)
        elif selected_key == 'data/skypile/parquet':
            sample = next(skypile_data_iterater.get_data(), None)
        elif selected_key == 'data/wiki_chinese':
            sample = next(wiki_data_iterater.get_data(), None)
        
        if sample is None:
            del weights[selected_key]
        else:
            current_chunk_data.append(sample)
            current_chunk_size += 1
            
            if current_chunk_size >= chunk_size:
                pq.write_table(
                    table=pa.Table.from_arrays([pa.array(current_chunk_data)], names=["text"]),
                    where=os.path.join(output_dir, f"chunk_{chunk_counter}.parquet"),
                    row_group_size=50000,
                    data_page_size=50000,
                )
                # 重置当前 chunk 相关变量
                chunk_counter += 1
                current_chunk_size = 0
                current_chunk_data = []
    
    # 如果有剩余样本，保存为一个额外的 chunk
    if current_chunk_data:
        pq.write_table(
            table=pa.Table.from_arrays([pa.array(current_chunk_data)], names=["text"]),
            where=os.path.join(output_dir, f"chunk_{chunk_counter}.parquet"),
            row_group_size=50000,
            data_page_size=50000,
        )


if __name__ == "__main__":
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")
    
    # gen_wiki_filter(
    #     "data/wiki_chinese/wikipedia-cn-20230720-filtered.json",
    #     eos_token=tokenizer.eos_token,
    #     test_length=-1,  # 只选择100条进行测试，全量则设置为 -1
    #     output_file="data/wiki_chinese/wiki_fi.parquet"
    # )
    
    # get_length_dist("data/skypile")
    # gen_sky("data/skypile", "data/skypile/parquet", tokenizer.eos_token)
    # gen_baike("data/BaiKe/563w_baidubaike.json", "data/BaiKe", tokenizer.eos_token)
    
    shuffle(["data/BaiKe", "data/skypile/parquet", "data/wiki_chinese"], 512 * 2 * 400, "data/split_parquet")
