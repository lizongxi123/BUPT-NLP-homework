import random

general_ratio = 0.2
domain_ratio = 0.8

with open('E:/natrual-language-processing/nlp数据集/bert/清洗后的数据.txt', 'r', encoding='utf-8') as f1, \
     open('E:/natrual-language-processing/nlp数据集/bert/清洗后的数据2.txt', 'r', encoding='utf-8') as f2:
    general_lines = f1.readlines()
    domain_lines = f2.readlines()

    total = min(len(general_lines), len(domain_lines) // domain_ratio)
    num_general = int(total * general_ratio)
    num_domain = int(total * domain_ratio)

    sampled = random.sample(general_lines, num_general) + random.sample(domain_lines, num_domain)
    random.shuffle(sampled)

with open('E:/natrual-language-processing/nlp数据集/bert/混合数据.txt', 'w', encoding='utf-8') as fout:
    fout.writelines(sampled)

print("按比例混合完成，输出 balanced_mixed.txt")