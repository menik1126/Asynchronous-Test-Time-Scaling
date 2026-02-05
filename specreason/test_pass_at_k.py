#!/usr/bin/env python3
"""
测试Pass@k算法的脚本
"""

def combination(n, k):
    """计算组合数 C(n, k)"""
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def calculate_pass_at_k(n, c, k):
    """计算Pass@k"""
    if c == 0:
        return 0.0
    elif c >= k:
        return 1.0
    else:
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        c_n_k = combination(n, k)
        c_n_minus_c_k = combination(n - c, k)
        
        if c_n_k > 0:
            pass_at_k = 1.0 - (c_n_minus_c_k / c_n_k)
            return round(pass_at_k, 4)
        else:
            return 0.0

# 测试用例
test_cases = [
    (16, 16, "全部正确"),  # 16次尝试，16次正确
    (16, 1, "1次正确"),    # 16次尝试，1次正确
    (16, 3, "3次正确"),    # 16次尝试，3次正确
    (16, 0, "0次正确"),    # 16次尝试，0次正确
]

print("Pass@k 算法测试")
print("=" * 50)

for n, c, desc in test_cases:
    print(f"\n{n}次尝试，{c}次正确 ({desc}):")
    for k in [1, 3, 5, 10, 16]:
        result = calculate_pass_at_k(n, c, k)
        print(f"  Pass@{k}: {result}")
        
        # 显示计算过程
        if 0 < c < k:
            c_n_k = combination(n, k)
            c_n_minus_c_k = combination(n - c, k)
            print(f"    C({n},{k}) = {c_n_k}, C({n-c},{k}) = {c_n_minus_c_k}")
            print(f"    Pass@{k} = 1 - {c_n_minus_c_k}/{c_n_k} = {result}") 