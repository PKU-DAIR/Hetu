import hetu as ht

# 测试hetu是否有data_transfer方法
try:
    print("检查hetu.data_transfer是否存在:")
    if hasattr(ht, 'data_transfer'):
        print("hetu.data_transfer方法存在")
    else:
        print("hetu.data_transfer方法不存在")
        
    # 查看hetu模块中可用的方法和属性
    print("\nhetu模块中的可用方法和属性:")
    for attr in dir(ht):
        if not attr.startswith('_'):  # 排除私有属性
            print(f"- {attr}")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
