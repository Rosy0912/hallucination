# 历史协议：答案 token 读取有误，不用于当前成绩

这些作业确实完成，但旧包装脚本采用 tokenizer.encode("Yes") / encode("No") 的最后一个 token（8241/3782）。固定 tokenizer 的 add_prefix_space=False，实际生成对应词首 token 3869/1939。旧 Base 评分及依赖其正确性标签的训练因此有偏差。AMBER 575/575 题的 APC 全部回退也是此错误的表现。

这里保留原始输出与历史复跑一致性证据，不能用来宣称修正版方法优于正确的 Base。四个独立包已修正并重新提取、训练、评测。CHAIR 的生成与评分本身不依赖答案 token，但旧训练标签受此问题影响。
