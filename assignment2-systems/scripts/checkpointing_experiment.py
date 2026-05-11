#不考虑 compute cost，最省 peak activation memory 的策略是递归嵌套 checkpoint

# def recursive_checkpoint(blocks, x):
#     if len(blocks) == 1:
#         return checkpoint(blocks[0], x, use_reentrant=False)

#     mid = len(blocks) // 2

#     def left_fn(x):
#         return recursive_checkpoint(blocks[:mid], x)

#     def right_fn(x):
#         return recursive_checkpoint(blocks[mid:], x)

#     x = checkpoint(left_fn, x, use_reentrant=False)
#     x = checkpoint(right_fn, x, use_reentrant=False)
#     return x

#      [ B1 B2 B3 B4 B5 B6 B7 B8 ]
#         /                 \
# [ B1 B2 B3 B4 ]      [ B5 B6 B7 B8 ]
#     /      \            /      \
# [ B1 B2 ] [ B3 B4 ]  [ B5 B6 ] [ B7 B8 ]
#   /  \      /  \       /  \      /  \
# B1    B2  B3    B4    B5  B6    B7   B8

# def group_1_to_8(x):
#     x = checkpoint(B1, x)
#     x = checkpoint(B2, x)
#     x = checkpoint(B3, x)
#     x = checkpoint(B4, x)
#     x = checkpoint(B5, x)
#     x = checkpoint(B6, x)
#     x = checkpoint(B7, x)
#     x = checkpoint(B8, x)
#     return x

# x = checkpoint(group_1_to_8, x)


#如果 checkpoint 的最小粒度只到 TransformerBlock，并且内层已经对每个 block 单独 checkpoint，那么它和递归二分 checkpoint 在 block 级 residual 保存上非常接近；
#递归二分的额外优势主要来自继续减少边界/重算峰值，或者进一步拆分 block 内部，但代价是更多重算和调度复杂度。