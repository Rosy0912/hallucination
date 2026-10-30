"""Answer token IDs for the pinned LLaVA-1.5 chat prompt ending in ASSISTANT:."""
def llava_answer_ids(tokenizer):
    # This tokenizer has add_prefix_space=False: bare Yes/No are *different*
    # low-probability tokens. Actual next-token generation uses word-start forms.
    encoded=[tokenizer.encode(s,add_special_tokens=False) for s in (' Yes',' No')]
    if any(len(ids)!=1 for ids in encoded):
        raise ValueError('Expected single word-start Yes/No tokens for pinned LLaVA tokenizer')
    ids=tuple(int(x[0]) for x in encoded)
    if ids!=(3869,1939):
        raise ValueError(f'Unexpected answer tokens {ids}; audit tokenizer/model revision before running')
    return ids
