import re

with open("flash_attn_rope_fused.py", "r") as f:
    content = f.read()

# Put back @cute.jit
content = content.replace("    def _rotate_smem_inplace(", "    @cute.jit\n    def _rotate_smem_inplace(")
content = content.replace("    def _load_cos_sin_K(", "    @cute.jit\n    def _load_cos_sin_K(")
content = content.replace("    def compute_one_n_block(", "    @cute.jit\n    def compute_one_n_block(")
content = content.replace("    def softmax_rescale_O(", "    @cute.jit\n    def softmax_rescale_O(")
content = content.replace("    def normalize_softmax(", "    @cute.jit\n    def normalize_softmax(")

# Remove namespace prefixes everywhere inside the file
content = content.replace("basic_params.", "")
content = content.replace("mma_params.", "")
content = content.replace("gmem_copy_params.", "")
content = content.replace("smem_copy_params.", "")
content = content.replace("softmax_params.", "")

# Now fix the method signatures
content = re.sub(
    r"def compute_one_n_block\(\s*self,\s*basic_params:[^,]+,\s*mma_params:[^,]+,\s*gmem_copy_params:[^,]+,\s*smem_copy_params:[^,]+,\s*softmax_params:[^,]+,",
    "def compute_one_n_block(self, mQ, mK, mCos, mSin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,",
    content
)

content = re.sub(
    r"def softmax_rescale_O\(\s*self,\s*basic_params:[^,]+,\s*mma_params:[^,]+,\s*softmax_params:[^,]+,",
    "def softmax_rescale_O(self, mQ, mK, m_block_mask, acc_O, thr_mma, row_max, row_sum, softmax_scale_log2,",
    content
)

content = re.sub(
    r"def _load_cos_sin_K\(\s*self,\s*basic_params:[^,]+,",
    "def _load_cos_sin_K(self, mK, mCos, mSin, sCosK, sSinK,",
    content
)

# Now fix the calls!
content = content.replace(
    "self.compute_one_n_block(\n                basic_params, mma_params, gmem_copy_params, smem_copy_params,\n                softmax_params,",
    "self.compute_one_n_block(\n                mQ, mK, m_cos, m_sin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,"
)
content = content.replace(
    "self.compute_one_n_block(\n            basic_params, mma_params, gmem_copy_params, smem_copy_params,\n            softmax_params,",
    "self.compute_one_n_block(\n            mQ, mK, m_cos, m_sin, m_block_mask, sK, sCosK, sSinK, thr_mma, tiled_mma, tSrQ, tSrK, tOrVt, acc_O, gmem_tiled_copy_QKV, tKVcKV, tKgK, tKsK, tVgV, tVsV, tKVpKV, smem_tiled_copy_Q, smem_tiled_copy_K, smem_tiled_copy_V, tSsQ, tSrQ_copy_view, tSsK, tSrK_copy_view, tOsVt, tOrVt_copy_view, row_max, row_sum, softmax_scale_log2,"
)

content = content.replace(
    "self.softmax_rescale_O(\n                basic_params, mma_params, softmax_params,",
    "self.softmax_rescale_O(\n                mQ, mK, m_block_mask, acc_O, thr_mma, row_max, row_sum, softmax_scale_log2,"
)

content = content.replace(
    "self._load_cos_sin_K(basic_params, n_block_idx)",
    "self._load_cos_sin_K(mK, mCos, mSin, sCosK, sSinK, n_block_idx)"
)
# Note: in kernel, it passes m_cos instead of mCos, m_sin instead of mSin
content = content.replace(
    "self._load_cos_sin_K(mK, m_cos, m_sin, sCosK, sSinK, n_block)",
    "self._load_cos_sin_K(mK, m_cos, m_sin, sCosK, sSinK, n_block)"
)
# wait, in kernel it was self._load_cos_sin_K(basic_params, n_block)
content = content.replace(
    "self._load_cos_sin_K(basic_params, n_block)",
    "self._load_cos_sin_K(mK, m_cos, m_sin, sCosK, sSinK, n_block)"
)

# Replace SimpleNamespace blocks completely
content = re.sub(r"# Group params for compute_one_n_block\n\s*basic_params = SimpleNamespace\([^)]+\)\n", "", content)

# But wait, there might be missing definitions if we delete the blocks completely? No, they are already defined!
content = re.sub(r"\s*basic_params = SimpleNamespace\([\s\S]*?\s*sSinK=sSinK,\n\s*\)\n", "", content)
content = re.sub(r"\s*mma_params = SimpleNamespace\([\s\S]*?\s*acc_O=acc_O,\n\s*\)\n", "", content)
content = re.sub(r"\s*gmem_copy_params = SimpleNamespace\([\s\S]*?\s*tKVpKV=tKVpKV,\n\s*\)\n", "", content)
content = re.sub(r"\s*smem_copy_params = SimpleNamespace\([\s\S]*?\s*tOrVt_copy_view=tOrVt_copy_view,\n\s*\)\n", "", content)
content = re.sub(r"\s*softmax_params = SimpleNamespace\([\s\S]*?\s*softmax_scale_log2=softmax_scale_log2,\n\s*\)\n", "", content)

with open("flash_attn_rope_fused.py", "w") as f:
    f.write(content)
