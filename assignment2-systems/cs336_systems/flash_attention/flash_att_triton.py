import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_k = tl.arange(0, K_TILE_SIZE)
    q_mask = offs_q < N_QUERIES

    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))

    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        key_offsets = key_tile_index * K_TILE_SIZE + offs_k
        k_mask = key_offsets < N_KEYS
        K_block_ptr = tl.make_block_ptr(
            K_ptr + stride_kb * batch_index,
            shape=(D, N_KEYS),
            strides=(stride_kd, stride_kk),
            offsets=(0, key_tile_index * K_TILE_SIZE),
            block_shape=(D, K_TILE_SIZE),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + stride_vb * batch_index,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        K_j = tl.load(K_block_ptr, boundary_check=(1, 0))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))

        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scale
        valid = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            valid = valid & (offs_q[:, None] >= key_offsets[None, :])
        S_ij = tl.where(valid, S_ij, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        m_scaler = tl.exp(m_i - m_i_new)
        l_i = m_scaler * l_i + tl.sum(P_ij, 1)
        o_i = o_i * m_scaler[:, None] + tl.dot(P_ij.to(tl.float32), V_j.to(tl.float32))
        m_i = m_i_new

    o_i = o_i * (1.0 / l_i[:, None])
    l_i = tl.log(l_i) + m_i

    tl.store(O_block_ptr, o_i.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i.to(L_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_k = tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < N_QUERIES

    Q_i = tl.load(
        Q_ptr + batch_index * stride_qb + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )
    O_i = tl.load(
        O_ptr + batch_index * stride_ob + offs_q[:, None] * stride_oq + offs_d[None, :] * stride_od,
        mask=q_mask[:, None],
        other=0.0,
    )
    dO_i = tl.load(
        dO_ptr + batch_index * stride_dob + offs_q[:, None] * stride_doq + offs_d[None, :] * stride_dod,
        mask=q_mask[:, None],
        other=0.0,
    )
    L_i = tl.load(L_ptr + batch_index * stride_lb + offs_q * stride_lq, mask=q_mask, other=0.0)
    delta_i = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        key_offsets = key_tile_index * K_TILE_SIZE + offs_k
        k_mask = key_offsets < N_KEYS
        K_j_t = tl.load(
            K_ptr + batch_index * stride_kb + offs_d[:, None] * stride_kd + key_offsets[None, :] * stride_kk,
            mask=k_mask[None, :],
            other=0.0,
        )
        V_j = tl.load(
            V_ptr + batch_index * stride_vb + key_offsets[:, None] * stride_vk + offs_d[None, :] * stride_vd,
            mask=k_mask[:, None],
            other=0.0,
        )
        S_ij = tl.dot(Q_i.to(tl.float32), K_j_t.to(tl.float32)) * scale
        valid = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            valid = valid & (offs_q[:, None] >= key_offsets[None, :])
        S_ij = tl.where(valid, S_ij, float("-inf"))
        P_ij = tl.exp(S_ij - L_i[:, None])
        dP_ij = tl.dot(dO_i.to(tl.float32), tl.trans(V_j.to(tl.float32)))
        dS_ij = P_ij * (dP_ij - delta_i[:, None]) * scale
        dQ_i += tl.dot(dS_ij.to(tl.float32), tl.trans(K_j_t.to(tl.float32)))

    tl.store(
        dQ_ptr + batch_index * stride_dqb + offs_q[:, None] * stride_dqq + offs_d[None, :] * stride_dqd,
        dQ_i,
        mask=q_mask[:, None],
    )


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    dO_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    offs_q = tl.arange(0, Q_TILE_SIZE)
    offs_k = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)
    k_mask = offs_k < N_KEYS

    K_j = tl.load(
        K_ptr + batch_index * stride_kb + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd,
        mask=k_mask[:, None],
        other=0.0,
    )
    V_j = tl.load(
        V_ptr + batch_index * stride_vb + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd,
        mask=k_mask[:, None],
        other=0.0,
    )
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for query_tile_index in range(0, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        query_offsets = query_tile_index * Q_TILE_SIZE + offs_q
        q_mask = query_offsets < N_QUERIES
        Q_i = tl.load(
            Q_ptr + batch_index * stride_qb + query_offsets[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=q_mask[:, None],
            other=0.0,
        )
        O_i = tl.load(
            O_ptr + batch_index * stride_ob + query_offsets[:, None] * stride_oq + offs_d[None, :] * stride_od,
            mask=q_mask[:, None],
            other=0.0,
        )
        dO_i = tl.load(
            dO_ptr + batch_index * stride_dob + query_offsets[:, None] * stride_doq + offs_d[None, :] * stride_dod,
            mask=q_mask[:, None],
            other=0.0,
        )
        L_i = tl.load(L_ptr + batch_index * stride_lb + query_offsets * stride_lq, mask=q_mask, other=0.0)
        delta_i = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
        S_ij = tl.dot(Q_i.to(tl.float32), tl.trans(K_j.to(tl.float32))) * scale
        valid = q_mask[:, None] & k_mask[None, :]
        if is_causal:
            valid = valid & (query_offsets[:, None] >= offs_k[None, :])
        S_ij = tl.where(valid, S_ij, float("-inf"))
        P_ij = tl.exp(S_ij - L_i[:, None])
        dV_j += tl.dot(tl.trans(P_ij.to(tl.float32)), dO_i.to(tl.float32))
        dP_ij = tl.dot(dO_i.to(tl.float32), tl.trans(V_j.to(tl.float32)))
        dS_ij = P_ij * (dP_ij - delta_i[:, None]) * scale
        dK_j += tl.dot(tl.trans(dS_ij.to(tl.float32)), Q_i.to(tl.float32))

    tl.store(
        dK_ptr + batch_index * stride_dkb + offs_k[:, None] * stride_dkk + offs_d[None, :] * stride_dkd,
        dK_j,
        mask=k_mask[:, None],
    )
    tl.store(
        dV_ptr + batch_index * stride_dvb + offs_k[:, None] * stride_dvk + offs_d[None, :] * stride_dvd,
        dV_j,
        mask=k_mask[:, None],
    )


def _tile_sizes(D: int) -> tuple[int, int]:
    if D <= 32:
        return 128, 128
    if D <= 64:
        return 64, 64
    return 32, 32


def flash_backward_impl(Q, K, V, output, L, dO, is_causal):
    batch_size, N_q, D = Q.shape
    _, N_k, _ = K.shape
    Q_TILE_SIZE, K_TILE_SIZE = (16, 16) if D >= 128 else (32, 32)
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    scale = D**-0.5
    T_q = math.ceil(N_q / Q_TILE_SIZE)
    T_k = math.ceil(N_k / K_TILE_SIZE)

    flash_bwd_dq_kernel[(T_q, batch_size)](
        Q,
        K,
        V,
        output,
        L,
        dO,
        dQ,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        L.stride(0),
        L.stride(1),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
        N_q,
        N_k,
        scale,
        D,
        Q_TILE_SIZE,
        K_TILE_SIZE,
        is_causal,
    )
    flash_bwd_dkdv_kernel[(T_k, batch_size)](
        Q,
        K,
        V,
        output,
        L,
        dO,
        dK,
        dV,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        L.stride(0),
        L.stride(1),
        dO.stride(0),
        dO.stride(1),
        dO.stride(2),
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dV.stride(0),
        dV.stride(1),
        dV.stride(2),
        N_q,
        N_k,
        scale,
        D,
        Q_TILE_SIZE,
        K_TILE_SIZE,
        is_causal,
    )
    return dQ, dK, dV


class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, N_q, D = q.shape
        _, N_k, _ = k.shape
        Q_TILE_SIZE, K_TILE_SIZE = _tile_sizes(D)
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        O_ptr = torch.empty_like(q)
        L_ptr = torch.empty((batch_size, N_q), dtype=torch.float32, device=q.device)

        flash_fwd_kernel[(T_q, batch_size)](
            q,
            k,
            v,
            O_ptr,
            L_ptr,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            O_ptr.stride(0),
            O_ptr.stride(1),
            O_ptr.stride(2),
            L_ptr.stride(0),
            L_ptr.stride(1),
            N_q,
            N_k,
            D**-0.5,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        ctx.save_for_backward(q, k, v, O_ptr, L_ptr)
        ctx.is_causal = is_causal
        return O_ptr

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, output, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_impl(q, k, v, output, L, grad_out, ctx.is_causal)
        return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype), None
