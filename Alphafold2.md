Alphafold2(
  (token_emb): Embedding(21, 256)
  (self_attn_rotary_emb): FixedPositionalEmbedding()
  (cross_attn_seq_rotary_emb): AxialRotaryEmbedding()
  (cross_attn_msa_rotary_emb): FixedPositionalEmbedding()
  (template_dist_emb): Embedding(37, 256)
  (template_num_pos_emb): Embedding(10, 256)
  (template_sidechain_emb): EnTransformer(
    (layers): ModuleList(
      (0): ModuleList(
        (0): None
        (1): Residual(
          (fn): PreNorm(
            (fn): EquivariantAttention(
              (to_qkv): Linear(in_features=256, out_features=768, bias=False)
              (to_out): Linear(in_features=256, out_features=256, bias=True)
              (coors_mlp): Sequential(
                (0): Linear(in_features=1, out_features=16, bias=True)
                (1): GELU()
                (2): Linear(in_features=16, out_features=1, bias=True)
              )
              (coors_gate): Sequential(
                (0): Linear(in_features=1, out_features=1, bias=True)
                (1): Tanh()
              )
              (norm_rel_coors): CoorsNorm()
              (rotary_emb): SinusoidalEmbeddings()
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=2048, bias=True)
                (1): GEGLU()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=1024, out_features=256, bias=True)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (1): ModuleList(
        (0): None
        (1): Residual(
          (fn): PreNorm(
            (fn): EquivariantAttention(
              (to_qkv): Linear(in_features=256, out_features=768, bias=False)
              (to_out): Linear(in_features=256, out_features=256, bias=True)
              (coors_mlp): Sequential(
                (0): Linear(in_features=1, out_features=16, bias=True)
                (1): GELU()
                (2): Linear(in_features=16, out_features=1, bias=True)
              )
              (coors_gate): Sequential(
                (0): Linear(in_features=1, out_features=1, bias=True)
                (1): Tanh()
              )
              (norm_rel_coors): CoorsNorm()
              (rotary_emb): SinusoidalEmbeddings()
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=2048, bias=True)
                (1): GEGLU()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=1024, out_features=256, bias=True)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (2): ModuleList(
        (0): None
        (1): Residual(
          (fn): PreNorm(
            (fn): EquivariantAttention(
              (to_qkv): Linear(in_features=256, out_features=768, bias=False)
              (to_out): Linear(in_features=256, out_features=256, bias=True)
              (coors_mlp): Sequential(
                (0): Linear(in_features=1, out_features=16, bias=True)
                (1): GELU()
                (2): Linear(in_features=16, out_features=1, bias=True)
              )
              (coors_gate): Sequential(
                (0): Linear(in_features=1, out_features=1, bias=True)
                (1): Tanh()
              )
              (norm_rel_coors): CoorsNorm()
              (rotary_emb): SinusoidalEmbeddings()
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=2048, bias=True)
                (1): GEGLU()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=1024, out_features=256, bias=True)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (3): ModuleList(
        (0): None
        (1): Residual(
          (fn): PreNorm(
            (fn): EquivariantAttention(
              (to_qkv): Linear(in_features=256, out_features=768, bias=False)
              (to_out): Linear(in_features=256, out_features=256, bias=True)
              (coors_mlp): Sequential(
                (0): Linear(in_features=1, out_features=16, bias=True)
                (1): GELU()
                (2): Linear(in_features=16, out_features=1, bias=True)
              )
              (coors_gate): Sequential(
                (0): Linear(in_features=1, out_features=1, bias=True)
                (1): Tanh()
              )
              (norm_rel_coors): CoorsNorm()
              (rotary_emb): SinusoidalEmbeddings()
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (2): Residual(
          (fn): PreNorm(
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=2048, bias=True)
                (1): GEGLU()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=1024, out_features=256, bias=True)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
  )
  (embedd_project): Linear(in_features=1280, out_features=256, bias=True)
  (net): SequentialSequence(
    (blocks): ModuleList(
      (0): ModuleList(
        (0): PreNorm(
          (fn): InterceptAxialAttention(
            (attn): AxialAttention(
              (attn_width): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (attn_height): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (attn_frames): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (1): PreNorm(
          (fn): InterceptFeedForward(
            (ff): LocalFeedForward(
              (net): Sequential(
                (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GELU()
                (2): DepthWiseConv2d(
                  (net): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                    (1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
                (3): GELU()
                (4): Dropout(p=0.0, inplace=False)
                (5): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (2): PreNorm(
          (fn): AxialAttention(
            (attn_width): Attention(
              (to_q): Linear(in_features=256, out_features=512, bias=False)
              (to_kv): Linear(in_features=256, out_features=1024, bias=False)
              (to_out): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (attn_height): Attention(
              (to_q): Linear(in_features=256, out_features=512, bias=False)
              (to_kv): Linear(in_features=256, out_features=1024, bias=False)
              (to_out): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (3): PreNorm(
          (fn): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): GEGLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=1024, out_features=256, bias=True)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): ModuleList(
        (0): InterceptAttention(
          (attn): PreNormCross(
            (fn): KronInputWrapper(
              (fn): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (1): PreNorm(
          (fn): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): GEGLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=1024, out_features=256, bias=True)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (2): InterceptAttention(
          (attn): PreNormCross(
            (fn): KronInputWrapper(
              (fn): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (3): PreNorm(
          (fn): InterceptFeedForward(
            (ff): LocalFeedForward(
              (net): Sequential(
                (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GELU()
                (2): DepthWiseConv2d(
                  (net): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                    (1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
                (3): GELU()
                (4): Dropout(p=0.0, inplace=False)
                (5): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): ModuleList(
        (0): PreNorm(
          (fn): InterceptAxialAttention(
            (attn): AxialAttention(
              (attn_width): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (attn_height): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (attn_frames): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (1): PreNorm(
          (fn): InterceptFeedForward(
            (ff): LocalFeedForward(
              (net): Sequential(
                (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GELU()
                (2): DepthWiseConv2d(
                  (net): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                    (1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
                (3): GELU()
                (4): Dropout(p=0.0, inplace=False)
                (5): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (2): PreNorm(
          (fn): AxialAttention(
            (attn_width): Attention(
              (to_q): Linear(in_features=256, out_features=512, bias=False)
              (to_kv): Linear(in_features=256, out_features=1024, bias=False)
              (to_out): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (attn_height): Attention(
              (to_q): Linear(in_features=256, out_features=512, bias=False)
              (to_kv): Linear(in_features=256, out_features=1024, bias=False)
              (to_out): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (3): PreNorm(
          (fn): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): GEGLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=1024, out_features=256, bias=True)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): ModuleList(
        (0): InterceptAttention(
          (attn): PreNormCross(
            (fn): KronInputWrapper(
              (fn): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (1): PreNorm(
          (fn): FeedForward(
            (net): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): GEGLU()
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=1024, out_features=256, bias=True)
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (2): InterceptAttention(
          (attn): PreNormCross(
            (fn): KronInputWrapper(
              (fn): Attention(
                (to_q): Linear(in_features=256, out_features=512, bias=False)
                (to_kv): Linear(in_features=256, out_features=1024, bias=False)
                (to_out): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
        (3): PreNorm(
          (fn): InterceptFeedForward(
            (ff): LocalFeedForward(
              (net): Sequential(
                (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): GELU()
                (2): DepthWiseConv2d(
                  (net): Sequential(
                    (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                    (1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
                  )
                )
                (3): GELU()
                (4): Dropout(p=0.0, inplace=False)
                (5): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (trunk_to_coords): CoordModuleMDS()
  (to_distogram_logits): Sequential(
    (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (1): Sequential(
      (0): Linear(in_features=256, out_features=2304, bias=True)
      (1): Rearrange('b h w c -> b c h w')
      (2): PixelShuffle(upscale_factor=3)
      (3): Rearrange('b c h w -> b h w c')
    )
    (2): Linear(in_features=256, out_features=37, bias=True)
  )
  (global_pool_attns): ModuleList()
  (trunk_to_structure_dim): Linear(in_features=256, out_features=4, bias=True)
  (lddt_linear): Linear(in_features=4, out_features=1, bias=True)
)