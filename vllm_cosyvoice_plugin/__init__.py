def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "CosyVoice2ForCausalLM",
        "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
    )
