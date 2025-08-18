from transformers import PretrainedConfig

class HRMText1Config(PretrainedConfig):
    model_type = "hrm_text1"
    
    def __init__(
        self,
        vocab_size=32100,
        n_embd=1024,
        n_head=16,
        block_size=512,
        d_ff=4096,
        dropout=0.1,
        halt_max_steps=8,
        halt_bias_init=-2.2,
        ponder_loss_weight=0.01,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.d_ff = d_ff
        self.dropout = dropout
        self.halt_max_steps = halt_max_steps
        self.halt_bias_init = halt_bias_init
        self.ponder_loss_weight = ponder_loss_weight
        
        super().__init__(**kwargs)