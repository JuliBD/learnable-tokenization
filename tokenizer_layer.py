import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers import LlamaConfig
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
import torch
from torch import nn
# from transformers.models.llama import modeling_llama
#LlamaDecoderLayer = modeling_llama.LlamaDecoderLayer
from stablemax import StableMax
from vocab_manager import VocabManager, DynamicHead

# This uses next token probabilities for deciding the token boundaries
class TokenizerLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 layer_idx: int,
                 starting_vocab: list[tuple[int]],
                 backbone_len: int = 1,
                 embeder_len: int = 1,
                 combination_threshold = 0.95, # if the tokenizer head predicts the next token with a higher prob, combine this token with the next
                 stochastic_threshold = False, # if this is True, the combination threshold gets ignored. Each token will have a probability of p 
                                               # to be combined with the next, where p is the probability with which the tokenizer head predicted the
                                               # next token
                 padding_side = "left"
                 ):
        super().__init__()
        if embeder_len < 1:
            raise ValueError(f"embedder_len has to be 1 or greater, but was {embeder_len}")
        
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        self.hidden_size = config.hidden_size
        self.common_decoder_backbone = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(backbone_len)]
            )
        
        self.tokenizer_head = nn.Linear(in_features=self.hidden_size, out_features=len(starting_vocab))
        self.prob_fn = StableMax() # e.g SoftMax, StableMax ...

        self.decoder_embedder = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(embeder_len)]
            )
        self.pad_embed = nn.Embedding(1, self.hidden_size)
        
        self.vocab_manager = VocabManager(starting_vocab)
        self.combination_threshold = combination_threshold
        self.stochastic_threshold = stochastic_threshold
        self.pass_last_token = True
        self.padding_side = padding_side

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        batched_tokens: list[list[tuple[int]]], # batch len x sequence len | meta token id (defined by a tuple of all bytes it represents)
        attention_mask: torch.Tensor = None,
        ):
        batch_size, seq_len, _ = hidden_states.shape
        # TODO: have to fix position ids to fit to acutall sequences
        position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decode_layer in self.common_decoder_backbone:
            layer_output = decode_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings
                #attention_mask=attention_mask
                )
            hidden_states = layer_output[0]

        tokenizer_head_logits = self.tokenizer_head(hidden_states)
        self.last_head_results = tokenizer_head_logits

        for decode_layer in self.decoder_embedder:
            layer_output =  decode_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings
                #attention_mask=None#attention_mask
                )
            hidden_states = layer_output[0]

        probs = self.prob_fn(tokenizer_head_logits)
        self.newest_probs = probs

        next_token_idxs = []
        for sequence in batched_tokens:
            baNext_token_ids = sequence[1:] # starting from the second token
            baNext_token_idxs = torch.tensor(self.vocab_manager.ids2idxs(baNext_token_ids))#self.token_ids_to_idxs(baNext_token_ids)
            next_token_idxs.append(baNext_token_idxs)

        for sequence in batched_tokens:
            next_token_idxs.append(sequence[1:]) # starting from the second token

        probs_ = probs[:,:-1] # removing the probs for the last token, because there is no next token / the next token is unknown
        embedder_out = hidden_states

        embeddings_to_forward = []
        new_batched_tokens = []
        largest_number_out_tokens = 0

        for i, batch_element in enumerate(zip(next_token_idxs, embedder_out, batched_tokens)):
            baNext_token_idx, baLast_hidden, baTokens = batch_element

            baNext_token_probs = probs_[i,torch.arange(next_token_idxs[0].shape[0]), baNext_token_idx]

            if self.stochastic_threshold:
                baBelow_threshold = (baNext_token_probs < torch.rand_like(baNext_token_probs))
            else:
                baBelow_threshold = baNext_token_probs < self.combination_threshold
            
            baBelow_threshold = torch.cat(
                [baBelow_threshold,torch.tensor([self.pass_last_token])]
            )

            token_to_combine = ()
            combined_tokens = []
            for i, sequence_element in enumerate(zip(baBelow_threshold, baTokens)):
                below_theshold, token = sequence_element
                if below_theshold:
                    combined_tokens.append(token_to_combine + token)
                    token_to_combine = ()
                else:
                    token_to_combine += token
            new_batched_tokens.append(combined_tokens)

            baEmbeddings_to_forward = baLast_hidden[baBelow_threshold]
            embeddings_to_forward.append(baEmbeddings_to_forward)

            # determine length of longest batch_element after tokenization for padding
            number_out_tokens = baEmbeddings_to_forward.shape[0]
            if number_out_tokens > largest_number_out_tokens:
                largest_number_out_tokens = number_out_tokens

        pad_embed = torch.nn.Embedding(1,hidden_states.shape[-1])
        padding_vector = pad_embed(torch.tensor([0]))[0]
        padded_embeddings_to_forward = []
        # padding all batch_elements to same size, so the batch can be unified again
        for batch_element in embeddings_to_forward:
            number_out_tokens = batch_element.shape[0]
            if number_out_tokens < largest_number_out_tokens: #pad if necessary
                padding_vectors = [padding_vector]* (largest_number_out_tokens - number_out_tokens)
                padding_vectors = torch.stack(padding_vectors)
                if self.padding_side == "right":
                    padded_batch_element = torch.cat([batch_element, padding_vectors], dim=0)
                else: # pad left
                    padded_batch_element = torch.cat([padding_vectors, batch_element], dim=0)

                padded_embeddings_to_forward.append(padded_batch_element)
            else:
                padded_embeddings_to_forward.append(batch_element)
            
        unified_forward = torch.stack(padded_embeddings_to_forward)

        return (unified_forward, new_batched_tokens, attention_mask)
    

# This uses weighing of each embedding for deciding the token boundaries
class TokenizerLayerV2(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 layer_idx: int,
                 starting_vocab: list[tuple[int]],
                 common_backbone_len: int = 1,
                 weighing_backbone_len: int = 1,
                 embeder_len: int = 1,
                 passing_threshold = 0.5, # if the tokenizer head predicts the next token with a higher prob, combine this token with the next
                 stochastic_threshold = False, # if this is True, the combination threshold gets ignored. Each token will have a probability of p 
                                               # to be combined with the next, where p is the probability with which the tokenizer head predicted the
                                               # next token
                 padding_side = "left"
                 ):
        super().__init__()
        if embeder_len < 1:
            raise ValueError(f"embedder_len has to be 1 or greater, but was {embeder_len}")
        
        self.config = config
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        self.hidden_size = config.hidden_size
        self.common_decoder_backbone = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(common_backbone_len)]
            )
        
        self.embedding_weighing_backbone = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(weighing_backbone_len)]
            )
        self.sigmoid = nn.Sigmoid()

        self.decoder_embedder = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(embeder_len)]
            )
        self.pad_embed = nn.Embedding(1, self.hidden_size)
        
        self.vocab_manager = VocabManager(starting_vocab)
        self.passing_threshold = passing_threshold
        self.stochastic_threshold = stochastic_threshold
        self.pass_last_token = True
        self.padding_side = padding_side
        self.embedding_weights = None # later the weights will be saved here for testing

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        batched_tokens: list[list[tuple[int]]], # batch len x sequence len | meta token id (defined by a tuple of all bytes it represents)
        attention_mask: torch.Tensor = None,
        ):
        batch_size, seq_len, _ = hidden_states.shape
        # TODO: have to fix position ids to fit to acutal sequences
        position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        )
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]
        dtype, device = hidden_states.dtype, hidden_states.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, sequence_length), device=device)
        target_length = attention_mask.shape[-1]
        causal_mask = self._get_causal_mask(
            attention_mask, sequence_length, target_length, dtype, device, cache_position, batch_size
        )
        
        for decode_layer in self.common_decoder_backbone:
            layer_output = decode_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask
                )
            hidden_states = layer_output[0]
        common_hidden_states = hidden_states
        
        for decode_layer in self.embedding_weighing_backbone:
            layer_output = decode_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                #attention_mask=causal_mask
                )
            hidden_states = layer_output[0]

        # + 6 because the sigmoid of 6 is almost 1 and at initialization all weights should be close to 1
        unnormalized_weights = hidden_states.mean(-1, keepdim=True) + 6
        embedding_weights = self.sigmoid(unnormalized_weights) * attention_mask.unsqueeze(-1)
        # pad_tensor = torch.tensor([1 if self.pass_last_token else 0]).expand(embedding_weights.shape[0],1,1)
        # padded_weights = torch.cat([embedding_weights, pad_tensor], dim=1, device = device)
        # embedding_weights = padded_weights[:,1:] # look ahead 1 for 
        self.embedding_weights = embedding_weights
        noise = torch.rand_like(common_hidden_states)
        # scale common_hidden_states proportional to weights and add noise scaled inverse to the weights
        nomralized_common_hidden = nn.functional.normalize(common_hidden_states, dim=-1)
        hidden_states = nomralized_common_hidden * embedding_weights + noise * (1 - embedding_weights)

        for decode_layer in self.decoder_embedder:
            layer_output =  decode_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask
                )
            hidden_states = layer_output[0]

        embedder_out = hidden_states

        embeddings_to_forward = []
        new_batched_tokens = []
        largest_number_out_tokens = 0

        for i, batch_element in enumerate(zip(embedding_weights, embedder_out, batched_tokens)):
            baEmbedding_weights_, baLast_hidden, baTokens = batch_element
            baEmbedding_weights = baEmbedding_weights_.squeeze()

            if self.stochastic_threshold:
                baPast_threshold = (baEmbedding_weights > torch.rand_like(baEmbedding_weights))
            else:
                baPast_threshold = baEmbedding_weights > self.passing_threshold

            token_to_combine = ()
            combined_tokens = []
            for i, sequence_element in enumerate(zip(baPast_threshold, baTokens)):
                past_theshold, token = sequence_element
                if past_theshold:
                    combined_tokens.append(token_to_combine + token)
                    token_to_combine = ()
                else:
                    token_to_combine += token
            new_batched_tokens.append(combined_tokens)

            baEmbeddings_to_forward = baLast_hidden[baPast_threshold]
            embeddings_to_forward.append(baEmbeddings_to_forward)

            # determine length of longest batch_element after tokenization for padding
            number_out_tokens = baEmbeddings_to_forward.shape[0]
            if number_out_tokens > largest_number_out_tokens:
                largest_number_out_tokens = number_out_tokens
        
        # padding new_batched_tokens to same size
        for i in range(len(new_batched_tokens)):
            if self.padding_side == "left":
                new_batched_tokens[i] = [(self.config.pad_token_id,)] * (largest_number_out_tokens - len(new_batched_tokens[i])) + new_batched_tokens[i]
            else:
                new_batched_tokens[i] = new_batched_tokens[i] + [(self.config.pad_token_id,)] * (largest_number_out_tokens - len(new_batched_tokens[i]))


        padding_vector = self.pad_embed(torch.tensor([0], device=hidden_states.device))[0]
        #pad_embed = torch.nn.Embedding(1,hidden_states.shape[-1])
        #padding_vector = pad_embed(torch.tensor([0]))[0]
        padded_embeddings_to_forward = []
        # padding all batch_elements to same size, so the batch can be unified again
        for batch_element in embeddings_to_forward:
            number_out_tokens = batch_element.shape[0]
            if number_out_tokens < largest_number_out_tokens: #pad if necessary
                padding_vectors = [padding_vector]* (largest_number_out_tokens - number_out_tokens)
                padding_vectors = torch.stack(padding_vectors)
                if self.padding_side == "right":
                    padded_batch_element = torch.cat([batch_element, padding_vectors], dim=0)
                else: # pad left
                    padded_batch_element = torch.cat([padding_vectors, batch_element], dim=0)

                padded_embeddings_to_forward.append(padded_batch_element)
            else:
                padded_embeddings_to_forward.append(batch_element)
            
        unified_forward = torch.stack(padded_embeddings_to_forward)

        return (unified_forward, new_batched_tokens, attention_mask)

    @staticmethod
    def _get_causal_mask(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
                         ):
        
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask

    def loss(self):
        return self.embedding_weights.mean()