import torch
from torch import nn
import torch.nn.functional as F
from _transformers.utils import embedding, EmbeddingLayer, Linear
from _transformers.decoder import Decoder
from _transformers.encoder import Encoder
from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor
from transformers import BeamScorer, StoppingCriteriaList, MaxLengthCriteria
from transformers.utils import ModelOutput
# from transformers.generation_utils import BeamSearchOutput
from torch import nn
import torch.distributed as dist
from typing import Optional, Union, List, Dict, Tuple, Any

# from transformers.generation_utils import beam_search
from transformers import GenerationMixin

class StyleTransformer(nn.Module,GenerationMixin):
    def __init__(self, vocab_size, num_style_embeds, num_enc_layers, num_dec_layers, model_dim, num_heads, dropout, max_allowed_length, batch_size=16, padding_idx=99):
        super().__init__()
        self.allowed_seq_len = max_allowed_length
        self.style_embeds = embedding(num_style_embeds, model_dim)
        self.embed = EmbeddingLayer(vocab_size, model_dim, max_allowed_length, padding_idx=padding_idx)
        self.encoder = Encoder(
            num_enc_layers=num_enc_layers,
            hidden_dim=model_dim,
            num_heads=num_heads,
            dropout_ratio=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_dec_layers,
            dropout_ratio=dropout,
        )
        # self.sos_token = torch.nn.Parameter(torch.randn(model_dim))
        self.sos_id = 2

        self.scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_allowed_length//2,
            num_beams=4,
            device='cuda'
        )
        self.logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(2, eos_token_id=1)])
        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_allowed_length//2)])

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            src_mask ,
            tgt_mask, 
            memory,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = 2,
            sos_token_id : Optional[int] = 0,
            eos_token_id: Optional[Union[int, List[int]]] = 3,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            output_scores: Optional[bool] = True,
            return_dict_in_generate: Optional[bool] = False,
            synced_gpus: bool = False,
            **model_kwargs,
        ): #-> Union[BeamSearchOutput, torch.LongTensor]:
            
            # init values
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            # output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
            # output_attentions = (
            #     output_attentions if output_attentions is not None else self.generation_config.output_attentions
            # )
            # output_hidden_states = (
            #     output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            # )
            # return_dict_in_generate = (
            #     return_dict_in_generate
            #     if return_dict_in_generate is not None
            #     else self.generation_config.return_dict_in_generate
            # )
            batch_size = len(beam_scorer._beam_hyps)
            num_beams = beam_scorer.num_beams

            # repeat the input_ids batch_size*num_beams
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).contiguous().view(
                batch_size * num_beams, -1
            ).long()
            pos_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).expand((batch_size * num_beams, -1)).to(input_ids.device)
            input_embeds = self.embed(input_ids, pos_ids) # (batch_size * num_beams, cur_len, model_dim)
            memory = memory.unsqueeze(0).repeat(num_beams, 1, 1, 1).view(-1, memory.shape[1], memory.shape[2])
            src_mask = src_mask.unsqueeze(0).repeat(num_beams, 1, 1, 1, 1).view(-1, src_mask.shape[1], src_mask.shape[2], src_mask.shape[3])
            # tgt_mask = tgt_mask.unsqueeze(0).repeat(num_beams, 1, 1, 1, 1).view(-1, tgt_mask.shape[1], tgt_mask.shape[2], tgt_mask.shape[3])

            batch_beam_size, _ = input_ids.shape
            if num_beams * batch_size != batch_beam_size:
                raise ValueError(
                    f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
                )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            beam_indices = (
                tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
            )
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate :
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
            # of the first beam are considered to avoid sampling the exact same tokens across all beams.
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            cur_len = 0
            while True:      
                # create a tgt mask
                # tgt_mask = torch.ones((cur_len + 1, cur_len + 1)).to(input_ids.device)

                outputs = self.decoder(
                    input_embeds, 
                    memory, 
                    tgt_mask[:, :, cur_len:cur_len+1, :cur_len+1],
                    src_mask,
                    1.0
                )

                next_token_logits = outputs[:, -1, :]
                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)

                next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores_processed,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # reshape for beam search
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
                n_eos_tokens = len(eos_token_id) if eos_token_id else 0
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size)
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=beam_indices,
                )

                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                pos_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).expand((batch_size * num_beams, -1)).to(input_ids.device)
                input_embeds = self.embed(input_ids, pos_ids)
                
                # change it to ModelOutput format
                outputs = ModelOutput(
                    last_hidden_state=outputs,
                    past_key_values=None,
                    decoder_hidden_states=decoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                )

                # input_ids = torch.cat([input_ids_main[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=None#self.config.is_encoder_decoder
                )
                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

                if return_dict_in_generate and output_scores:
                    beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

                # increase cur_len
                cur_len = cur_len + 1

                if beam_scorer.is_done or self.stopping_criteria(input_ids, scores):
                    if not synced_gpus:
                        break
                    else:
                        this_peer_finished = True

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=self.stopping_criteria.max_length,
                beam_indices=beam_indices,
            )

            # if return_dict_in_generate:
            #     if not output_scores:
            #         sequence_outputs["sequence_scores"] = None

            #     if self.config.is_encoder_decoder:
            #         return BeamSearchEncoderDecoderOutput(
            #             sequences=sequence_outputs["sequences"],
            #             sequences_scores=sequence_outputs["sequence_scores"],
            #             scores=scores,
            #             beam_indices=sequence_outputs["beam_indices"],
            #             encoder_attentions=encoder_attentions,
            #             encoder_hidden_states=encoder_hidden_states,
            #             decoder_attentions=decoder_attentions,
            #             cross_attentions=cross_attentions,
            #             decoder_hidden_states=decoder_hidden_states,
            #         )
            #     else:
            #         return BeamSearchDecoderOnlyOutput(
            #             sequences=sequence_outputs["sequences"],
            #             sequences_scores=sequence_outputs["sequence_scores"],
            #             scores=scores,
            #             beam_indices=sequence_outputs["beam_indices"],
            #             attentions=decoder_attentions,
            #             hidden_states=decoder_hidden_states,
            #         )
            # else:
            return sequence_outputs["sequences"]


    def forward(self, src, tgt, inp_lengths, style, temperature, generate=False, differentiable_decode=False):
        batch_size = src.shape[0]
        max_seq_len = src.shape[1]
        
        assert max_seq_len <= self.allowed_seq_len, f"Max sequence length {max_seq_len} is greater than allowed sequence length {self.allowed_seq_len}"

        pos_idx = torch.arange(self.allowed_seq_len).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(src.device)

        src_mask = pos_idx[:, :max_seq_len] >= inp_lengths.unsqueeze(-1)
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), dim=1).view(batch_size, 1, 1, max_seq_len+1)

        style_embedding = self.style_embeds(style).view(batch_size, 1, -1)
        enc = torch.cat((style_embedding, self.embed(src, pos_idx[:, :max_seq_len])), dim=1)
        memory=self.encoder(enc, src_mask)

        # sos_token = self.sos_token.view(1, 1, -1).expand((batch_size, -1, -1))
        tgt_mask = torch.ones((self.allowed_seq_len, self.allowed_seq_len)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.allowed_seq_len, self.allowed_seq_len)
        
        if not generate:
            max_dec_length = tgt.shape[1]
            # decoder_embeds = torch.cat((self.embed(dec_input, pos_idx[:, :max_dec_length])), dim=1)
            decoder_embeds = self.embed(tgt, pos_idx[:, :max_dec_length])
            logits = self.decoder(
                decoder_embeds, 
                memory, 
                tgt_mask[:, :, :max_dec_length, :max_dec_length], 
                src_mask,
                temperature
            )
            return logits
        else:
            sos_token = (torch.ones((batch_size, 1)) * self.sos_id).to(src.device).long()

            logits = []
            next_token = sos_token.to(src.device)
            prev_states = None

            # beam search
            output_seq = self.beam_search(
                input_ids = next_token,
                src_mask = src_mask,
                tgt_mask = tgt_mask,
                memory = memory,
                beam_scorer = self.scorer,
                logits_processor = self.logits_processor,
                max_length = self.allowed_seq_len//2,
            )
            # print(output_seq)
            # for i in range(self.allowed_seq_len):
            #     logit, prev_states = self.decoder.regressive_generate(
            #         next_token,
            #         memory,
            #         src_mask,
            #         tgt_mask[:, :, i:i+1, :i+1],
            #         temperature,
            #         prev_states
            #     )
            #     logits.append(logit)
            #     if differentiable_decode:
            #         next_token = self.embed(logit.exp(), pos_idx[:, i:i+1])
            #     else:
            #         next_token = self.embed(logit.argmax(-1), pos_idx[:, i:i+1])
            #     # next_token = torch.cat((next_token, self.embed(logit.argmax(-1), pos_idx[:, i:i+1])), dim=1)
            # logits = torch.cat(logits, 1)


            return output_seq

class Discriminator(nn.Module):
    def __init__(
            self, 
        vocab_size, num_style_embeds, num_enc_layers, 
        model_dim, num_heads, dropout, max_allowed_length, num_classes, 
        padding_idx=99
    ):
        super().__init__()
        self.allowed_seq_len = max_allowed_length
        self.style_embeds = embedding(num_style_embeds, model_dim)
        self.embed = EmbeddingLayer(vocab_size, model_dim, max_allowed_length, padding_idx=padding_idx)
        self.encoder = Encoder(
            num_enc_layers=num_enc_layers,
            hidden_dim=model_dim,
            num_heads=num_heads,
            dropout_ratio=dropout
        )  
        self.cls_token = nn.Parameter(torch.rand(model_dim))
        self.cls_head = Linear(model_dim, num_classes)

    def forward(self, src, input_lengths, style_ids=None):
        batch_size = src.shape[0]
        max_seq_len = src.shape[1]
        num_extra_tokens = 1 if style_ids is None else 2
        pos_ids = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(src.device)

        src_mask = pos_ids>=input_lengths.unsqueeze(-1)

        for _ in range(num_extra_tokens):
            src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), dim=1)
        src_mask = src_mask.view(batch_size, 1, 1, max_seq_len+num_extra_tokens)

        cls_token = self.cls_token.view(1, 1, -1).expand((batch_size, -1, -1))

        if style_ids is not None:
            style_emb = self.style_embeds(style_ids).unsqueeze(1)
            enc = torch.cat((cls_token, style_emb), dim=1)
        enc = torch.cat((enc, self.embed(src, pos_ids)), dim=1)
        enc = self.encoder(enc, src_mask)
        logits = self.cls_head(enc[:, 0])

        return F.log_softmax(logits, -1)


# if __name__ == '__main__':
#     disc = Discriminator(100, 2, 4, 128, 4, 0.1, 300, 2)
#     src = torch.randint(0, 100, (2, 10))
#     inp_tokens = torch.randint(0, 200, size=(2, ))
#     style_ids = torch.tensor([0, 1])
#     a = disc(
#         src, inp_tokens, style_ids
#     )
#     print(a.shape)

if __name__ == "__main__":
    # dummy data
    model = StyleTransformer(400, 300, 2, 4, 4, 128, 4, 0.1, 10, )
    # dec2 = Decoder2(4, 128, 100, 4, 0.1)
    src = torch.randint(0, 100, (2, 10))
    tgt = torch.randint(0, 100, (2, 10))
    src_mask = torch.randint(0, 2, (2, 10)).float()
    tgt_mask = torch.randint(0, 2, (2, 1)).float()
    gen_text = model(
        src, 
        tgt, 
        src_mask, 
        torch.tensor([0, 1]), 
        generate=True
    )
    print(gen_text)