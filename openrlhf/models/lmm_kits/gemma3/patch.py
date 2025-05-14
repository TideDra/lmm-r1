from ..base.patch import BasePatch
import torch


class Gemma3_Patch(BasePatch):
    def _add_get_inputs_embeds():
        from transformers import Gemma3ForConditionalGeneration
        from transformers.utils import is_torchdynamo_compiling

        def get_inputs_embeds(self, input_ids, pixel_values=None, **kwargs):
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # Merge text and images
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values)

                if input_ids is None:
                    special_image_mask = inputs_embeds == self.get_input_embeddings()(
                        torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                    )
                else:
                    special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                if (
                    not is_torchdynamo_compiling()
                    and inputs_embeds[special_image_mask].numel() != image_features.numel()
                ):
                    image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                    raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                        "tokens from image embeddings."
                    )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            else:
                fake_pixel_values = torch.zeros(
                    (1, 3, 224, 224), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                image_features = self.get_image_features(fake_pixel_values)
                inputs_embeds = inputs_embeds + 0 * image_features.mean()
            return inputs_embeds

        Gemma3ForConditionalGeneration.get_inputs_embeds = get_inputs_embeds

    def _add_get_position_ids():
        from transformers import Gemma3ForConditionalGeneration

        def get_position_ids(self, input_ids, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids

        Gemma3ForConditionalGeneration.get_position_ids = get_position_ids

    def _add_offset_split_position_ids():
        from transformers import Gemma3ForConditionalGeneration

        def offset_split_position_ids(self, split_position_ids, hacked_position_ids):
            # For common position_ids, hacked_position_ids is what we want
            return hacked_position_ids

        Gemma3ForConditionalGeneration.offset_split_position_ids = offset_split_position_ids

    def apply_liger_kernel(self):
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3

        apply_liger_kernel_to_gemma3()

    @classmethod
    def _load_all_patches(cls):
        cls._add_get_inputs_embeds()
        cls._add_get_position_ids()
        cls._add_offset_split_position_ids()


Patch = Gemma3_Patch()
