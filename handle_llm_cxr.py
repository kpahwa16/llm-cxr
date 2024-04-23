import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import logging
import re
from typing import List, Tuple
import torch
from torch.nn.functional import softmax
import numpy as np
import subprocess
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from .consts import END_KEY, PROMPT_FOR_GENERATION_FORMAT, PROMPT_FOR_GENERATION_FORMAT_NOINPUT, RESPONSE_KEY
import os

print(os.environ['PATH'])
# Configure transformers to reduce unnecessary log messages
# logging.set_verbosity_error()

def load_model_and_tokenizer(model_path):
    """Load a pretrained model and tokenizer based on a given path."""
    try:
        print("inside handle script, model path:", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error status if loading fails


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        ValueError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self, *args, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs
    ):
        """Initialize the pipeline

        Args:
            do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        super().__init__(*args, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                         **kwargs)

    def _sanitize_parameters(self,
                             return_full_text: bool = None,
                             **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id
        }

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        instruction_text, input_text = instruction_text
        if input_text is None:
            prompt_text = PROMPT_FOR_GENERATION_FORMAT_NOINPUT.format(instruction=instruction_text)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
            )
        elif type(input_text) is str:
            prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text, input=input_text)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
            )
        elif type(input_text) is list:
            prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text, input=CXR_VQ_VQ_REPLACE_TEMPLATE)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
            )

            assert len(inputs["input_ids"]) == 1
            assert len(inputs["attention_mask"]) == 1
            assert torch.all(inputs["attention_mask"])
            inputs["input_ids"] = torch.tensor(get_inject_vq_fun(self.tokenizer)(inputs["input_ids"][0].numpy().tolist(), input_text))[None, ...]
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
    
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
    
        outputs = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
            output_scores=True,  # Request scores to compute probabilities
            return_dict_in_generate=True,  # Ensure outputs are returned in a dict
            **generate_kwargs,
        )
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )   
        input_length = 1 if self.model.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        # breakpoint()
        print("generated_tokens", generated_tokens)
        print("transition_scores", transition_scores)
        print("generated_sequence", outputs.sequences)

        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")
            # corrected above line of code
        
        for score in outputs.scores:
            if torch.isinf(score).any() or torch.isnan(score).any():
                print("Warning: Inf or NaN values in scores tensor detected.")

    
        # Calculate probabilities from scores
        #probabilities = [softmax(scores, dim=-1) for scores in outputs.scores]
        # print("outputs.scores:")
        # print(outputs.scores)
     
        probabilities = [softmax(score, dim=-1) for score in outputs.scores]
        generated_tokens = outputs.sequences[:, len(input_ids[0]):] if input_ids is not None else outputs.sequences
        words = []
        word_probabilities = []
        current_word = ""
        current_probs = []

        for i, token_id in enumerate(generated_tokens.squeeze()):
            token = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if token.startswith(" ") or i == len(generated_tokens.squeeze()) - 1:
                if current_word:  # Save the previous word and its average probability
                    words.append(current_word.strip())
                    word_probabilities.append(np.mean(current_probs))
                current_word = token
                current_probs = [probabilities[i].max().item()]
            else:
                current_word += token
                current_probs.append(probabilities[i].max().item())

        print(probabilities)
        # token_probabilities = []
        generated_sequence = outputs.sequences
    
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
    
        # Extract vq if necessary, adjust based on your needs
        vq = get_extract_vq_fun(self.tokenizer)(generated_sequence)
    
        instruction_text = model_inputs.pop("instruction_text")
        
        return {
            "generated_sequence": generated_sequence,
            "words": words,
            "word_probabilities": word_probabilities,
            "input_ids": input_ids,
            "instruction_text": instruction_text # Include token probabilities here
            }

    def postprocess(self, model_outputs, response_key_token_id, end_key_token_id, return_full_text: bool = False):

            generated_sequence = model_outputs["generated_sequence"][0]
            # generated_vq = model_outputs["generated_vq"]
            instruction_text = model_outputs["instruction_text"]
            words = model_outputs["words"]
            word_probabilities = model_outputs["word_probabilities"]
            generated_text = " ".join(words)
            generated_sequence: List[List[int]] = generated_sequence.numpy().tolist()
            records = []
            for sequence in generated_sequence:

                # The response will be set to this variable if we can identify it.
                decoded = None

                # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
                if response_key_token_id and end_key_token_id:
                    # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                    # prompt, we should definitely find it.  We will return the tokens found after this token.
                    try:
                        response_pos = sequence.index(response_key_token_id)
                    except ValueError:
                        logger.warn(f"Could not find response key {response_key_token_id} in: {sequence}")
                        response_pos = None

                    if response_pos:
                        # Next find where "### End" is located.  The model has been trained to end its responses with this
                        # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                        # this token, as the response could be truncated.  If we don't find it then just return everything
                        # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                        try:
                            end_pos = sequence.index(end_key_token_id)
                        except ValueError:
                            end_pos = None

                        decoded = self.tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()

                if not decoded:
                    # Otherwise we'll decode everything and use a regex to find the response and end.

                    fully_decoded = self.tokenizer.decode(sequence)

                    # The response appears after "### Response:".  The model has been trained to append "### End" at the
                    # end.
                    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

                    if m:
                        decoded = m.group(1).strip()
                    else:
                        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                        # return everything after "### Response:".
                        m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                        if m:
                            decoded = m.group(1).strip()
                        else:
                            logger.warn(f"Failed to find response in:\n{fully_decoded}")

                # If the full text is requested, then append the decoded text to the original instruction.
                # This technically isn't the full text, as we format the instruction in the prompt the model has been
                # trained on, but to the client it will appear to be the full text.
                if return_full_text:
                    decoded = f"{instruction_text}\n{decoded}"

                rec = {"generated_text": decoded, "words": words, "word_probabilities": word_probabilities}
                print("rec_postprocess:")
                print(rec)
                records.append(rec)

            return records   
    


def generate_response(
    instruction: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> str:
    """Given an instruction, uses the model and tokenizer to generate a response.  This formats the instruction in
    the instruction format that the model was fine-tuned on.

    Args:
        instruction (str): _description_
        model (PreTrainedModel): the model to use
        tokenizer (PreTrainedTokenizer): the tokenizer to use

    Returns:
        str: response
    """

    generation_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer, **kwargs)
    results = generation_pipeline(instruction)

    if not results or not isinstance(results, list) or not isinstance(results[0], dict):
        print("No results were generated by the pipeline or output format is incorrect.")
        return "No output generated.", [], []
    out = results[0]
    generated_text =  out.get("generated_text", "")
    words = out.get("words", [])
    word_probabilities = out.get("word_probabilities", [])
    return generated_text, words, word_probabilities

def main():
    if len(sys.argv) < 2:
        print("Usage: python handle_llm_cxr.py '{\"llm_model_path\": \"path/to/model\", \"instruction_text\": \"optional\", \"input_text\": \"optional\"}'", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print("Invalid JSON input.", file=sys.stderr)
        sys.exit(1)

    model_path = config.get('llm_model_path')
    if not model_path:
        print("Model path must be specified in the JSON input.", file=sys.stderr)
        sys.exit(1)

    instruction_text = config.get('instruction_text', 'Generate medical report based on the given X-ray image.')
    input_text = config.get('input_text', '')  # Default to empty string if not specified
    print("inside handle.py")
    model, tokenizer = load_model_and_tokenizer(model_path)
    print("inside handle.py")
    generated_text, words, word_probabilities = generate_response(
        instruction_text,
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=128
    )
    print("generated_text", generated_text)
    print("words", words)
    print("word_probabilities", word_probabilities)
    return generated_text, words, word_probabilities

if __name__ == "__main__":
    main()
