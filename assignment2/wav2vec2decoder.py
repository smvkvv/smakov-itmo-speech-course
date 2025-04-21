from typing import List, Tuple
import kenlm
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
    ):
        """
        Initialization of Wav2Vec2Decoder class

        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def id_to_char(self, pred_id: int) -> str | None:
        return self.vocab.get(pred_id, self.vocab[3])

    def from_preds_to_str(self, pred_ids: list[int]) -> str:
        chars = list(filter(None, map(self.id_to_char, pred_ids)))

        transcript = []
        prev_char = None

        for char in chars:
            if char == self.vocab[self.blank_token_id]:
                prev_char = None
                continue
            if char != prev_char:
                transcript.append(char)
                prev_char = char

        return ''.join(transcript).replace(self.word_delimiter, ' ').strip()

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)

        Returns:
            str: Decoded transcript
        """
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        return self.from_preds_to_str(pred_ids)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring

        Returns:
            Union[str, List[Tuple[float, List[int]]]]:
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        timestamps = range(log_probs.shape[0])
        beams = [([], 0.0)]

        for t_logprobs, t in zip(log_probs, timestamps):
            new_beams = []

            for pred_ids, score in beams:
                topk_log_probs, topk_indices = torch.topk(t_logprobs, self.beam_width)
                for log_prob, token_idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_seq = pred_ids + [token_idx]
                    new_score = score + log_prob
                    new_beams.append((new_seq, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_width]

        beams = [(self.from_preds_to_str(pred_id), score) for pred_id, score in beams]
        best_hypothesis = beams[0][0]

        if return_beams:
            return beams
        else:
            return best_hypothesis

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size

        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        # <YOUR CODE GOES HERE>
        return

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs

        Args:
            beams (list): List of tuples (hypothesis, log_prob)

        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        rescored_beams = []
        for beam in beams:
            hypothesis, log_prob = beam
            lm_prob = self.lm_model.score(hypothesis)
            rescored_prob = log_prob + self.alpha * lm_prob
            rescored_beams.append((hypothesis, rescored_prob))

        best_hypothesis = max(rescored_beams, key=lambda x: x[1])[0]

        return ''.join(best_hypothesis).replace(self.word_delimiter, ' ')

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method

        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and
                      "beam_lm_rescore" is a beam search with second pass LM rescoring

        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")
