import math
import Levenshtein
import torchaudio

def normalized_levenshtein(s1: str, s2: str) -> float:
    # https://www.cse.lehigh.edu/%7Elopresti/Publications/1996/sdair96.pdf
    d = Levenshtein.distance(s1, s2)
    m = min(len(s1), len(s2))
    return 1.0 / math.exp(d / (m - d))


def test(decoder, audio_path, true_transcription) -> None:
    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Normalized Levenshtein distance: {normalized_levenshtein(true_transcription, transcript.strip())}")
