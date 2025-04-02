import torchaudio.functional as taf
from typing import List, Dict
from sacrebleu.metrics import BLEU, CHRF

def compute_wer_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) using torchaudio.
    
    Args:
        reference: Reference transcript (ground truth)
        hypothesis: Hypothesis transcript (prediction)
        
    Returns:
        Dictionary containing WER and CER values
    """
    # Tokenize to words for WER
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate word edit distance and WER
    word_edits = taf.edit_distance(ref_words, hyp_words)
    wer = word_edits / len(ref_words) if ref_words else 0
    
    # Tokenize to characters for CER
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    
    # Calculate character edit distance and CER
    char_edits = taf.edit_distance(ref_chars, hyp_chars)
    cer = char_edits / len(ref_chars) if ref_chars else 0
    
    return {"wer": wer, "cer": cer}


def compute_metrics(st_target: List[str] = None,
             asr_target: List[str] = None,
             st_pred: List[str] = None,
             asr_pred: List[str] = None,
            ) -> Dict[str, float]:
    """
    Evaluate the model performance:
    - WER and CER for ASR
    
    Args: 
        st_target: List of target sentences for ST (Speech Translation)
        asr_target: List of target sentences for ASR (Automatic Speech Recognition)
        st_pred: List of predicted sentences for ST
        asr_pred: List of predicted sentences for ASR
    Returns:
        results: Dictionary containing the evaluation metrics
    """
    results = {}
    
    if asr_target is not None and asr_pred is not None:
        # Process ASR results - calculate WER and CER
        asr_wer_sum = 0.0
        asr_cer_sum = 0.0
        
        for i in range(len(asr_target)):
            reference = asr_target[i]
            hypothesis = asr_pred[i]
            
            # Calculate WER and CER
            metrics = compute_wer_cer(reference, hypothesis)
            asr_wer_sum += metrics["wer"]
            asr_cer_sum += metrics["cer"]
        
        # Calculate averages
        if len(asr_target) > 0:
            results["wer"] = asr_wer_sum / len(asr_target)
            results["cer"] = asr_cer_sum / len(asr_target)
        else:
            results["wer"] = 0.0
            results["cer"] = 0.0
    
    if st_target is not None and st_pred is not None:   
        bleu = BLEU(lowercase=True)
        chrf = CHRF(lowercase=True)
        bleu = bleu.corpus_score(st_pred, [st_target])
        chrf = chrf.corpus_score(st_pred, [st_target])
        results["bleu"] = bleu.score
        results["chrf"] = chrf.score
    return results