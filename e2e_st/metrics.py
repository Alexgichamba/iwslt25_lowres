import torchaudio.functional as taf
from typing import List, Dict, Union
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
             corpus_level: bool = True,
            ) -> Dict[str, Union[float, List[float]]]:
    """
    Evaluate the model performance with metrics for ASR and ST.
    
    Args: 
        st_target: List of target sentences for ST (Speech Translation)
        asr_target: List of target sentences for ASR (Automatic Speech Recognition)
        st_pred: List of predicted sentences for ST
        asr_pred: List of predicted sentences for ASR
        corpus_level: If True, compute metrics at the corpus level (dict of float)
                     If False, compute individual metrics for each sentence (dict of lists)
    Returns:
        results: Dictionary containing the evaluation metrics
    """
    results = {}
    
    if asr_target is not None and asr_pred is not None:
        if corpus_level:
            # Process ASR results at corpus level - calculate average WER and CER
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
        else:
            # Store individual WER and CER scores
            wer_scores = []
            cer_scores = []
            
            for i in range(len(asr_target)):
                reference = asr_target[i]
                hypothesis = asr_pred[i]
                
                # Calculate WER and CER
                metrics = compute_wer_cer(reference, hypothesis)
                wer_scores.append(metrics["wer"])
                cer_scores.append(metrics["cer"])
            
            results["wer"] = wer_scores
            results["cer"] = cer_scores
    
    if st_target is not None and st_pred is not None:
        if corpus_level:
            # Compute corpus-level BLEU and CHRF scores
            bleu = BLEU(lowercase=True)
            chrf = CHRF(lowercase=True, word_order=2) # chrf++
            bleu_score = bleu.corpus_score(st_pred, [st_target])
            chrf_score = chrf.corpus_score(st_pred, [st_target])
            results["bleu"] = bleu_score.score
            results["chrf"] = chrf_score.score
        else:
            # Compute sentence-level BLEU and CHRF scores
            bleu = BLEU(lowercase=True)
            chrf = CHRF(lowercase=True)
            
            bleu_scores = []
            chrf_scores = []
            
            for i in range(len(st_target)):
                # For sentence-level scores, we need to provide individual sentences
                # SacreBLEU expects references as a list of lists
                bleu_score = bleu.sentence_score(st_pred[i], [st_target[i]])
                chrf_score = chrf.sentence_score(st_pred[i], [st_target[i]])
                
                bleu_scores.append(bleu_score.score)
                chrf_scores.append(chrf_score.score)
            
            results["bleu"] = bleu_scores
            results["chrf"] = chrf_scores
            
    return results