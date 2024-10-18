from rouge import Rouge
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu_score(references, hypotheses):
    print("References:", references)
    print("Hypotheses:", hypotheses)
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print("BLEU score (sacrebleu):", bleu.score)
    return bleu.score

def calculate_bleu_score_nltk(references, hypotheses):
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])
    print("BLEU score (nltk):", bleu_score)
    return bleu_score

def calculate_rouge_scores(references, hypotheses):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    return scores